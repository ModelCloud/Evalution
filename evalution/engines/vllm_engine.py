# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
import os
import sys
import threading
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from itertools import chain
from pathlib import Path
from typing import Any

from evalution.config import Model
from evalution.engines.base import (
    BaseInferenceSession,
    GenerationOutput,
    GenerationRequest,
    LoglikelihoodOutput,
    LoglikelihoodRequest,
    RollingLoglikelihoodOutput,
    RollingLoglikelihoodRequest,
    SharedEngineConfig,
)
from evalution.engines.transformers_common import (
    _friendly_batch_size,
    _load_tokenizer_from_model,
    _normalize_batch_size,
    _resolve_tokenizer_source,
    _truncate_at_stop,
)

_AUTO_BATCH_SIZE = "auto"
_DEFAULT_VLLM_CHECKOUT_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "vllm",
    Path.cwd() / "vllm",
    Path.cwd().parent / "vllm",
)


@dataclass(slots=True)
class VLLM(SharedEngineConfig):
    """Configure Evalution to run generation and scoring through vLLM."""

    # This engine intentionally models vLLM as a first-class Evalution backend
    # instead of routing through GPTQModel or the legacy TransformersCompat path.
    # That lets us preserve vLLM-specific behavior such as request-id based
    # continuous batching, prompt_logprobs scoring, and local checkout loading.
    tokenizer_mode: str = "auto"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    quantization: str | None = None
    max_model_len: int | None = None
    enforce_eager: bool = False
    tokenizer_revision: str | None = None
    vllm_path: str | None = None
    llm_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, model: Model) -> BaseInferenceSession:
        """Construct a vLLM-backed inference session for the requested model."""

        self.resolved_engine = "VLLM"
        return VLLMSession.from_config(self, model)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the engine configuration for reporting and YAML output."""

        return asdict(self)


@dataclass(slots=True)
class VLLMSession(BaseInferenceSession):
    """Own one live vLLM runtime plus Evalution-specific request preparation logic."""

    config: VLLM
    model_config: Model
    llm: Any
    tokenizer: Any
    prepare_tokenizer: Any | None
    sampling_params_cls: Any
    _generation_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _tokenizer_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _continuous_request_counter: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_config(cls, config: VLLM, model_config: Model) -> VLLMSession:
        """Load vLLM, construct the runtime, and attach tokenizers for request preparation."""

        vllm_module = _import_vllm(config.vllm_path)
        trust_remote_code = (
            config.trust_remote_code
            if config.trust_remote_code is not None
            else model_config.trust_remote_code
        )
        tokenizer_name = _resolve_vllm_tokenizer_name(model_config)
        # We keep a separate "prepare tokenizer" alongside vLLM's runtime tokenizer.
        # Evalution needs a stable tokenizer for prompt rendering and request
        # pre-tokenization even when the runtime backend owns the actual model.
        prepare_tokenizer = _load_tokenizer_from_model(
            _resolve_tokenizer_source(model_config),
            revision=model_config.revision,
            trust_remote_code=trust_remote_code,
            **model_config.tokenizer_kwargs,
        )
        if hasattr(prepare_tokenizer, "padding_side"):
            prepare_tokenizer.padding_side = config.padding_side

        llm_kwargs = dict(config.llm_kwargs)
        llm_kwargs.update(model_config.model_kwargs)
        llm_kwargs.setdefault("model", model_config.path)
        llm_kwargs.setdefault("tokenizer", tokenizer_name)
        llm_kwargs.setdefault("tokenizer_mode", config.tokenizer_mode)
        llm_kwargs.setdefault("tensor_parallel_size", config.tensor_parallel_size)
        llm_kwargs.setdefault("gpu_memory_utilization", config.gpu_memory_utilization)
        llm_kwargs.setdefault("enforce_eager", config.enforce_eager)
        llm_kwargs.setdefault("trust_remote_code", trust_remote_code)
        llm_kwargs.setdefault("revision", model_config.revision)
        llm_kwargs.setdefault("tokenizer_revision", config.tokenizer_revision or model_config.revision)
        if config.dtype is not None:
            llm_kwargs.setdefault("dtype", config.dtype)
        if config.quantization is not None:
            llm_kwargs.setdefault("quantization", config.quantization)
        if config.max_model_len is not None:
            llm_kwargs.setdefault("max_model_len", config.max_model_len)
        if config.seed is not None:
            llm_kwargs.setdefault("seed", config.seed)

        llm = vllm_module.LLM(**llm_kwargs)
        tokenizer = llm.get_tokenizer()
        return cls(
            config=config,
            model_config=model_config,
            llm=llm,
            tokenizer=tokenizer,
            prepare_tokenizer=prepare_tokenizer,
            sampling_params_cls=vllm_module.SamplingParams,
        )

    @property
    def batch_size(self) -> int | str:
        """Expose the configured batch size policy for compatibility with other engines."""

        return self.config.batch_size

    def describe_execution(self) -> dict[str, Any]:
        """Report the runtime settings that influence vLLM execution behavior."""

        return {
            "generation_backend": "vllm_generate",
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "quantization": self.config.quantization,
            "max_model_len": self._max_scoring_input_length(),
        }

    def prepare_requests(self, requests: list[GenerationRequest]) -> list[GenerationRequest]:
        """Normalize prompts and input ids before generation reaches the runtime."""

        tokenizer = self.prepare_tokenizer or self.tokenizer
        with self._tokenizer_lock:
            prepared: list[GenerationRequest] = []
            for request in requests:
                # Generation requests are normalized once up front so downstream
                # paths can assume both rendered_prompt and input_ids exist.
                # This matches the newer engine contract where request shaping is
                # separate from the runtime scheduling logic.
                rendered_prompt = (
                    request.rendered_prompt
                    if request.rendered_prompt is not None
                    else self._render_request_with_tokenizer(tokenizer, request)
                )
                input_ids = (
                    list(request.input_ids)
                    if request.input_ids is not None
                    else self._tokenize_text(tokenizer, rendered_prompt, add_special_tokens=False)
                )
                prepared.append(
                    replace(
                        request,
                        rendered_prompt=rendered_prompt,
                        input_ids=input_ids,
                    )
                )
            return prepared

    def resolve_batch_size(self, requests: list[GenerationRequest]) -> int:
        """Resolve an explicit or heuristic batch size for the given request set."""

        configured_batch_size = _normalize_batch_size(self.config.batch_size)
        if configured_batch_size != _AUTO_BATCH_SIZE:
            return int(configured_batch_size)

        max_num_seqs = self.config.llm_kwargs.get("max_num_seqs")
        if isinstance(max_num_seqs, int) and max_num_seqs > 0:
            return min(len(requests) or 1, max_num_seqs)
        return min(len(requests) or 1, _friendly_batch_size(len(requests) or 1))

    def generate(
            self,
            requests: list[GenerationRequest],
            *,
            batch_size: int | None = None,
    ) -> list[GenerationOutput]:
        """Generate completions for a finite request list."""

        if not requests:
            return []

        prepared_requests = self.prepare_requests(requests)
        effective_batch_size = batch_size or self.resolve_batch_size(prepared_requests)
        with self._generation_lock:
            # New-style engines must support continuous batching semantics. When
            # the installed vLLM exposes request-level scheduling primitives, we
            # drive the engine directly and reconcile outputs by request id.
            # Older or limited runtimes still get a compat fallback below.
            if not self._supports_request_level_continuous_batching():
                return self._generate_blocking(prepared_requests, batch_size=effective_batch_size)

            outputs_by_position: list[GenerationOutput | None] = [None] * len(prepared_requests)
            for position, output in self._generate_engine_continuous(
                enumerate(prepared_requests),
                batch_size=effective_batch_size,
            ):
                outputs_by_position[int(position)] = output
            if any(output is None for output in outputs_by_position):
                raise RuntimeError("vLLM continuous generation returned incomplete results")
            return [output for output in outputs_by_position if output is not None]

    def generate_continuous(
            self,
            requests: Iterable[tuple[Any, GenerationRequest]],
            *,
            batch_size: int | None = None,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        """Yield completions as soon as the runtime finishes each streamed request."""

        def iterator() -> Iterator[tuple[Any, GenerationOutput]]:
            """Wrap the streaming path so locks are acquired only during iteration."""

            request_iter = iter(requests)
            preview_items: list[tuple[Any, GenerationRequest]] = []
            for _ in range(64):
                try:
                    preview_items.append(next(request_iter))
                except StopIteration:
                    break
            if not preview_items:
                return

            # We only peek enough requests to resolve an "auto" batch size.
            # The rest of the stream stays lazy so continuous batching can refill
            # slots from the original iterator as soon as requests complete.
            effective_batch_size = batch_size or self.resolve_batch_size(
                [request for _, request in preview_items]
            )
            items = chain(preview_items, request_iter)

            with self._generation_lock:
                if self._supports_request_level_continuous_batching():
                    yield from self._generate_engine_continuous(items, batch_size=effective_batch_size)
                    return
                yield from self._generate_blocking_continuous(items, batch_size=effective_batch_size)

        return iterator()

    def _generate_blocking(
            self,
            prepared_requests: list[GenerationRequest],
            *,
            batch_size: int,
    ) -> list[GenerationOutput]:
        """Run generation in fixed batches against runtimes without request-level scheduling."""

        outputs: list[GenerationOutput] = []
        for start in range(0, len(prepared_requests), batch_size):
            batch = prepared_requests[start: start + batch_size]
            prompts = [self._prompt_from_generation_request(request) for request in batch]
            params = [self._sampling_params_for_generation(request) for request in batch]
            batch_outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)
            for request, result in zip(batch, batch_outputs, strict=True):
                outputs.append(self._generation_output_from_result(request, result))
        return outputs

    def _generate_blocking_continuous(
            self,
            requests: Iterable[tuple[Any, GenerationRequest]],
            *,
            batch_size: int,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        """Expose blocking generation through the continuous iterator contract."""

        # Compat path for runtimes that cannot accept requests incrementally.
        # It still exposes the same iterator interface to Evalution, but it
        # advances in fixed microbatches instead of true slot-by-slot refill.
        batch: list[tuple[Any, GenerationRequest]] = []
        for item in requests:
            batch.append(item)
            if len(batch) == batch_size:
                prepared_batch = [
                    prepared_request
                    for prepared_request in self.prepare_requests([request for _, request in batch])
                ]
                outputs = self._generate_blocking(prepared_batch, batch_size=len(prepared_batch))
                for (request_key, _request), output in zip(batch, outputs, strict=True):
                    yield request_key, output
                batch = []
        if batch:
            prepared_batch = [
                prepared_request
                for prepared_request in self.prepare_requests([request for _, request in batch])
            ]
            outputs = self._generate_blocking(prepared_batch, batch_size=len(prepared_batch))
            for (request_key, _request), output in zip(batch, outputs, strict=True):
                yield request_key, output

    def _generate_engine_continuous(
            self,
            requests: Iterable[tuple[Any, GenerationRequest]],
            *,
            batch_size: int,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        """Drive vLLM's request-level scheduler and refill slots as requests complete."""

        request_iter = iter(requests)
        # request_id is the stable join key between Evalution and vLLM. Result
        # order is not guaranteed to match submission order once the runtime can
        # complete and refill slots independently.
        inflight_requests: dict[str, tuple[Any, GenerationRequest]] = {}
        llm_engine = self.llm.llm_engine
        source_exhausted = False

        def submit_one() -> bool:
            """Prepare and submit one more request when capacity is available."""

            nonlocal source_exhausted
            if source_exhausted:
                return False
            try:
                request_key, request = next(request_iter)
            except StopIteration:
                source_exhausted = True
                return False

            prepared_request = self.prepare_requests([request])[0]
            request_id = self._next_continuous_request_id()
            # add_request reserves one inflight slot. The outer loop maintains
            # the invariant that inflight count never exceeds batch_size.
            llm_engine.add_request(
                request_id,
                self._prompt_from_generation_request(prepared_request),
                self._sampling_params_for_generation(prepared_request),
                prompt_text=prepared_request.rendered_prompt,
            )
            inflight_requests[request_id] = (request_key, prepared_request)
            return True

        try:
            while len(inflight_requests) < batch_size and submit_one():
                continue

            while inflight_requests:
                step_outputs = llm_engine.step()
                if not step_outputs:
                    if not llm_engine.has_unfinished_requests():
                        raise RuntimeError("vLLM engine stopped before all continuous requests completed")
                    continue

                for result in step_outputs:
                    request_id = getattr(result, "request_id", None)
                    if request_id is None:
                        raise RuntimeError("vLLM continuous batching returned an output without request_id")
                    request_state = inflight_requests.get(request_id)
                    if request_state is None:
                        raise RuntimeError(
                            f"vLLM continuous batching returned unknown request_id={request_id!r}"
                        )
                    if not getattr(result, "finished", False):
                        continue

                    request_key, request = inflight_requests.pop(request_id)
                    yield request_key, self._generation_output_from_result(request, result)

                    # Continuous batching refill: every time a request finishes,
                    # immediately backfill the newly opened slot before waiting
                    # for the next engine step.
                    while len(inflight_requests) < batch_size and submit_one():
                        continue
        finally:
            abort_request = getattr(llm_engine, "abort_request", None)
            if callable(abort_request) and inflight_requests:
                with suppress(Exception):
                    abort_request(list(inflight_requests.keys()), internal=True)

    def _generation_output_from_result(
            self,
            request: GenerationRequest,
            result: Any,
    ) -> GenerationOutput:
        """Convert one vLLM result object into Evalution's generation output shape."""

        completion = result.outputs[0] if getattr(result, "outputs", None) else None
        text = "" if completion is None else _truncate_at_stop(completion.text, request.stop).strip()
        metadata = dict(request.metadata)
        if completion is not None:
            metadata.update(
                {
                    "finish_reason": getattr(completion, "finish_reason", None),
                    "stop_reason": getattr(completion, "stop_reason", None),
                    "completion_token_count": len(getattr(completion, "token_ids", []) or []),
                }
            )
        metadata["prompt_token_count"] = len(getattr(result, "prompt_token_ids", []) or [])
        return GenerationOutput(
            prompt=request.rendered_prompt or "",
            text=text,
            metadata=metadata,
        )

    def _supports_request_level_continuous_batching(self) -> bool:
        """Detect whether the runtime exposes the primitives needed for slot refill."""

        llm_engine = getattr(self.llm, "llm_engine", None)
        return (
            llm_engine is not None
            and callable(getattr(llm_engine, "add_request", None))
            and callable(getattr(llm_engine, "step", None))
            and callable(getattr(llm_engine, "has_unfinished_requests", None))
        )

    def _next_continuous_request_id(self) -> str:
        """Allocate a stable request id used to reconcile out-of-order vLLM results."""

        request_id = f"req_{self._continuous_request_counter}"
        self._continuous_request_counter += 1
        return request_id

    def loglikelihood(
            self,
            requests: list[LoglikelihoodRequest],
            *,
            batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]:
        """Score explicit continuation tokens using vLLM prompt log probabilities."""

        if not requests:
            return []

        prepared_requests = [self._prepare_loglikelihood_request(request) for request in requests]
        effective_batch_size = batch_size or self.resolve_batch_size(
            [
                GenerationRequest(rendered_prompt="", input_ids=prefix_ids + target_ids, max_new_tokens=0)
                for prefix_ids, target_ids, _metadata in prepared_requests
            ]
        )

        outputs: list[LoglikelihoodOutput] = []
        with self._generation_lock:
            for start in range(0, len(prepared_requests), effective_batch_size):
                batch = prepared_requests[start: start + effective_batch_size]
                prompts = []
                params = []
                score_starts: list[int] = []
                for prefix_ids, target_ids, _metadata in batch:
                    if prefix_ids:
                        prompt_token_ids = [*prefix_ids, *target_ids]
                        score_start = len(prefix_ids)
                    else:
                        # For empty-context scoring we synthesize a one-token
                        # prefix so the first continuation token still has a
                        # valid predecessor, matching the semantics used by the
                        # other scoring engines.
                        prompt_token_ids = [self._prefix_token_id(), *target_ids]
                        score_start = 1
                    prompts.append({"prompt_token_ids": prompt_token_ids})
                    params.append(
                        # Newer vLLM rejects max_tokens=0, so we request the
                        # smallest legal decode and ignore generated tokens.
                        # The score comes from prompt_logprobs only.
                        self.sampling_params_cls(
                            max_tokens=1,
                            temperature=0.0,
                            prompt_logprobs=1,
                            detokenize=False,
                            skip_special_tokens=False,
                        )
                    )
                    score_starts.append(score_start)

                batch_outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)
                for (prefix_ids, target_ids, metadata), result, score_start in zip(
                        batch,
                        batch_outputs,
                        score_starts,
                        strict=True,
                ):
                    prompt_logprobs = getattr(result, "prompt_logprobs", None)
                    if prompt_logprobs is None:
                        raise RuntimeError("vLLM did not return prompt_logprobs for loglikelihood scoring")

                    total_logprob = 0.0
                    is_greedy = True
                    for offset, token_id in enumerate(target_ids):
                        position = score_start + offset
                        token_logprob = prompt_logprobs[position]
                        if token_logprob is None or token_id not in token_logprob:
                            raise RuntimeError(
                                "vLLM prompt_logprobs omitted the scored continuation token"
                            )
                        logprob_info = token_logprob[token_id]
                        total_logprob += float(logprob_info.logprob)
                        rank = getattr(logprob_info, "rank", None)
                        if rank is None:
                            rank = self._best_rank_from_prompt_logprob(token_logprob, token_id)
                        is_greedy = is_greedy and rank == 1

                    outputs.append(
                        LoglikelihoodOutput(
                            logprob=total_logprob,
                            is_greedy=is_greedy,
                            token_count=len(target_ids),
                            metadata=metadata,
                        )
                    )
        return outputs

    def loglikelihood_rolling(
            self,
            requests: list[RollingLoglikelihoodRequest],
            *,
            batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]:
        """Score long texts by reducing rolling perplexity to ordinary loglikelihood windows."""

        if not requests:
            return []

        # Rolling perplexity is reduced to ordinary loglikelihood over disjoint
        # windows so we can reuse the exact same prompt_logprobs scoring path.
        scoring_requests: list[LoglikelihoodRequest] = []
        request_window_counts: list[int] = []
        for request in requests:
            token_list = (
                list(request.input_ids)
                if request.input_ids is not None
                else self._tokenize_loglikelihood_context(request.text)
            )
            windows = list(self._rolling_loglikelihood_windows(token_list))
            request_window_counts.append(len(windows))
            for context_ids, continuation_ids in windows:
                scoring_requests.append(
                    LoglikelihoodRequest(
                        context_input_ids=context_ids,
                        continuation_input_ids=continuation_ids,
                        metadata=dict(request.metadata),
                    )
                )

        scored = self.loglikelihood(scoring_requests, batch_size=batch_size)
        outputs: list[RollingLoglikelihoodOutput] = []
        cursor = 0
        for request, window_count in zip(requests, request_window_counts, strict=True):
            if window_count == 0:
                outputs.append(
                    RollingLoglikelihoodOutput(
                        logprob=0.0,
                        token_count=0,
                        metadata=dict(request.metadata),
                    )
                )
                continue

            request_outputs = scored[cursor: cursor + window_count]
            cursor += window_count
            outputs.append(
                RollingLoglikelihoodOutput(
                    logprob=sum(output.logprob for output in request_outputs),
                    token_count=sum(output.token_count for output in request_outputs),
                    metadata=dict(request.metadata),
                )
            )
        return outputs

    def gc(self) -> None:
        """Release reusable vLLM prefix-cache state when the runtime supports it."""

        with self._generation_lock:
            reset_prefix_cache = getattr(self.llm, "reset_prefix_cache", None)
            if callable(reset_prefix_cache):
                with suppress(Exception):
                    reset_prefix_cache()

    def close(self) -> None:
        """Shut down vLLM resources and drop tokenizer/runtime references."""

        with self._generation_lock:
            llm_engine = getattr(self.llm, "llm_engine", None)
            shutdown = getattr(llm_engine, "shutdown", None)
            if callable(shutdown):
                with suppress(Exception):
                    shutdown()
            engine_core = getattr(llm_engine, "engine_core", None)
            shutdown_engine_core = getattr(engine_core, "shutdown", None)
            if callable(shutdown_engine_core):
                with suppress(Exception):
                    shutdown_engine_core()
            with suppress(Exception):
                del self.llm
            with suppress(Exception):
                del self.tokenizer
            with suppress(Exception):
                del self.prepare_tokenizer

    def _render_request_with_tokenizer(self, tokenizer: Any, request: GenerationRequest) -> str:
        """Render either a plain prompt or a chat-formatted prompt string."""

        if request.messages is not None:
            apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
            if not callable(apply_chat_template):
                raise ValueError("generation requests with messages require tokenizer.apply_chat_template")
            return apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
            )
        if request.prompt is None:
            raise ValueError("generation requests must define either `prompt` or `messages`")
        return request.prompt

    def _prompt_from_generation_request(self, request: GenerationRequest) -> dict[str, Any]:
        """Translate a prepared request into the prompt payload expected by vLLM."""

        if request.num_beams != 1:
            raise ValueError("VLLM currently requires num_beams=1")
        if request.input_ids is None:
            raise ValueError("prepared generation requests must include input_ids")
        prompt: dict[str, Any] = {"prompt_token_ids": list(request.input_ids)}
        if request.rendered_prompt is not None:
            prompt["prompt"] = request.rendered_prompt
        return prompt

    def _sampling_params_for_generation(self, request: GenerationRequest) -> Any:
        """Build the vLLM sampling-parameter object for one generation request."""

        if request.num_beams != 1:
            raise ValueError("VLLM currently requires num_beams=1")
        return self.sampling_params_cls(
            max_tokens=request.max_new_tokens if request.max_new_tokens is not None else self.config.max_new_tokens,
            stop=list(request.stop) if request.stop else None,
            temperature=request.temperature if request.do_sample else 0.0,
            detokenize=True,
            skip_special_tokens=False,
        )

    def _prepare_loglikelihood_request(
            self,
            request: LoglikelihoodRequest,
    ) -> tuple[list[int], list[int], dict[str, Any]]:
        """Convert one loglikelihood request into token ids plus copied metadata."""

        tokenizer = self.prepare_tokenizer or self.tokenizer
        if request.context_input_ids is not None:
            prefix_ids = list(request.context_input_ids)
        elif request.context:
            prefix_ids = self._tokenize_loglikelihood_context(request.context)
        else:
            prefix_ids = []

        if request.continuation_input_ids is not None:
            target_ids = list(request.continuation_input_ids)
        elif request.continuation:
            target_ids = self._tokenize_text(tokenizer, request.continuation, add_special_tokens=False)
        else:
            target_ids = []

        if not target_ids:
            raise ValueError("loglikelihood requests must provide a non-empty continuation")
        return prefix_ids, target_ids, dict(request.metadata)

    def _tokenize_loglikelihood_context(self, text: str) -> list[int]:
        """Tokenize scoring context while preserving synthetic-prefix behavior when needed."""

        tokenizer = self.prepare_tokenizer or self.tokenizer
        tokenizer_kwargs: dict[str, Any] = {}
        prefix_text = self._decoded_prefix_token_text()
        if prefix_text and text.startswith(prefix_text):
            tokenizer_kwargs["add_special_tokens"] = False
        return self._tokenize_text(tokenizer, text, **tokenizer_kwargs)

    def _rolling_loglikelihood_windows(
            self,
            token_list: list[int],
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Split tokens into the overlapping windows needed for rolling scoring."""

        if not token_list:
            return

        max_seq_len = self._max_scoring_input_length()
        prefix_token = self._prefix_token_id()
        pred_len = max_seq_len
        predicted = 0

        # The first window includes a synthetic prefix token for consistency
        # with empty-context token scoring. Later windows overlap by one token
        # and are made disjoint in _make_disjoint_window().
        first_seq_len = min(max_seq_len, len(token_list))
        first_window = (
            [prefix_token] + token_list[: first_seq_len - 1],
            token_list[:first_seq_len],
        )
        yield self._make_disjoint_window(first_window)
        predicted += first_seq_len

        while predicted < len(token_list):
            window_pred_len = min(len(token_list) - predicted, pred_len)
            window_end = predicted + window_pred_len
            window = (
                token_list[window_end - max_seq_len - 1: window_end - 1],
                token_list[window_end - window_pred_len: window_end],
            )
            yield self._make_disjoint_window(window)
            predicted += window_pred_len

    def _make_disjoint_window(
            self,
            pair: tuple[list[int], list[int]],
    ) -> tuple[list[int], list[int]]:
        """Trim an overlapping rolling window into a disjoint scoring pair."""

        context_ids, continuation_ids = pair
        return context_ids[: len(context_ids) - (len(continuation_ids) - 1)], continuation_ids

    def _prefix_token_id(self) -> int:
        """Choose a tokenizer token id that can act as a synthetic scoring prefix."""

        tokenizer = self.prepare_tokenizer or self.tokenizer
        for token_id in (
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "eos_token_id", None),
                getattr(tokenizer, "pad_token_id", None),
        ):
            if token_id is not None:
                return int(token_id)
        raise ValueError(
            "token-level scoring requires tokenizer.bos_token_id, eos_token_id, or pad_token_id"
        )

    def _decoded_prefix_token_text(self) -> str | None:
        """Decode the synthetic prefix token when the tokenizer exposes a decode API."""

        tokenizer = self.prepare_tokenizer or self.tokenizer
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            return None
        with suppress(Exception):
            decoded = decode([self._prefix_token_id()], skip_special_tokens=False)
            if decoded:
                return str(decoded)
        with suppress(Exception):
            decoded = decode(self._prefix_token_id(), skip_special_tokens=False)
            if decoded:
                return str(decoded)
        return None

    def _max_scoring_input_length(self) -> int:
        """Resolve the safest maximum sequence length to use for scoring windows."""

        candidate_lengths = [
            self.config.max_model_len,
            getattr(self.prepare_tokenizer or self.tokenizer, "model_max_length", None),
            getattr(getattr(self.llm, "model_config", None), "max_model_len", None),
            getattr(
                getattr(
                    getattr(getattr(self.llm, "llm_engine", None), "vllm_config", None),
                    "model_config",
                    None,
                ),
                "max_model_len",
                None,
            ),
        ]
        resolved = [
            int(length)
            for length in candidate_lengths
            if isinstance(length, int) and 1 < length < 1_000_000
        ]
        if resolved:
            return min(resolved)
        return 2048

    def _best_rank_from_prompt_logprob(self, token_logprob: dict[int, Any], token_id: int) -> int | None:
        """Infer a token rank from raw prompt-logprob scores when vLLM omits rank metadata."""

        token_info = token_logprob.get(token_id)
        if token_info is None:
            return None
        ranked = sorted(
            ((candidate_id, float(candidate.logprob)) for candidate_id, candidate in token_logprob.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        for index, (candidate_id, _score) in enumerate(ranked, start=1):
            if candidate_id == token_id:
                return index
        return None

    def _tokenize_text(self, tokenizer: Any, text: str, **kwargs: Any) -> list[int]:
        """Tokenize text through either encode() or the tokenizer call interface."""

        with self._tokenizer_lock:
            encode = getattr(tokenizer, "encode", None)
            if callable(encode):
                encoded = encode(text, **kwargs)
                if isinstance(encoded, list):
                    return [int(token_id) for token_id in encoded]

            tokenized = tokenizer(text, **kwargs)
            input_ids = tokenized["input_ids"]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return [int(token_id) for token_id in input_ids]


def _resolve_vllm_tokenizer_name(model_config: Model) -> str:
    """Resolve the tokenizer identifier that should be passed into vLLM."""

    tokenizer = model_config.tokenizer
    if isinstance(tokenizer, os.PathLike):
        return os.fspath(tokenizer)
    if isinstance(tokenizer, str):
        return tokenizer
    if model_config.tokenizer_path is not None:
        return model_config.tokenizer_path
    return model_config.path


def _import_vllm(vllm_path: str | None) -> Any:
    """Import vLLM, optionally falling back to a nearby local checkout."""

    try:
        return importlib.import_module("vllm")
    except ModuleNotFoundError as exc:
        # Evalution is often developed alongside a local vLLM checkout. We try
        # the configured path first, then a few common sibling-repo locations,
        # before failing with a targeted import error.
        search_paths = []
        if vllm_path:
            search_paths.append(Path(vllm_path))
        search_paths.extend(_DEFAULT_VLLM_CHECKOUT_CANDIDATES)

        for candidate in search_paths:
            if not candidate.exists():
                continue
            root = str(candidate)
            if root not in sys.path:
                sys.path.insert(0, root)
            try:
                return importlib.import_module("vllm")
            except ModuleNotFoundError:
                continue

        raise ModuleNotFoundError(
            "vllm is not importable; install it or configure `vllm_path` to a local checkout"
        ) from exc
