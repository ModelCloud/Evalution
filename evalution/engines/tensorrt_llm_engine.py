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

# Keep the engine's auto-batching sentinel aligned with the shared runtime helpers.
_AUTO_BATCH_SIZE = "auto"
# Search common sibling-checkout locations so local TensorRT-LLM development works without pip install.
_DEFAULT_TENSORRT_LLM_CHECKOUT_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "TensorRT-LLM",
    Path.cwd() / "TensorRT-LLM",
    Path.cwd().parent / "TensorRT-LLM",
    Path(__file__).resolve().parents[3] / "tensorrt_llm",
    Path.cwd() / "tensorrt_llm",
    Path.cwd().parent / "tensorrt_llm",
)


@dataclass(slots=True)
class TensorRTLLM(SharedEngineConfig):
    """Configure Evalution to run generation and scoring through TensorRT-LLM."""

    # Mirror the major TensorRT-LLM constructor knobs we want to surface through Evalution and YAML.
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    tokenizer_revision: str | None = None
    runtime_backend: str | None = None
    tensorrt_llm_path: str | None = None
    llm_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, model: Model) -> BaseInferenceSession:
        """Build build."""
        self.resolved_engine = "TensorRTLLM"
        return TensorRTLLMSession.from_config(self, model)

    def to_dict(self) -> dict[str, Any]:
        """Implement to dict for tensor rtllm."""
        return asdict(self)


@dataclass(slots=True)
class TensorRTLLMSession(BaseInferenceSession):
    """Own one live TensorRT-LLM runtime plus Evalution request translation logic."""

    # Keep the live runtime and tokenizer state on the session so one built engine can be reused
    # across multiple suites without rebuilding the backend for every test case.
    config: TensorRTLLM
    model_config: Model
    llm: Any
    tokenizer: Any
    prepare_tokenizer: Any | None
    sampling_params_cls: Any
    # Serialize backend calls because TensorRT-LLM runtime objects and tokenizers are not assumed
    # to be re-entrant across concurrent Evalution requests.
    _generation_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _tokenizer_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    # Stable synthetic ids let Evalution reconcile out-of-order continuous results back to callers.
    _continuous_request_counter: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_config(cls, config: TensorRTLLM, model_config: Model) -> TensorRTLLMSession:
        """Load TensorRT-LLM plus tokenizers and normalize Evalution config into runtime kwargs."""

        tensorrt_llm_module = _import_tensorrt_llm(config.tensorrt_llm_path)
        trust_remote_code = (
            config.trust_remote_code
            if config.trust_remote_code is not None
            else model_config.trust_remote_code
        )
        tokenizer_name = _resolve_runtime_tokenizer_name(model_config)
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
        llm_kwargs.setdefault("tensor_parallel_size", config.tensor_parallel_size)
        llm_kwargs.setdefault("trust_remote_code", trust_remote_code)
        llm_kwargs.setdefault("revision", model_config.revision)
        llm_kwargs.setdefault("tokenizer_revision", config.tokenizer_revision or model_config.revision)
        if config.runtime_backend is not None:
            llm_kwargs.setdefault("backend", config.runtime_backend)
        if config.dtype is not None:
            llm_kwargs.setdefault("dtype", config.dtype)
        if config.max_model_len is not None:
            llm_kwargs.setdefault("max_model_len", config.max_model_len)
        if config.seed is not None:
            llm_kwargs.setdefault("seed", config.seed)

        llm = tensorrt_llm_module.LLM(**llm_kwargs)
        get_tokenizer = getattr(llm, "get_tokenizer", None)
        tokenizer = get_tokenizer() if callable(get_tokenizer) else getattr(llm, "tokenizer", None)
        if tokenizer is None:
            tokenizer = prepare_tokenizer
        return cls(
            config=config,
            model_config=model_config,
            llm=llm,
            tokenizer=tokenizer,
            prepare_tokenizer=prepare_tokenizer,
            sampling_params_cls=tensorrt_llm_module.SamplingParams,
        )

    @property
    def batch_size(self) -> int | str:
        """Implement batch size for tensor rtllmsession."""
        return self.config.batch_size

    def describe_execution(self) -> dict[str, Any]:
        """Expose stable runtime metadata that is useful in result payloads and snapshots."""

        return {
            "generation_backend": (
                "tensorrt_llm_continuous"
                if self._supports_request_level_continuous_batching()
                else "tensorrt_llm_generate"
            ),
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "runtime_backend": self._runtime_backend(),
            "max_model_len": self._max_scoring_input_length(),
        }

    def prepare_requests(self, requests: list[GenerationRequest]) -> list[GenerationRequest]:
        """Render prompts and token ids once so downstream generation paths can stay simple."""

        tokenizer = self.prepare_tokenizer or self.tokenizer
        with self._tokenizer_lock:
            prepared: list[GenerationRequest] = []
            for request in requests:
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
        """Resolve either an explicit batch size or a conservative auto heuristic."""

        configured_batch_size = _normalize_batch_size(self.config.batch_size)
        if configured_batch_size != _AUTO_BATCH_SIZE:
            return int(configured_batch_size)

        max_batch_size = self.config.llm_kwargs.get("max_batch_size")
        if isinstance(max_batch_size, int) and max_batch_size > 0:
            return min(len(requests) or 1, max_batch_size)
        return min(len(requests) or 1, _friendly_batch_size(len(requests) or 1))

    def generate(
            self,
            requests: list[GenerationRequest],
            *,
            batch_size: int | None = None,
    ) -> list[GenerationOutput]:
        """Generate one completion per request while preserving caller order."""

        if not requests:
            return []

        prepared_requests = self.prepare_requests(requests)
        effective_batch_size = batch_size or self.resolve_batch_size(prepared_requests)
        with self._generation_lock:
            if not self._supports_request_level_continuous_batching():
                return self._generate_blocking(prepared_requests, batch_size=effective_batch_size)

            outputs_by_position: list[GenerationOutput | None] = [None] * len(prepared_requests)
            for position, output in self._generate_engine_continuous(
                    enumerate(prepared_requests),
                    batch_size=effective_batch_size,
            ):
                outputs_by_position[int(position)] = output
            if any(output is None for output in outputs_by_position):
                raise RuntimeError("TensorRT-LLM continuous generation returned incomplete results")
            return [output for output in outputs_by_position if output is not None]

    def generate_continuous(
            self,
            requests: Iterable[tuple[Any, GenerationRequest]],
            *,
            batch_size: int | None = None,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        """Yield completions in runtime finish order while preserving caller request ids."""

        def iterator() -> Iterator[tuple[Any, GenerationOutput]]:
            """Implement iterator for tensor rtllmsession."""
            request_iter = iter(requests)
            preview_items: list[tuple[Any, GenerationRequest]] = []
            for _ in range(64):
                try:
                    preview_items.append(next(request_iter))
                except StopIteration:
                    break
            if not preview_items:
                return

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
        """Run fixed batches against runtimes that only expose blocking generation APIs."""

        outputs: list[GenerationOutput] = []
        for start in range(0, len(prepared_requests), batch_size):
            batch = prepared_requests[start: start + batch_size]
            prompts = [self._prompt_from_generation_request(request) for request in batch]
            params = [self._sampling_params_for_generation(request) for request in batch]
            batch_outputs = self._generate_with_runtime(prompts, params)
            for request, result in zip(batch, batch_outputs, strict=True):
                outputs.append(self._generation_output_from_result(request, result))
        return outputs

    def _generate_blocking_continuous(
            self,
            requests: Iterable[tuple[Any, GenerationRequest]],
            *,
            batch_size: int,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        """Adapt blocking generation into Evalution's continuous iterator contract."""

        batch: list[tuple[Any, GenerationRequest]] = []
        for item in requests:
            batch.append(item)
            if len(batch) == batch_size:
                prepared_batch = self.prepare_requests([request for _, request in batch])
                outputs = self._generate_blocking(prepared_batch, batch_size=len(prepared_batch))
                for (request_key, _request), output in zip(batch, outputs, strict=True):
                    yield request_key, output
                batch = []
        if batch:
            prepared_batch = self.prepare_requests([request for _, request in batch])
            outputs = self._generate_blocking(prepared_batch, batch_size=len(prepared_batch))
            for (request_key, _request), output in zip(batch, outputs, strict=True):
                yield request_key, output

    def _generate_engine_continuous(
            self,
            requests: Iterable[tuple[Any, GenerationRequest]],
            *,
            batch_size: int,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        """Drive request-level runtime scheduling and refill freed slots eagerly."""

        request_iter = iter(requests)
        inflight_requests: dict[str, tuple[Any, GenerationRequest]] = {}
        llm_engine = self.llm.llm_engine
        source_exhausted = False

        def submit_one() -> bool:
            """Implement submit one for tensor rtllmsession."""
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
                        raise RuntimeError("TensorRT-LLM engine stopped before all continuous requests completed")
                    continue

                for result in step_outputs:
                    request_id = getattr(result, "request_id", None)
                    if request_id is None:
                        raise RuntimeError(
                            "TensorRT-LLM continuous batching returned an output without request_id"
                        )
                    request_state = inflight_requests.get(request_id)
                    if request_state is None:
                        raise RuntimeError(
                            f"TensorRT-LLM continuous batching returned unknown request_id={request_id!r}"
                        )
                    if not getattr(result, "finished", False):
                        continue

                    request_key, request = inflight_requests.pop(request_id)
                    yield request_key, self._generation_output_from_result(request, result)

                    while len(inflight_requests) < batch_size and submit_one():
                        continue
        finally:
            abort_request = getattr(llm_engine, "abort_request", None)
            if callable(abort_request) and inflight_requests:
                with suppress(Exception):
                    abort_request(list(inflight_requests.keys()), internal=True)

    def _generate_with_runtime(self, prompts: list[Any], params: list[Any]) -> list[Any]:
        """Normalize minor generate signature differences across TensorRT-LLM versions."""

        try:
            outputs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)
        except TypeError:
            outputs = self.llm.generate(prompts, sampling_params=params)
        if isinstance(outputs, list):
            return list(outputs)
        return [outputs]

    def _generation_output_from_result(self, request: GenerationRequest, result: Any) -> GenerationOutput:
        """Convert one backend request result into Evalution's generation output shape."""

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
        """Detect whether the runtime exposes the primitives needed for true slot refill."""

        llm_engine = getattr(self.llm, "llm_engine", None)
        return (
                llm_engine is not None
                and callable(getattr(llm_engine, "add_request", None))
                and callable(getattr(llm_engine, "step", None))
                and callable(getattr(llm_engine, "has_unfinished_requests", None))
        )

    def _next_continuous_request_id(self) -> str:
        """Allocate a stable synthetic request id for runtime-side reconciliation."""

        request_id = f"req_{self._continuous_request_counter}"
        self._continuous_request_counter += 1
        return request_id

    def loglikelihood(
            self,
            requests: list[LoglikelihoodRequest],
            *,
            batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]:
        """Score explicit continuation tokens using backend prompt log-probabilities."""

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
                        prompt_token_ids = [self._prefix_token_id(), *target_ids]
                        score_start = 1
                    prompts.append({"prompt_token_ids": prompt_token_ids})
                    params.append(
                        self.sampling_params_cls(
                            max_tokens=1,
                            temperature=0.0,
                            prompt_logprobs=1,
                            detokenize=False,
                            skip_special_tokens=False,
                        )
                    )
                    score_starts.append(score_start)

                batch_outputs = self._generate_with_runtime(prompts, params)
                for (prefix_ids, target_ids, metadata), result, score_start in zip(
                        batch,
                        batch_outputs,
                        score_starts,
                        strict=True,
                ):
                    prompt_logprobs = getattr(result, "prompt_logprobs", None)
                    if prompt_logprobs is None:
                        raise RuntimeError(
                            "TensorRT-LLM did not return prompt_logprobs for loglikelihood scoring"
                        )

                    total_logprob = 0.0
                    is_greedy = True
                    for offset, token_id in enumerate(target_ids):
                        position = score_start + offset
                        token_logprob = prompt_logprobs[position]
                        if token_logprob is None or token_id not in token_logprob:
                            raise RuntimeError(
                                "TensorRT-LLM prompt_logprobs omitted the scored continuation token"
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
        """Reduce rolling perplexity-style scoring to ordinary loglikelihood windows."""

        if not requests:
            return []

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
        """Release reusable backend caches between suites without unloading the whole model."""

        with self._generation_lock:
            for attribute in ("reset_prefix_cache", "clear_kv_cache", "reset_kv_cache"):
                releaser = getattr(self.llm, attribute, None)
                if callable(releaser):
                    with suppress(Exception):
                        releaser()

    def close(self) -> None:
        """Shut down the live runtime and drop heavyweight tokenizer/runtime references."""

        with self._generation_lock:
            shutdown = getattr(self.llm, "shutdown", None)
            if callable(shutdown):
                with suppress(Exception):
                    shutdown()
            llm_engine = getattr(self.llm, "llm_engine", None)
            shutdown_engine = getattr(llm_engine, "shutdown", None)
            if callable(shutdown_engine):
                with suppress(Exception):
                    shutdown_engine()
            with suppress(Exception):
                del self.llm
            with suppress(Exception):
                del self.tokenizer
            with suppress(Exception):
                del self.prepare_tokenizer

    def _render_request_with_tokenizer(self, tokenizer: Any, request: GenerationRequest) -> str:
        """Render either a plain prompt or chat messages into one backend-ready string."""

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
        """Translate a prepared request into the prompt payload expected by TensorRT-LLM."""

        if request.num_beams != 1:
            raise ValueError("TensorRT-LLM currently requires num_beams=1")
        if request.input_ids is None:
            raise ValueError("prepared generation requests must include input_ids")
        prompt: dict[str, Any] = {"prompt_token_ids": list(request.input_ids)}
        if request.rendered_prompt is not None:
            prompt["prompt"] = request.rendered_prompt
        return prompt

    def _sampling_params_for_generation(self, request: GenerationRequest) -> Any:
        """Build one SamplingParams object from Evalution's request-level generation controls."""

        if request.num_beams != 1:
            raise ValueError("TensorRT-LLM currently requires num_beams=1")
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
        """Convert one scoring request into prefix ids, continuation ids, and copied metadata."""

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
        """Tokenize scoring context while preserving the synthetic-prefix scoring semantics."""

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
        """Split a token list into the disjoint scoring windows used for rolling perplexity."""

        if not token_list:
            return

        max_seq_len = self._max_scoring_input_length()
        prefix_token = self._prefix_token_id()
        pred_len = max_seq_len
        predicted = 0

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
        """Trim one overlapping rolling window down to the disjoint continuation that is scored."""

        context_ids, continuation_ids = pair
        return context_ids[: len(context_ids) - (len(continuation_ids) - 1)], continuation_ids

    def _prefix_token_id(self) -> int:
        """Pick a tokenizer token that can safely serve as the synthetic empty-context prefix."""

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
        """Resolve the most conservative available max sequence length for scoring windows."""

        candidate_lengths = [
            self.config.max_model_len,
            getattr(self.prepare_tokenizer or self.tokenizer, "model_max_length", None),
            getattr(getattr(self.llm, "model_config", None), "max_model_len", None),
            getattr(getattr(self.llm, "args", None), "max_model_len", None),
            getattr(getattr(self.llm, "args", None), "max_seq_len", None),
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
        """Infer rank metadata when the backend returns logprobs without an explicit rank field."""

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

    def _runtime_backend(self) -> str | None:
        """Expose the runtime backend name in a stable string form for execution metadata."""

        args = getattr(self.llm, "args", None)
        backend = getattr(args, "backend", None)
        if backend is None:
            backend = self.config.runtime_backend
        return None if backend is None else str(backend)

    def _tokenize_text(self, tokenizer: Any, text: str, **kwargs: Any) -> list[int]:
        """Tokenize through either encode() or the tokenizer call interface."""

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


def _resolve_runtime_tokenizer_name(model_config: Model) -> str:
    """Resolve the tokenizer identifier or path that should be passed into TensorRT-LLM."""

    tokenizer = model_config.tokenizer
    if isinstance(tokenizer, os.PathLike):
        return os.fspath(tokenizer)
    if isinstance(tokenizer, str):
        return tokenizer
    if model_config.tokenizer_path is not None:
        return model_config.tokenizer_path
    return model_config.path


def _import_tensorrt_llm(tensorrt_llm_path: str | None) -> Any:
    """Import TensorRT-LLM, optionally retrying through common local checkout locations."""

    def _ensure_transformers_compat() -> None:
        """Patch known import-surface drift between TensorRT-LLM and newer transformers builds."""

        try:
            transformers_module = importlib.import_module("transformers")
        except Exception:
            return

        if hasattr(transformers_module, "AutoModelForVision2Seq"):
            return

        class _MissingAutoModelForVision2Seq:
            """Fallback placeholder for TensorRT-LLM imports on text-only workloads."""

            def __new__(cls, *args: Any, **kwargs: Any) -> Any:
                """Implement new for missing auto model for vision2 seq."""
                raise RuntimeError(
                    "transformers.AutoModelForVision2Seq is unavailable in this environment"
                )

        setattr(transformers_module, "AutoModelForVision2Seq", _MissingAutoModelForVision2Seq)

    _ensure_transformers_compat()
    try:
        return importlib.import_module("tensorrt_llm")
    except ModuleNotFoundError as exc:
        search_paths = []
        if tensorrt_llm_path:
            search_paths.append(Path(tensorrt_llm_path))
        search_paths.extend(_DEFAULT_TENSORRT_LLM_CHECKOUT_CANDIDATES)

        for candidate in search_paths:
            if not candidate.exists():
                continue
            root = str(candidate)
            if root not in sys.path:
                sys.path.insert(0, root)
            try:
                _ensure_transformers_compat()
                return importlib.import_module("tensorrt_llm")
            except ModuleNotFoundError:
                continue

        raise ModuleNotFoundError(
            "tensorrt_llm is not importable; install it or configure `tensorrt_llm_path` to a local checkout"
        ) from exc
