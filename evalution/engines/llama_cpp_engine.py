# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import gc
import importlib
import sys
import threading
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from itertools import chain, islice
from pathlib import Path
from typing import Any

from evalution.config import Model
from evalution.engines.base import (
    BaseEngineDeviceConfig,
    BaseInferenceSession,
    GenerationOutput,
    GenerationRequest,
    LoglikelihoodOutput,
    LoglikelihoodRequest,
    RollingLoglikelihoodOutput,
    RollingLoglikelihoodRequest,
    SharedEngineConfig,
)
from evalution.engines.continuous import stream_request_results
from evalution.engines.transformers_common import (
    _friendly_batch_size,
    _load_tokenizer_from_model,
    _normalize_batch_size,
    _resolve_tokenizer_source,
    _truncate_at_stop,
)

# Keep the engine's auto-batching sentinel aligned with the shared runtime helpers.
_AUTO_BATCH_SIZE = "auto"
# Search common sibling-checkout locations so local source builds work before pip installation.
_DEFAULT_LLAMA_CPP_CHECKOUT_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "llama-cpp-python",
    Path.cwd() / "llama-cpp-python",
    Path.cwd().parent / "llama-cpp-python",
    Path(__file__).resolve().parents[3] / "llama_cpp_python",
    Path.cwd() / "llama_cpp_python",
    Path.cwd().parent / "llama_cpp_python",
)


@dataclass(slots=True)
class LlamaCpp(BaseEngineDeviceConfig, SharedEngineConfig):
    """Configure Evalution to run generation and scoring through llama.cpp."""

    # Mirror the llama-cpp-python constructor knobs we want to expose directly through Evalution.
    n_gpu_layers: int | None = None
    main_gpu: int = 0
    split_mode: int = 1
    tensor_split: list[float] | None = None
    n_ctx: int = 4096
    n_batch: int = 512
    n_ubatch: int = 512
    n_threads: int | None = None
    n_threads_batch: int | None = None
    flash_attn: bool = False
    offload_kqv: bool = True
    use_mmap: bool = True
    use_mlock: bool = False
    chat_format: str | None = None
    verbose: bool = False
    logits_all: bool = True
    llama_cpp_path: str | None = None
    llama_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, model: Model) -> BaseInferenceSession:
        """Construct a llama.cpp-backed inference session for the requested model."""

        self.resolved_engine = "LlamaCpp"
        return LlamaCppSession.from_config(self, model)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the engine configuration for reporting and YAML output."""

        return asdict(self)


@dataclass(slots=True)
class LlamaCppSession(BaseInferenceSession):
    """Own one live llama.cpp runtime plus Evalution request translation logic."""

    # Keep the live runtime and tokenizer state on the session so one built engine can be reused
    # across multiple suites without reloading the underlying GGUF model every time.
    config: LlamaCpp
    model_config: Model
    llm: Any
    llama_module: Any
    prepare_tokenizer: Any | None
    effective_device: str
    gpu_offload_supported: bool
    effective_n_gpu_layers: int
    _generation_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _tokenizer_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    @classmethod
    def from_config(cls, config: LlamaCpp, model_config: Model) -> LlamaCppSession:
        """Load llama.cpp, construct the runtime, and optionally attach a tokenizer for chat formatting."""

        llama_module = _import_llama_cpp(config.llama_cpp_path)
        gpu_offload_supported = _llama_supports_gpu_offload(llama_module)
        effective_device = _resolve_effective_device(config.device, gpu_offload_supported)
        effective_n_gpu_layers = _resolve_n_gpu_layers(
            requested_n_gpu_layers=config.n_gpu_layers,
            effective_device=effective_device,
            gpu_offload_supported=gpu_offload_supported,
        )

        llm_kwargs = dict(config.llama_kwargs)
        llm_kwargs.update(model_config.model_kwargs)
        llm_kwargs.setdefault("model_path", model_config.path)
        llm_kwargs.setdefault("n_ctx", config.n_ctx)
        llm_kwargs.setdefault("n_batch", config.n_batch)
        llm_kwargs.setdefault("n_ubatch", config.n_ubatch)
        llm_kwargs.setdefault("n_gpu_layers", effective_n_gpu_layers)
        llm_kwargs.setdefault("main_gpu", config.main_gpu)
        llm_kwargs.setdefault("split_mode", config.split_mode)
        llm_kwargs.setdefault("tensor_split", config.tensor_split)
        llm_kwargs.setdefault("flash_attn", config.flash_attn)
        llm_kwargs.setdefault("offload_kqv", config.offload_kqv)
        llm_kwargs.setdefault("use_mmap", config.use_mmap)
        llm_kwargs.setdefault("use_mlock", config.use_mlock)
        llm_kwargs.setdefault("chat_format", config.chat_format)
        llm_kwargs.setdefault("verbose", config.verbose)
        llm_kwargs.setdefault("logits_all", config.logits_all)
        if config.n_threads is not None:
            llm_kwargs.setdefault("n_threads", config.n_threads)
        if config.n_threads_batch is not None:
            llm_kwargs.setdefault("n_threads_batch", config.n_threads_batch)
        if config.seed is not None:
            llm_kwargs.setdefault("seed", config.seed)

        llm = llama_module.Llama(**llm_kwargs)
        prepare_tokenizer = _maybe_load_prepare_tokenizer(
            model_config=model_config,
            trust_remote_code=(
                config.trust_remote_code
                if config.trust_remote_code is not None
                else model_config.trust_remote_code
            ),
            padding_side=config.padding_side,
        )
        return cls(
            config=config,
            model_config=model_config,
            llm=llm,
            llama_module=llama_module,
            prepare_tokenizer=prepare_tokenizer,
            effective_device=effective_device,
            gpu_offload_supported=gpu_offload_supported,
            effective_n_gpu_layers=effective_n_gpu_layers,
        )

    @property
    def batch_size(self) -> int | str:
        """Expose the configured batch size policy for compatibility with other engines."""

        return self.config.batch_size

    def describe_execution(self) -> dict[str, Any]:
        """Expose stable llama.cpp runtime metadata in result payloads and logs."""

        return {
            "generation_backend": "llama_cpp_completion",
            "continuous_batching": "queue_emulated",
            "device": self.effective_device,
            "gpu_offload_supported": self.gpu_offload_supported,
            "n_gpu_layers": self.effective_n_gpu_layers,
            "flash_attn": self.config.flash_attn,
            "max_model_len": self._max_scoring_input_length(),
        }

    def resolve_batch_size(self, requests: list[GenerationRequest]) -> int:
        """Resolve either an explicit batch size or a conservative auto heuristic."""

        configured_batch_size = _normalize_batch_size(self.config.batch_size)
        if configured_batch_size != _AUTO_BATCH_SIZE:
            return int(configured_batch_size)
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

        effective_batch_size = batch_size or self.resolve_batch_size(requests)
        outputs: list[GenerationOutput] = []
        with self._generation_lock:
            for start in range(0, len(requests), effective_batch_size):
                for request in requests[start : start + effective_batch_size]:
                    outputs.append(self._generate_one(request))
        return outputs

    def generate_continuous(
        self,
        requests: Iterable[tuple[Any, GenerationRequest]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        """Emulate continuous generation with explicit request and result queues."""

        def iterator() -> Iterator[tuple[Any, GenerationOutput]]:
            request_iter = iter(requests)
            if batch_size is not None:
                effective_batch_size = batch_size
                first_item = next(request_iter, None)
                if first_item is None:
                    return
                items = chain((first_item,), request_iter)
            else:
                preview_items = list(islice(request_iter, 64))
                if not preview_items:
                    return
                effective_batch_size = self.resolve_batch_size(
                    [request for _, request in preview_items]
                )
                items = chain(preview_items, request_iter)

            def consume_requests(
                stop_event: threading.Event,
                request_queue: Any,
                put_result: Any,
            ) -> None:
                """Drain queued requests in fixed-size chunks and emit completed outputs."""

                with self._generation_lock:
                    pending_batch: list[tuple[Any, GenerationRequest]] = []
                    for request_key, request in request_queue.iter_requests(stop_event=stop_event):
                        pending_batch.append((request_key, request))
                        if len(pending_batch) < effective_batch_size:
                            continue
                        self._emit_generated_batch(pending_batch, put_result)
                        pending_batch = []
                    if pending_batch:
                        self._emit_generated_batch(pending_batch, put_result)

            yield from stream_request_results(
                items,
                producer_name=f"{type(self).__name__}.request_producer",
                consumer_name=f"{type(self).__name__}.request_consumer",
                process_requests=consume_requests,
                require_non_main_thread=self.request_executor_requires_non_main_thread,
                request_queue_max_size=max(effective_batch_size * 2, 1),
            )

        return iterator()

    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]:
        """Score continuations through llama.cpp logits gathered from direct token evaluation."""

        if not requests:
            return []

        del batch_size
        prepared_requests = [self._prepare_loglikelihood_request(request) for request in requests]
        chunk_counts: list[int] = []
        chunk_outputs: list[list[LoglikelihoodOutput]] = []
        with self._generation_lock:
            for prefix_ids, target_ids, metadata in prepared_requests:
                request_chunks = self._build_loglikelihood_chunks(
                    prefix_ids=prefix_ids,
                    target_ids=target_ids,
                    metadata=metadata,
                )
                chunk_counts.append(len(request_chunks))
                for input_ids, score_start, score_count, chunk_metadata in request_chunks:
                    chunk_outputs.append(
                        [
                            self._score_loglikelihood_chunk(
                                input_ids=input_ids,
                                score_start=score_start,
                                score_count=score_count,
                                metadata=chunk_metadata,
                            )
                        ]
                    )

        outputs: list[LoglikelihoodOutput] = []
        cursor = 0
        for request, chunk_count in zip(requests, chunk_counts, strict=True):
            request_chunk_outputs = [
                chunk_outputs[cursor + index][0]
                for index in range(chunk_count)
            ]
            cursor += chunk_count
            outputs.append(
                LoglikelihoodOutput(
                    logprob=sum(output.logprob for output in request_chunk_outputs),
                    is_greedy=all(output.is_greedy for output in request_chunk_outputs),
                    token_count=sum(output.token_count for output in request_chunk_outputs),
                    metadata=dict(request.metadata),
                )
            )
        return outputs

    def loglikelihood_continuous(
        self,
        requests: Iterable[tuple[Any, LoglikelihoodRequest]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[Any, LoglikelihoodOutput]]:
        """Emulate continuous log-likelihood scoring with request and result queues."""

        def iterator() -> Iterator[tuple[Any, LoglikelihoodOutput]]:
            request_iter = iter(requests)
            if batch_size is not None:
                effective_batch_size = batch_size
                first_item = next(request_iter, None)
                if first_item is None:
                    return
                items = chain((first_item,), request_iter)
            else:
                preview_items = list(islice(request_iter, 64))
                if not preview_items:
                    return
                effective_batch_size = min(len(preview_items), _friendly_batch_size(len(preview_items)))
                items = chain(preview_items, request_iter)

            def consume_requests(
                stop_event: threading.Event,
                request_queue: Any,
                put_result: Any,
            ) -> None:
                """Drain queued requests in fixed-size chunks and emit scored outputs."""

                with self._generation_lock:
                    pending_batch: list[tuple[Any, LoglikelihoodRequest]] = []
                    for request_key, request in request_queue.iter_requests(stop_event=stop_event):
                        pending_batch.append((request_key, request))
                        if len(pending_batch) < effective_batch_size:
                            continue
                        self._emit_scored_batch(pending_batch, put_result)
                        pending_batch = []
                    if pending_batch:
                        self._emit_scored_batch(pending_batch, put_result)

            yield from stream_request_results(
                items,
                producer_name=f"{type(self).__name__}.loglikelihood_request_producer",
                consumer_name=f"{type(self).__name__}.loglikelihood_request_consumer",
                process_requests=consume_requests,
                require_non_main_thread=self.request_executor_requires_non_main_thread,
                request_queue_max_size=max(effective_batch_size * 2, 1),
            )

        return iterator()

    def loglikelihood_rolling(
        self,
        requests: list[RollingLoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]:
        """Score rolling log-likelihood requests token by token for perplexity-style suites."""

        del batch_size
        if not requests:
            return []

        scoring_requests: list[LoglikelihoodRequest] = []
        request_window_counts: list[int] = []
        with self._tokenizer_lock:
            for request in requests:
                token_list = (
                    list(request.input_ids)
                    if request.input_ids is not None
                    else self._tokenize_text(request.text, add_bos=False)
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

        scored = self.loglikelihood(scoring_requests)
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

            request_outputs = scored[cursor : cursor + window_count]
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
        """Reset reusable llama.cpp context state between suites."""

        with self._generation_lock:
            reset = getattr(self.llm, "reset", None)
            if callable(reset):
                with suppress(Exception):
                    reset()
        gc.collect()

    def close(self) -> None:
        """Release the runtime and tokenizer resources owned by this session."""

        with self._generation_lock:
            close = getattr(self.llm, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()
            with suppress(Exception):
                del self.llm
            with suppress(Exception):
                del self.prepare_tokenizer

    def _emit_generated_batch(
        self,
        batch: list[tuple[Any, GenerationRequest]],
        put_result: Any,
    ) -> None:
        """Run one fixed batch through generation and send outputs back to the queue bridge."""

        outputs = self.generate([request for _, request in batch], batch_size=len(batch))
        for (request_key, _request), output in zip(batch, outputs, strict=True):
            put_result(request_key, output)

    def _emit_scored_batch(
        self,
        batch: list[tuple[Any, LoglikelihoodRequest]],
        put_result: Any,
    ) -> None:
        """Run one fixed batch through log-likelihood scoring and send outputs back to the queue bridge."""

        outputs = self.loglikelihood([request for _, request in batch], batch_size=len(batch))
        for (request_key, _request), output in zip(batch, outputs, strict=True):
            put_result(request_key, output)

    def _generate_one(self, request: GenerationRequest) -> GenerationOutput:
        """Generate one completion using either the text or chat llama.cpp API."""

        if request.num_beams != 1:
            raise ValueError("LlamaCpp currently requires num_beams=1")

        if request.messages is not None and request.rendered_prompt is None and self.prepare_tokenizer is None:
            response = self.llm.create_chat_completion(
                messages=list(request.messages),
                max_tokens=request.max_new_tokens,
                temperature=request.temperature if request.do_sample else 0.0,
                stop=list(request.stop) if request.stop else None,
                seed=self.config.seed,
                stream=False,
                logprobs=False,
            )
            text = _extract_chat_completion_text(response)
            prompt_text = self._messages_display_prompt(request.messages)
            metadata = {
                **dict(request.metadata),
                "finish_reason": response["choices"][0].get("finish_reason"),
                "usage": response.get("usage"),
            }
            return GenerationOutput(prompt=prompt_text, text=text, metadata=metadata)

        prompt_text, prompt_tokens = self._prepare_generation_prompt(request)
        response = self.llm.create_completion(
            prompt=prompt_tokens if request.input_ids is not None else prompt_text,
            max_tokens=request.max_new_tokens,
            temperature=request.temperature if request.do_sample else 0.0,
            stop=list(request.stop) if request.stop else None,
            seed=self.config.seed,
            stream=False,
        )
        choice = response["choices"][0]
        text = _truncate_at_stop(choice["text"], request.stop)
        metadata = {
            **dict(request.metadata),
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage"),
        }
        return GenerationOutput(prompt=prompt_text, text=text, metadata=metadata)

    def _prepare_generation_prompt(self, request: GenerationRequest) -> tuple[str, list[int]]:
        """Render one request into the prompt text and prompt tokens consumed by llama.cpp."""

        if request.rendered_prompt is not None:
            prompt_text = request.rendered_prompt
        elif request.messages is not None:
            if self.prepare_tokenizer is not None:
                apply_chat_template = getattr(self.prepare_tokenizer, "apply_chat_template", None)
                if not callable(apply_chat_template):
                    raise ValueError("generation requests with messages require tokenizer.apply_chat_template")
                prompt_text = apply_chat_template(
                    request.messages,
                    tokenize=False,
                    add_generation_prompt=request.add_generation_prompt,
                )
            else:
                prompt_text = self._messages_display_prompt(request.messages)
        elif request.prompt is not None:
            prompt_text = request.prompt
        else:
            raise ValueError("generation requests must define either `prompt` or `messages`")

        prompt_tokens = (
            list(request.input_ids)
            if request.input_ids is not None
            else self._tokenize_text(prompt_text, add_bos=True)
        )
        return prompt_text, prompt_tokens

    def _prepare_loglikelihood_request(
        self,
        request: LoglikelihoodRequest,
    ) -> tuple[list[int], list[int], dict[str, Any]]:
        """Convert one scoring request into prefix ids, continuation ids, and copied metadata."""

        with self._tokenizer_lock:
            if request.context_input_ids is not None:
                prefix_ids = list(request.context_input_ids)
            elif request.context:
                prefix_ids = self._tokenize_text(request.context, add_bos=True)
            else:
                prefix_ids = []

            if request.continuation_input_ids is not None:
                target_ids = list(request.continuation_input_ids)
            elif request.continuation:
                target_ids = self._tokenize_text(request.continuation, add_bos=False)
            else:
                target_ids = []

        if not target_ids:
            raise ValueError("loglikelihood requests must provide a non-empty continuation")
        return prefix_ids, target_ids, dict(request.metadata)

    def _build_loglikelihood_chunks(
        self,
        *,
        prefix_ids: list[int],
        target_ids: list[int],
        metadata: dict[str, Any],
    ) -> list[tuple[list[int], int, int, dict[str, Any]]]:
        """Split a scored continuation into disjoint windows that fit the active context length."""

        max_input_length = self._max_scoring_input_length()
        max_scored_window = max_input_length + 1
        history_ids = list(prefix_ids)
        if not history_ids:
            history_ids = [self._prefix_token_id()]

        chunks: list[tuple[list[int], int, int, dict[str, Any]]] = []
        cursor = 0
        while cursor < len(target_ids):
            remaining = len(target_ids) - cursor
            target_count = min(remaining, max_input_length)
            context_count = min(len(history_ids), max_scored_window - target_count)
            if context_count <= 0:
                raise ValueError("model context window is too small to score continuation tokens")

            context_ids = history_ids[-context_count:]
            continuation_slice = target_ids[cursor : cursor + target_count]
            chunks.append(
                (
                    context_ids + continuation_slice,
                    len(context_ids),
                    len(continuation_slice),
                    dict(metadata),
                )
            )
            history_ids.extend(continuation_slice)
            cursor += target_count
        return chunks

    def _score_loglikelihood_chunk(
        self,
        *,
        input_ids: list[int],
        score_start: int,
        score_count: int,
        metadata: dict[str, Any],
    ) -> LoglikelihoodOutput:
        """Evaluate one scoring chunk and gather the exact continuation token log-probabilities."""

        if len(input_ids) > self._max_input_tokens():
            raise ValueError("scoring chunk exceeds llama.cpp context window")

        self.llm.reset()
        self.llm.eval(input_ids)
        logprobs = self.llama_module.Llama.logits_to_logprobs(self.llm._scores)

        total_logprob = 0.0
        is_greedy = True
        for offset in range(score_count):
            token_position = score_start + offset
            token_id = int(input_ids[token_position])
            token_logprobs = logprobs[token_position - 1]
            total_logprob += float(token_logprobs[token_id])
            if int(token_logprobs.argmax()) != token_id:
                is_greedy = False

        return LoglikelihoodOutput(
            logprob=total_logprob,
            is_greedy=is_greedy,
            token_count=score_count,
            metadata=dict(metadata),
        )

    def _rolling_loglikelihood_windows(
        self,
        token_list: list[int],
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Split one token list into the disjoint scoring windows used for rolling perplexity."""

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
                token_list[window_end - max_seq_len - 1 : window_end - 1],
                token_list[window_end - window_pred_len : window_end],
            )
            yield self._make_disjoint_window(window)
            predicted += window_pred_len

    def _make_disjoint_window(
        self,
        pair: tuple[list[int], list[int]],
    ) -> tuple[list[int], list[int]]:
        """Trim overlapping context tokens because the scorer already conditions on their prefix."""

        context_ids, continuation_ids = pair
        return context_ids[: len(context_ids) - (len(continuation_ids) - 1)], continuation_ids

    def _prefix_token_id(self) -> int:
        """Return the synthetic prefix token used when a scored request has no explicit context."""

        token_bos = getattr(self.llm, "token_bos", None)
        if callable(token_bos):
            return int(token_bos())
        return 1

    def _max_input_tokens(self) -> int:
        """Return the runtime context window used by llama.cpp for prompt evaluation."""

        n_ctx = getattr(self.llm, "n_ctx", None)
        if callable(n_ctx):
            return int(n_ctx())
        runtime_n_ctx = getattr(self.llm, "_n_ctx", None)
        if isinstance(runtime_n_ctx, int):
            return runtime_n_ctx
        return int(self.config.n_ctx)

    def _max_scoring_input_length(self) -> int:
        """Reserve one token position for next-token prediction during log-likelihood scoring."""

        return max(self._max_input_tokens() - 1, 1)

    def _tokenize_text(self, text: str, *, add_bos: bool) -> list[int]:
        """Tokenize text through llama.cpp while keeping BOS control explicit per call site."""

        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=False))

    def _messages_display_prompt(self, messages: list[dict[str, str]]) -> str:
        """Materialize a stable display prompt when llama.cpp handles chat templating internally."""

        return "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )


def _import_llama_cpp(llama_cpp_path: str | None) -> Any:
    """Import llama-cpp-python and optionally fall back to a local checkout path."""

    explicit_candidates = [Path(llama_cpp_path)] if llama_cpp_path is not None else []
    candidates = [candidate for candidate in explicit_candidates + list(_DEFAULT_LLAMA_CPP_CHECKOUT_CANDIDATES)]

    try:
        return importlib.import_module("llama_cpp")
    except ModuleNotFoundError as original_exc:
        for candidate in candidates:
            if not candidate.exists():
                continue
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            with suppress(ModuleNotFoundError):
                return importlib.import_module("llama_cpp")
        raise ModuleNotFoundError(
            "LlamaCpp engine requires the optional `llama-cpp-python` dependency"
        ) from original_exc


def _llama_supports_gpu_offload(llama_module: Any) -> bool:
    """Detect whether the installed llama.cpp build was compiled with GPU offload support."""

    llama_cpp_low_level = getattr(llama_module, "llama_cpp", None)
    if llama_cpp_low_level is None:
        return False
    supports_gpu_offload = getattr(llama_cpp_low_level, "llama_supports_gpu_offload", None)
    if not callable(supports_gpu_offload):
        return False
    with suppress(Exception):
        return bool(supports_gpu_offload())
    return False


def _resolve_effective_device(requested_device: str | None, gpu_offload_supported: bool) -> str:
    """Resolve the runtime device selection while validating CUDA-specific requests early."""

    normalized = (requested_device or "").strip().lower()
    if normalized in {"", "auto"}:
        return "cuda" if gpu_offload_supported else "cpu"
    if normalized in {"cuda", "gpu"} and not gpu_offload_supported:
        raise RuntimeError(
            "LlamaCpp requested CUDA execution but the installed llama.cpp build does not support GPU offload"
        )
    if normalized in {"cuda", "gpu"}:
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    raise ValueError(f"unsupported LlamaCpp device: {requested_device!r}")


def _resolve_n_gpu_layers(
    *,
    requested_n_gpu_layers: int | None,
    effective_device: str,
    gpu_offload_supported: bool,
) -> int:
    """Resolve how aggressively llama.cpp should offload layers to the visible GPU."""

    if requested_n_gpu_layers is not None:
        if requested_n_gpu_layers > 0 and not gpu_offload_supported:
            raise RuntimeError(
                "LlamaCpp requested GPU layer offload but the installed llama.cpp build does not support it"
            )
        return requested_n_gpu_layers
    if effective_device == "cuda" and gpu_offload_supported:
        return -1
    return 0


def _maybe_load_prepare_tokenizer(
    *,
    model_config: Model,
    trust_remote_code: bool,
    padding_side: str,
) -> Any | None:
    """Load an optional tokenizer for chat template rendering when the caller supplies one explicitly."""

    tokenizer_source = _resolve_tokenizer_source(model_config)
    if tokenizer_source is None:
        return None
    if model_config.tokenizer is None and model_config.tokenizer_path is None:
        model_path = Path(model_config.path)
        if model_path.suffix.lower() == ".gguf":
            return None
    with suppress(Exception):
        tokenizer = _load_tokenizer_from_model(
            tokenizer_source,
            revision=model_config.revision,
            trust_remote_code=trust_remote_code,
            **model_config.tokenizer_kwargs,
        )
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = padding_side
        return tokenizer
    return None


def _extract_chat_completion_text(response: dict[str, Any]) -> str:
    """Normalize llama.cpp chat-completion payloads into one plain completion string."""

    message = response["choices"][0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content)
