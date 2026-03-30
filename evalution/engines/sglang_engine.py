# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import asyncio
import gc
import importlib
import queue
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from itertools import chain, islice
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evalution.config import Model
from evalution.engines.base import (
    BaseEngine,
    BaseInferenceSession,
    GenerationOutput,
    GenerationRequest,
    LoglikelihoodOutput,
    LoglikelihoodRequest,
    RollingLoglikelihoodOutput,
    RollingLoglikelihoodRequest,
)
from evalution.engines.continuous import stream_request_results
from evalution.engines.transformers_common import (
    _AUTO_BATCH_SIZE,
    _ScoringChunk,
    _clone_prepare_tokenizer,
    _friendly_batch_size,
    _load_tokenizer_from_model,
    _normalize_batch_size,
    _normalize_tokenizer_special_tokens,
    _resolve_tokenizer_source,
    _seed_transformer_runtime,
    _seed_with_internal_apis,
)
from evalution.logbar import get_logger


class _SGLangClient(ABC):
    @abstractmethod
    def generate(self, **payload: Any) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def async_generate(self, **payload: Any) -> dict[str, Any]: ...

    def gc(self) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(slots=True)
class _SGLangPythonClient(_SGLangClient):
    engine: Any

    def generate(self, **payload: Any) -> list[dict[str, Any]]:
        response = self.engine.generate(**payload)
        return _normalize_sglang_response(response)

    async def async_generate(self, **payload: Any) -> dict[str, Any]:
        response = await self.engine.async_generate(**payload)
        normalized = _normalize_sglang_response(response)
        if len(normalized) != 1:
            raise RuntimeError("sglang async_generate returned an unexpected batched response")
        return normalized[0]

    def gc(self) -> None:
        flush_cache = getattr(self.engine, "flush_cache", None)
        if callable(flush_cache):
            flush_cache()

    def close(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


@dataclass(slots=True)
class _LoadedSGLangRuntime:
    client: _SGLangClient
    tokenizer: Any
    prepare_tokenizer: Any | None
    model: Any
    input_device: Any


@dataclass(slots=True)
class SGLang(BaseEngine):
    # SGLang integration stays in-process through `sglang.Engine`; no HTTP server is used.
    # Expose SGLang runtime kwargs using the same names as ServerArgs / Engine kwargs.
    dtype: str | None = "auto"
    device: str | None = None
    seed: int | None = None
    trust_remote_code: bool | None = None
    padding_side: str = "left"
    resolved_engine: str | None = field(default=None, init=False)
    base_url: str | None = None
    batch_size: int | str = "auto"
    max_new_tokens: int = 256
    tokenizer_mode: str = "auto"
    tokenizer_worker_num: int = 1
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    context_length: int | None = None
    quantization: str | None = None
    mem_fraction_static: float | None = None
    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    attention_backend: str | None = None
    sampling_backend: str | None = None
    max_running_requests: int | None = None
    max_total_tokens: int | None = None
    sampling_params: dict[str, Any] = field(default_factory=dict)

    def build(self, model: Model) -> BaseInferenceSession:
        self.resolved_engine = "SGLang"
        return SGLangSession.from_config(self, model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SGLangSession(BaseInferenceSession):
    config: SGLang
    model_config: Model
    model: Any
    tokenizer: Any
    input_device: Any
    prepare_tokenizer: Any | None = None
    requested_attn_implementation: str | None = None
    effective_attn_implementation: str | None = None
    paged_attention_enabled: bool = False
    generation_backend: str = "sglang.generate"
    standard_batch_size_cap: int | None = None
    stop_criteria_cache: dict[tuple[str, ...], Any] = field(default_factory=dict, repr=False)
    auto_batch_size_cache: dict[tuple[Any, ...], int] = field(default_factory=dict, repr=False)
    execution_logged: bool = field(default=False, repr=False)
    client: _SGLangClient | None = field(default=None, repr=False)
    _generation_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _tokenizer_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _prepare_tokenizer_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    @classmethod
    def from_config(cls, config: SGLang, model_config: Model) -> SGLangSession:
        runtime = load_sglang_runtime(config, model_config)
        return cls(
            config=config,
            model_config=model_config,
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            prepare_tokenizer=runtime.prepare_tokenizer,
            input_device=runtime.input_device,
            generation_backend="sglang.generate",
            client=runtime.client,
        )

    @property
    def batch_size(self) -> int | str:
        return self.config.batch_size

    def describe_execution(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "requested_attn_implementation": self.requested_attn_implementation,
                "effective_attn_implementation": self.effective_attn_implementation,
                "paged_attention": self.paged_attention_enabled,
                "generation_backend": self.generation_backend,
                "logprob_backend": "sglang.generate",
                "standard_batch_size_cap": self.standard_batch_size_cap,
            }

    def generate(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int | None = None,
    ) -> list[GenerationOutput]:
        """Generate completions for a fixed request list through the in-process SGLang runtime."""

        if not requests:
            return []

        with self._generation_lock:
            effective_batch_size = batch_size or self.resolve_batch_size(requests)
            with self._state_lock:
                standard_batch_size_cap = self.standard_batch_size_cap
            if standard_batch_size_cap is not None:
                effective_batch_size = min(effective_batch_size, standard_batch_size_cap)
            self._log_generation_execution()
            return self._generate_standard(requests, batch_size=effective_batch_size)

    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]:
        """Score continuation logprobs using SGLang's prompt-logprob metadata."""

        if not requests:
            return []

        with self._generation_lock:
            prepared_requests = [self._prepare_loglikelihood_request(request) for request in requests]
            effective_batch_size = batch_size or self._resolve_scoring_batch_size(prepared_requests)
            return self._score_prepared_loglikelihood_requests(
                prepared_requests,
                batch_size=effective_batch_size,
            )

    def loglikelihood_continuous(
        self,
        requests: Iterable[tuple[Any, LoglikelihoodRequest]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[Any, LoglikelihoodOutput]]:
        """Keep fixed-size scoring batches full while the caller streams requests lazily."""

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
                preview_prepared = [
                    self._prepare_loglikelihood_request(request)
                    for _request_key, request in preview_items
                ]
                effective_batch_size = self._resolve_scoring_batch_size(preview_prepared)
                items = chain(preview_items, request_iter)

            def consume_requests(
                stop_event: threading.Event,
                request_queue: Any,
                put_result: Any,
            ) -> None:
                with self._generation_lock:
                    prepared_batch: list[tuple[Any, tuple[list[int], list[int], dict[str, Any]]]] = []
                    for request_key, request in request_queue.iter_requests(stop_event=stop_event):
                        prepared_batch.append(
                            (
                                request_key,
                                self._prepare_loglikelihood_request(request),
                            )
                        )
                        if len(prepared_batch) < effective_batch_size:
                            continue
                        self._emit_scored_loglikelihood_batch(prepared_batch, put_result)
                        prepared_batch = []

                    if prepared_batch:
                        self._emit_scored_loglikelihood_batch(prepared_batch, put_result)

            yield from stream_request_results(
                items,
                producer_name=f"{type(self).__name__}.loglikelihood_request_producer",
                consumer_name=f"{type(self).__name__}.loglikelihood_request_consumer",
                process_requests=consume_requests,
                require_non_main_thread=self.request_executor_requires_non_main_thread,
                request_queue_max_size=max(effective_batch_size * 2, 1),
            )

        return iterator()

    def generate_continuous(
        self,
        requests: Iterable[tuple[Any, GenerationRequest]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        """Yield completions in finish order while keeping SGLang work in process."""

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
                    [request for _request_key, request in preview_items]
                )
                items = chain(preview_items, request_iter)

            with self._generation_lock:
                with self._state_lock:
                    standard_batch_size_cap = self.standard_batch_size_cap
                if standard_batch_size_cap is not None:
                    effective_batch_size = min(effective_batch_size, standard_batch_size_cap)
                self._log_generation_execution()
                yield from self._generate_sglang_continuous(
                    items,
                    batch_size=effective_batch_size,
                )

        return iterator()

    def loglikelihood_rolling(
        self,
        requests: list[RollingLoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]:
        """Score full text spans token by token via the shared rolling-loglikelihood path."""

        if not requests:
            return []

        scoring_requests: list[LoglikelihoodRequest] = []
        request_window_counts: list[int] = []
        with self._tokenizer_lock:
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

    def close(self) -> None:
        try:
            with self._generation_lock:
                with self._prepare_tokenizer_lock:
                    with self._tokenizer_lock:
                        with self._state_lock:
                            self.stop_criteria_cache.clear()
                            self.auto_batch_size_cache.clear()
                            with suppress(Exception):
                                del self.model
                            with suppress(Exception):
                                del self.tokenizer
                            with suppress(Exception):
                                del self.prepare_tokenizer
            self.gc()
        finally:
            if self.client is not None:
                self.client.close()

    def gc(self) -> None:
        with self._generation_lock:
            with self._state_lock:
                self.stop_criteria_cache.clear()
                self.auto_batch_size_cache.clear()
                self.execution_logged = False
        gc.collect()
        with suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
        if self.client is not None:
            self.client.gc()

    def _render_request(self, request: GenerationRequest) -> str:
        if request.rendered_prompt is not None:
            return request.rendered_prompt
        return self._render_request_with_tokenizer(self.tokenizer, request)

    def _render_request_with_tokenizer(self, tokenizer: Any, request: GenerationRequest) -> str:
        if request.messages is not None:
            return tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
            )
        if request.prompt is None:
            raise ValueError("generation requests must define either `prompt` or `messages`")
        return request.prompt

    def prepare_requests(self, requests: list[GenerationRequest]) -> list[GenerationRequest]:
        tokenizer = self.prepare_tokenizer or self.tokenizer
        tokenizer_lock = self._lock_for_tokenizer(tokenizer)
        with tokenizer_lock:
            prepared: list[GenerationRequest] = list(requests)
            missing_indexes: list[int] = []
            missing_prompts: list[str] = []

            for index, request in enumerate(prepared):
                if request.rendered_prompt is not None and request.input_ids is not None:
                    continue
                missing_indexes.append(index)
                missing_prompts.append(
                    request.rendered_prompt
                    if request.rendered_prompt is not None
                    else self._render_request_with_tokenizer(tokenizer, request)
                )

            if not missing_indexes:
                return prepared

            encoded = tokenizer(
                missing_prompts,
                add_special_tokens=False,
                padding=False,
            )["input_ids"]
            for index, rendered_prompt, input_ids in zip(
                missing_indexes,
                missing_prompts,
                encoded,
                strict=True,
            ):
                prepared[index] = replace(
                    prepared[index],
                    rendered_prompt=rendered_prompt,
                    input_ids=list(input_ids),
                )
            del encoded
            return prepared

    def resolve_batch_size(self, requests: list[GenerationRequest]) -> int:
        configured_batch_size = _normalize_batch_size(self.config.batch_size)
        if configured_batch_size != _AUTO_BATCH_SIZE:
            return int(configured_batch_size)
        return min(len(requests) or 1, _friendly_batch_size(len(requests) or 1))

    def _prepare_loglikelihood_request(
        self,
        request: LoglikelihoodRequest,
    ) -> tuple[list[int], list[int], dict[str, Any]]:
        with self._tokenizer_lock:
            if request.context_input_ids is not None:
                prefix_ids = list(request.context_input_ids)
            elif request.context:
                prefix_ids = self._tokenize_loglikelihood_context(request.context)
            else:
                prefix_ids = []

            if request.continuation_input_ids is not None:
                target_ids = list(request.continuation_input_ids)
            elif request.continuation:
                target_ids = self.tokenizer(
                    request.continuation,
                    add_special_tokens=False,
                )["input_ids"]
            else:
                target_ids = []

        if not target_ids:
            raise ValueError("loglikelihood requests must provide a non-empty continuation")
        return list(prefix_ids), list(target_ids), dict(request.metadata)

    def _loglikelihood_request_sort_key(
        self,
        item: tuple[int, tuple[list[int], list[int], dict[str, Any]]],
    ) -> tuple[int, tuple[int, ...]]:
        _request_index, (prefix_ids, target_ids, _metadata) = item
        combined = tuple(prefix_ids + target_ids)
        return (-len(combined), combined)

    def _tokenize_loglikelihood_context(self, text: str) -> list[int]:
        tokenizer_kwargs: dict[str, Any] = {}
        prefix_text = self._decoded_prefix_token_text()
        if prefix_text and text.startswith(prefix_text):
            tokenizer_kwargs["add_special_tokens"] = False
        return list(self.tokenizer(text, **tokenizer_kwargs)["input_ids"])

    def _resolve_scoring_batch_size(
        self,
        requests: list[tuple[list[int], list[int], dict[str, Any]]],
    ) -> int:
        batch_requests = [
            GenerationRequest(
                rendered_prompt="",
                input_ids=prefix_ids + target_ids,
                max_new_tokens=0,
            )
            for prefix_ids, target_ids, _metadata in requests
        ]
        return self.resolve_batch_size(batch_requests)

    def _build_loglikelihood_chunks(
        self,
        *,
        request_index: int,
        prefix_ids: list[int],
        target_ids: list[int],
        metadata: dict[str, Any],
    ) -> list[_ScoringChunk]:
        max_input_length = self._max_scoring_input_length()
        max_scored_window = max_input_length + 1
        history_ids = list(prefix_ids)
        if not history_ids:
            history_ids = [self._prefix_token_id()]

        chunks: list[_ScoringChunk] = []
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
                _ScoringChunk(
                    request_index=request_index,
                    input_ids=context_ids + continuation_slice,
                    score_start=len(context_ids),
                    score_count=len(continuation_slice),
                    metadata=dict(metadata),
                )
            )
            history_ids.extend(continuation_slice)
            cursor += target_count
        return chunks

    def _rolling_loglikelihood_windows(
        self,
        token_list: list[int],
    ) -> Any:
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
        context_ids, continuation_ids = pair
        return context_ids[: len(context_ids) - (len(continuation_ids) - 1)], continuation_ids

    def _prefix_token_id(self) -> int:
        for token_id in (
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
        ):
            if token_id is not None:
                return int(token_id)
        raise ValueError(
            "token-level scoring requires tokenizer.bos_token_id, eos_token_id, or pad_token_id"
        )

    def _decoded_prefix_token_text(self) -> str | None:
        with suppress(Exception):
            decoded = self.tokenizer.decode(
                [self._prefix_token_id()],
                skip_special_tokens=False,
            )
            if decoded:
                return str(decoded)
        with suppress(Exception):
            decoded = self.tokenizer.decode(
                self._prefix_token_id(),
                skip_special_tokens=False,
            )
            if decoded:
                return str(decoded)
        return None

    def _max_scoring_input_length(self) -> int:
        model_length = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        tokenizer_length = getattr(self.tokenizer, "model_max_length", None)
        candidate_lengths = [
            int(length)
            for length in (model_length, tokenizer_length)
            if isinstance(length, int) and 1 < length < 1_000_000
        ]
        if candidate_lengths:
            return min(candidate_lengths)
        return 2048

    def _log_generation_execution(self) -> None:
        with self._state_lock:
            if self.execution_logged:
                return
            backend = self.generation_backend
            attention = self.effective_attn_implementation or self.requested_attn_implementation
            batch_size_cap = self.standard_batch_size_cap
            self.execution_logged = True
        get_logger().info(
            "transformer generation using backend=%s attention=%s batch_size_cap=%s",
            backend,
            attention,
            batch_size_cap,
        )

    def _lock_for_tokenizer(self, tokenizer: Any) -> threading.RLock:
        if tokenizer is not None and tokenizer is self.prepare_tokenizer and tokenizer is not self.tokenizer:
            return self._prepare_tokenizer_lock
        return self._tokenizer_lock

    def _prepare_request_for_generation(self, request: GenerationRequest) -> tuple[str, list[int]]:
        with self._tokenizer_lock:
            rendered_prompt = self._render_request(request)
            input_ids = (
                list(request.input_ids)
                if request.input_ids is not None
                else self.tokenizer(
                    rendered_prompt,
                    add_special_tokens=False,
                )["input_ids"]
            )
        return rendered_prompt, input_ids

    def _generate_standard(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int,
    ) -> list[GenerationOutput]:
        prepared = self.prepare_requests(requests)
        outputs: list[GenerationOutput] = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            payload = {
                "input_ids": [list(request.input_ids) for request in batch],
                "sampling_params": [self._build_sampling_params(request) for request in batch],
            }
            responses = self.client.generate(**payload)
            if len(responses) != len(batch):
                raise RuntimeError("sglang returned an unexpected number of generation results")
            for request, response in zip(batch, responses, strict=True):
                outputs.append(self._generation_output_from_response(request, response))
        return outputs

    def _generate_sglang_continuous(
        self,
        requests: Iterable[tuple[Any, GenerationRequest]],
        *,
        batch_size: int,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        result_queue: queue.Queue[Any] = queue.Queue()
        stop_event = threading.Event()
        sentinel = object()

        def worker() -> None:
            async def run() -> None:
                request_iter = iter(requests)
                pending: set[asyncio.Task[tuple[Any, GenerationRequest, dict[str, Any]]]] = set()

                async def submit_one() -> bool:
                    if stop_event.is_set():
                        return False
                    try:
                        request_key, request = next(request_iter)
                    except StopIteration:
                        return False

                    async def execute_one() -> tuple[Any, GenerationRequest, dict[str, Any]]:
                        rendered_prompt, input_ids = self._prepare_request_for_generation(request)
                        response = await self.client.async_generate(
                            input_ids=list(input_ids),
                            sampling_params=self._build_sampling_params(request),
                        )
                        response.setdefault("_evalution_rendered_prompt", rendered_prompt)
                        return request_key, request, response

                    pending.add(asyncio.create_task(execute_one()))
                    return True

                while len(pending) < batch_size and await submit_one():
                    continue

                while pending and not stop_event.is_set():
                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        request_key, request, response = await task
                        result_queue.put(
                            (request_key, self._generation_output_from_response(request, response))
                        )
                    while len(pending) < batch_size and await submit_one():
                        continue

                if pending:
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

            try:
                asyncio.run(run())
            except Exception as exc:
                result_queue.put(exc)
            finally:
                result_queue.put(sentinel)

        worker_thread = threading.Thread(
            target=worker,
            name="evalution-sglang-continuous",
            daemon=True,
        )
        worker_thread.start()

        try:
            while True:
                item = result_queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop_event.set()
            worker_thread.join()

    def _generation_output_from_response(
        self,
        request: GenerationRequest,
        response: dict[str, Any],
    ) -> GenerationOutput:
        meta_info = dict(response.get("meta_info") or {})
        text = str(response.get("text") or "")
        prompt = (
            response.get("_evalution_rendered_prompt")
            or request.rendered_prompt
            or self._render_request(request)
        )
        # In-process SGLang can return prompt + completion in the same text field.
        # Evalution exposes only the completion to match the other engines.
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return GenerationOutput(
            prompt=prompt,
            text=text,
            metadata={
                **dict(request.metadata),
                "sglang_meta": meta_info,
            },
        )

    def _score_chunks(
        self,
        chunks: list[Any],
        *,
        batch_size: int,
    ) -> list[LoglikelihoodOutput]:
        scored_chunks: list[LoglikelihoodOutput] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            payload = {
                "input_ids": [list(chunk.input_ids) for chunk in batch],
                "sampling_params": [
                    {
                        # Mirror lm-eval's SGLang path: prompt logprobs are requested in a
                        # non-generation scoring mode that still asks the engine to decode one token.
                        "max_new_tokens": 1,
                        "temperature": 0.0,
                    }
                    for _ in batch
                ],
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 2,
                "token_ids_logprob": [
                    _deduplicate_preserve_order(
                        chunk.input_ids[chunk.score_start : chunk.score_start + chunk.score_count]
                    )
                    for chunk in batch
                ],
            }
            responses = self.client.generate(**payload)
            if len(responses) != len(batch):
                raise RuntimeError("sglang returned an unexpected number of scoring results")

            for chunk, response in zip(batch, responses, strict=True):
                meta_info = dict(response.get("meta_info") or {})
                target_ids = chunk.input_ids[chunk.score_start : chunk.score_start + chunk.score_count]
                all_input_token_logprobs = _coerce_position_entries(meta_info.get("input_token_logprobs"))
                if len(all_input_token_logprobs) == chunk.score_count:
                    input_token_logprobs = all_input_token_logprobs
                else:
                    input_token_logprobs = all_input_token_logprobs[
                        chunk.score_start : chunk.score_start + chunk.score_count
                    ]
                if len(input_token_logprobs) != chunk.score_count:
                    raise RuntimeError("sglang returned too few input_token_logprobs for scoring")

                all_top_logprobs = _coerce_nested_position_entries(meta_info.get("input_top_logprobs"))
                if len(all_top_logprobs) == chunk.score_count:
                    top_logprobs = all_top_logprobs
                else:
                    top_logprobs = all_top_logprobs[
                        chunk.score_start : chunk.score_start + chunk.score_count
                    ]
                requested_logprobs = _coerce_nested_position_entries(
                    meta_info.get("input_token_ids_logprobs")
                )
                requested_logits = _coerce_nested_position_entries(
                    meta_info.get("input_token_ids_logits")
                )
                token_logits = _coerce_position_entries(meta_info.get("input_token_logits"))
                output_top_logprobs = _coerce_nested_position_entries(meta_info.get("output_top_logprobs"))
                output_requested_logprobs = _coerce_nested_position_entries(
                    meta_info.get("output_token_ids_logprobs")
                )

                total_logprob = 0.0
                is_greedy = True
                for position, target_id in enumerate(target_ids):
                    selected = input_token_logprobs[position]
                    selected_token_id = selected.get("token_id")
                    # Different SGLang builds expose the requested-token score in slightly
                    # different fields, so we fall back through the public shapes here.
                    requested_scores = _entries_to_score_map(
                        requested_logprobs[position] if position < len(requested_logprobs) else []
                    )
                    if not requested_scores:
                        requested_scores = _entries_to_score_map(
                            output_requested_logprobs[position]
                            if position < len(output_requested_logprobs)
                            else []
                        )
                    selected_score = selected.get("score")
                    if selected_score is None and selected_token_id is not None and int(selected_token_id) == int(target_id):
                        selected_score = requested_scores.get(int(target_id))
                    if selected_score is None:
                        raise RuntimeError(
                            "sglang did not provide a usable logprob for the requested continuation token"
                        )
                    total_logprob += float(selected_score)
                    top_choice = (
                        top_logprobs[position][0]
                        if position < len(top_logprobs) and top_logprobs[position]
                        else (
                            output_top_logprobs[position][0]
                            if position < len(output_top_logprobs) and output_top_logprobs[position]
                            else None
                        )
                    )
                    greedy_token_id = top_choice["token_id"] if top_choice is not None else None
                    if greedy_token_id is not None and greedy_token_id != int(target_id):
                        is_greedy = False

                scored_chunks.append(
                    LoglikelihoodOutput(
                        logprob=total_logprob,
                        is_greedy=is_greedy,
                        token_count=chunk.score_count,
                        metadata={
                            **dict(chunk.metadata),
                        },
                    )
                )
        return scored_chunks

    def _score_prepared_loglikelihood_requests(
        self,
        prepared_requests: list[tuple[list[int], list[int], dict[str, Any]]],
        *,
        batch_size: int,
    ) -> list[LoglikelihoodOutput]:
        chunks: list[_ScoringChunk] = []
        ordered_requests = list(enumerate(prepared_requests))
        ordered_requests.sort(key=self._loglikelihood_request_sort_key)
        for request_index, (prefix_ids, target_ids, metadata) in ordered_requests:
            chunks.extend(
                self._build_loglikelihood_chunks(
                    request_index=request_index,
                    prefix_ids=prefix_ids,
                    target_ids=target_ids,
                    metadata=metadata,
                )
            )

        chunk_scores = self._score_chunks(chunks, batch_size=batch_size)
        totals = [
            LoglikelihoodOutput(
                logprob=0.0,
                is_greedy=True,
                token_count=0,
                metadata=dict(metadata),
            )
            for _prefix_ids, _target_ids, metadata in prepared_requests
        ]
        for chunk, chunk_output in zip(chunks, chunk_scores, strict=True):
            current = totals[chunk.request_index]
            totals[chunk.request_index] = LoglikelihoodOutput(
                logprob=current.logprob + chunk_output.logprob,
                is_greedy=current.is_greedy and chunk_output.is_greedy,
                token_count=current.token_count + chunk_output.token_count,
                metadata=current.metadata,
            )
        return totals

    def _emit_scored_loglikelihood_batch(
        self,
        prepared_batch: list[tuple[Any, tuple[list[int], list[int], dict[str, Any]]]],
        put_result: Any,
    ) -> None:
        batch_outputs = self._score_prepared_loglikelihood_requests(
            [prepared_request for _request_key, prepared_request in prepared_batch],
            batch_size=len(prepared_batch),
        )
        for (request_key, _prepared_request), output in zip(prepared_batch, batch_outputs, strict=True):
            put_result(request_key, output)

    def _build_sampling_params(self, request: GenerationRequest) -> dict[str, Any]:
        if request.num_beams != 1:
            raise ValueError("sglang engine does not support beam search; set num_beams=1")
        params = {
            "max_new_tokens": request.max_new_tokens
            if request.max_new_tokens is not None
            else self.config.max_new_tokens,
            "temperature": request.temperature if request.do_sample else 0.0,
        }

        if self.config.sampling_params:
            params.update(self.config.sampling_params)
        
        if request.stop:
            params["stop"] = list(request.stop)
        return params


def load_sglang_runtime(config: SGLang, model_config: Model) -> _LoadedSGLangRuntime:
    _seed_transformer_runtime(config.seed)

    trust_remote_code = (
        config.trust_remote_code
        if config.trust_remote_code is not None
        else model_config.trust_remote_code
    )
    tokenizer = _load_tokenizer_from_model(
        _resolve_tokenizer_source(model_config),
        revision=model_config.revision,
        trust_remote_code=trust_remote_code,
        **model_config.tokenizer_kwargs,
    )
    tokenizer.padding_side = config.padding_side
    _normalize_tokenizer_special_tokens(tokenizer=tokenizer)

    prepare_tokenizer = _clone_prepare_tokenizer(
        tokenizer=tokenizer,
        model_config=model_config,
        trust_remote_code=trust_remote_code,
        model=None,
    )

    client = _build_sglang_client(config, model_config)
    _seed_with_internal_apis(getattr(client, "engine", None), config.seed)
    model_length = _resolve_sglang_context_length(client, tokenizer=tokenizer)
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=model_length))
    input_device = SimpleNamespace(type="cpu")

    return _LoadedSGLangRuntime(
        client=client,
        tokenizer=tokenizer,
        prepare_tokenizer=prepare_tokenizer,
        model=model,
        input_device=input_device,
    )


def _build_sglang_client(config: SGLang, model_config: Model) -> _SGLangClient:
    if config.base_url is not None:
        raise ValueError("sglang engine no longer supports server/http mode; use in-process Engine")

    engine_module = importlib.import_module("sglang.srt.entrypoints.engine")
    engine_kwargs = dict(model_config.model_kwargs)
    engine_kwargs.setdefault("model_path", model_config.path)
    engine_kwargs.setdefault(
        "trust_remote_code",
        (
            config.trust_remote_code
            if config.trust_remote_code is not None
            else model_config.trust_remote_code
        ),
    )
    tokenizer_source = _resolve_tokenizer_source(model_config)
    if tokenizer_source != model_config.path:
        engine_kwargs.setdefault("tokenizer_path", tokenizer_source)
    if config.device is not None:
        engine_kwargs.setdefault("device", config.device)
    if config.dtype is not None:
        engine_kwargs.setdefault("dtype", config.dtype)
    # Map Evalution's explicit SGLang fields onto the native Engine kwargs.
    engine_kwargs.setdefault("tokenizer_mode", config.tokenizer_mode)
    engine_kwargs.setdefault("tokenizer_worker_num", config.tokenizer_worker_num)
    engine_kwargs.setdefault("skip_tokenizer_init", config.skip_tokenizer_init)
    engine_kwargs.setdefault("load_format", config.load_format)
    engine_kwargs.setdefault("tp_size", config.tp_size)
    engine_kwargs.setdefault("dp_size", config.dp_size)
    engine_kwargs.setdefault("pp_size", config.pp_size)
    if model_config.revision is not None:
        engine_kwargs.setdefault("revision", model_config.revision)
    if config.context_length is not None:
        engine_kwargs.setdefault("context_length", config.context_length)
    if config.quantization is not None:
        engine_kwargs.setdefault("quantization", config.quantization)
    if config.mem_fraction_static is not None:
        engine_kwargs.setdefault("mem_fraction_static", config.mem_fraction_static)
    if config.attention_backend is not None:
        engine_kwargs.setdefault("attention_backend", config.attention_backend)
    if config.sampling_backend is not None:
        engine_kwargs.setdefault("sampling_backend", config.sampling_backend)
    if config.max_running_requests is not None:
        engine_kwargs.setdefault("max_running_requests", config.max_running_requests)
    if config.max_total_tokens is not None:
        engine_kwargs.setdefault("max_total_tokens", config.max_total_tokens)
    if config.seed is not None:
        engine_kwargs.setdefault("random_seed", config.seed)
    return _SGLangPythonClient(engine=engine_module.Engine(**engine_kwargs))


def _resolve_sglang_context_length(client: _SGLangClient, *, tokenizer: Any) -> int | None:
    engine = getattr(client, "engine", None)
    tokenizer_manager = getattr(engine, "tokenizer_manager", None)
    for candidate in (
        getattr(tokenizer_manager, "context_len", None),
        getattr(getattr(engine, "server_args", None), "context_length", None),
        getattr(getattr(engine, "server_args", None), "max_model_len", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None

def _normalize_sglang_response(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if not isinstance(payload, dict):
        raise TypeError("sglang response must be a dict or a list of dicts")

    text = payload.get("text")
    meta_info = payload.get("meta_info")
    if isinstance(text, list):
        meta_items = meta_info if isinstance(meta_info, list) else [meta_info] * len(text)
        return [
            {
                "text": item_text,
                "meta_info": item_meta or {},
            }
            for item_text, item_meta in zip(text, meta_items, strict=True)
        ]
    return [{"text": text, "meta_info": meta_info or {}}]


def _coerce_position_entries(raw_positions: Any) -> list[dict[str, Any]]:
    if raw_positions is None:
        return []
    output: list[dict[str, Any]] = []
    for entry in raw_positions:
        if entry is None:
            output.append({"score": None, "token_id": None, "text": None})
            continue
        output.append(_coerce_score_entry(entry))
    return output


def _coerce_nested_position_entries(raw_positions: Any) -> list[list[dict[str, Any]]]:
    if raw_positions is None:
        return []
    output: list[list[dict[str, Any]]] = []
    for position in raw_positions:
        if position is None:
            output.append([])
            continue
        output.append(
            [
                _coerce_score_entry(entry)
                for entry in position
                if entry is not None
            ]
        )
    return output


def _coerce_score_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        token_id = entry.get("token_id", entry.get("id"))
        score = entry.get("score", entry.get("logprob", entry.get("logit")))
        return {
            "score": float(score) if score is not None else None,
            "token_id": int(token_id) if token_id is not None else None,
            "text": entry.get("text"),
        }
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return {
            "score": float(entry[0]) if entry[0] is not None else None,
            "token_id": int(entry[1]) if entry[1] is not None else None,
            "text": entry[2] if len(entry) > 2 else None,
        }
    raise TypeError(f"unsupported sglang score entry: {entry!r}")


def _entries_to_score_map(entries: list[dict[str, Any]]) -> dict[int, float]:
    output: dict[int, float] = {}
    for entry in entries:
        token_id = entry.get("token_id")
        score = entry.get("score")
        if token_id is None or score is None:
            continue
        output[int(token_id)] = float(score)
    return output


def _deduplicate_preserve_order(token_ids: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in seen:
            continue
        seen.add(token_id)
        output.append(token_id)
    return output
