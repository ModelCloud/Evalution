# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import gc
import inspect
import random
import threading
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from itertools import chain, islice
from statistics import mean
from typing import Any

import pcre
from packaging.version import Version
from tokenicer import Tokenicer

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
from evalution.engines.memory import build_memory_profile, gib_to_bytes, resolve_dtype
from evalution.logbar import get_logger, loglikelihood_progress_title, manual_progress

_AUTO_BATCH_SIZE = "auto"
# PyPI source distributions first ship `generation/continuous_batching` in transformers 4.56.0.
_CONTINUOUS_BATCHING_MIN_TRANSFORMERS_VERSION = Version("4.56.0")
_UNEXPECTED_LOADER_KWARG_PATTERN = pcre.compile(r"unexpected keyword argument '([^']+)'")
_AUTO_BATCH_LADDER = (
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    40,
    48,
    64,
    80,
    96,
    128,
    160,
    192,
    256,
    320,
    384,
    512,
    640,
    768,
    896,
    1024,
    1280,
    1536,
    2048,
)
_TRANSFORMERS_NO_TIE_WEIGHTS_PATCH_LOCK = threading.Lock()
_TRANSFORMERS_NO_TIE_WEIGHTS_ACTIVE: ContextVar[bool] = ContextVar(
    "evalution_transformers_no_tie_weights_active",
    default=False,
)
# Let benchmarks that already own a public progress bar suppress the internal chunk-level scorer bar.
_LOGLIKELIHOOD_DISABLE_CHUNK_PROGRESS_METADATA_KEY = (
    "_evalution_disable_loglikelihood_chunk_progress"
)


@dataclass(slots=True)
class _ScoringChunk:
    # Track one model forward slice plus which token span contributes to the score.
    request_index: int
    input_ids: list[int]
    score_start: int
    score_count: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class _LoadedTransformerRuntime:
    # Bundle the shared runtime objects returned by the common model/tokenizer loader.
    model: Any
    tokenizer: Any
    prepare_tokenizer: Any | None
    input_device: Any
    requested_attn_implementation: str | None


@dataclass(slots=True)
class _TransformersCommonConfig(BaseEngine):
    # Hold the load and generation controls shared by both transformer engine variants.
    dtype: str | None = "auto"
    attn_implementation: str | None = None
    device: str | None = None
    device_map: str | dict[str, Any] | None = None
    seed: int | None = None
    batch_size: int | str = _AUTO_BATCH_SIZE
    max_new_tokens: int = 256
    trust_remote_code: bool | None = None
    padding_side: str = "left"
    resolved_engine: str | None = field(default=None, init=False)

    # Keep engine serialization stable across runtime APIs and test assertions.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BaseTransformerSession(BaseInferenceSession):
    # Share standard generation, scoring, and resource-management behavior across transformer engines.
    config: Any
    model_config: Model
    model: Any
    tokenizer: Any
    input_device: Any
    prepare_tokenizer: Any | None = None
    requested_attn_implementation: str | None = None
    effective_attn_implementation: str | None = None
    paged_attention_enabled: bool = False
    generation_backend: str = "generate"
    standard_batch_size_cap: int | None = None
    stop_criteria_cache: dict[tuple[str, ...], Any] = field(default_factory=dict, repr=False)
    auto_batch_size_cache: dict[tuple[Any, ...], int] = field(default_factory=dict, repr=False)
    execution_logged: bool = field(default=False, repr=False)
    _generation_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _tokenizer_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _prepare_tokenizer_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    # Run fixed-batch generation for engines that do not own a paged continuous batching manager.
    def generate(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int | None = None,
    ) -> list[GenerationOutput]:
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

    # Score short continuation spans with direct causal-LM forwards and token log-prob gathers.
    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]:
        if not requests:
            return []

        with self._generation_lock:
            prepared_requests = [self._prepare_loglikelihood_request(request) for request in requests]
            effective_batch_size = batch_size or self._resolve_scoring_batch_size(prepared_requests)
            return self._score_prepared_loglikelihood_requests(
                prepared_requests,
                batch_size=effective_batch_size,
            )

    # Keep log-likelihood request submission decoupled from caller iteration so suites can stream
    # rows lazily while this session keeps fixed-size scoring batches full on a worker thread.
    def loglikelihood_continuous(
        self,
        requests: Any,
        *,
        batch_size: int | None = None,
    ) -> Any:
        def iterator() -> Any:
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
                        metadata = dict(request.metadata)
                        metadata[_LOGLIKELIHOOD_DISABLE_CHUNK_PROGRESS_METADATA_KEY] = True
                        prepared_batch.append(
                            (
                                request_key,
                                self._prepare_loglikelihood_request(
                                    replace(request, metadata=metadata)
                                ),
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

    # Score longer token sequences first and break ties deterministically.
    def _loglikelihood_request_sort_key(
        self,
        item: tuple[int, tuple[list[int], list[int], dict[str, Any]]],
    ) -> tuple[int, tuple[int, ...]]:
        _request_index, (prefix_ids, target_ids, _metadata) = item
        combined = tuple(prefix_ids + target_ids)
        return (-len(combined), combined)

    # Reuse continuation scoring to evaluate rolling text likelihood for perplexity-style suites.
    def loglikelihood_rolling(
        self,
        requests: list[RollingLoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]:
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

    # Feed fixed-batch engines through shared request/result queues so the caller acts as the
    # RequestProducer, this session owns the RequestConsumer loop, and user-visible yields stay
    # decoupled from backend work while suites keep one continuous API.
    def generate_continuous(
        self,
        requests: Any,
        *,
        batch_size: int | None = None,
    ) -> Any:
        def iterator() -> Any:
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

            # This session is the RequestConsumer here: it drains RequestQueue, batches items, and
            # executes them through the standard fixed-batch backend before sending Result items
            # back to the caller thread.
            def consume_requests(
                stop_event: threading.Event,
                request_queue: Any,
                put_result: Any,
            ) -> None:
                with self._generation_lock:
                    with self._state_lock:
                        standard_batch_size_cap = self.standard_batch_size_cap
                    if standard_batch_size_cap is not None:
                        capped_batch_size = min(effective_batch_size, standard_batch_size_cap)
                    else:
                        capped_batch_size = effective_batch_size
                    self._log_generation_execution()
                    for request_key, output in self._generate_standard_continuous(
                        request_queue.iter_requests(stop_event=stop_event),
                        batch_size=capped_batch_size,
                        stop_event=stop_event,
                    ):
                        put_result(request_key, output)

            yield from stream_request_results(
                items,
                producer_name=f"{type(self).__name__}.request_producer",
                consumer_name=f"{type(self).__name__}.request_consumer",
                process_requests=consume_requests,
                require_non_main_thread=self.request_executor_requires_non_main_thread,
                request_queue_max_size=max(effective_batch_size * 2, 1),
            )

        return iterator()

    # Close standard session state and release model/tokenizer references.
    def close(self) -> None:
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

    # Drop per-suite caches and ask the CUDA allocator to release unused memory.
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

    # Render one request into the text prompt that the tokenizer/model should consume.
    def _render_request(self, request: GenerationRequest) -> str:
        if request.rendered_prompt is not None:
            return request.rendered_prompt
        return self._render_request_with_tokenizer(self.tokenizer, request)

    # Use the supplied tokenizer so preparation can tokenize without mutating the live tokenizer state.
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

    @property
    def batch_size(self) -> int | str:
        return self.config.batch_size

    # Report the runtime execution mode used by the active session.
    def describe_execution(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "requested_attn_implementation": self.requested_attn_implementation,
                "effective_attn_implementation": self.effective_attn_implementation,
                "paged_attention": self.paged_attention_enabled,
                "generation_backend": self.generation_backend,
                "standard_batch_size_cap": self.standard_batch_size_cap,
            }

    # Let suite execution pre-render and pretokenize requests before generation starts.
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

    # Resolve the concrete batch size for this suite using either an explicit value or the auto heuristic.
    def resolve_batch_size(self, requests: list[GenerationRequest]) -> int:
        configured_batch_size = _normalize_batch_size(self.config.batch_size)
        if configured_batch_size != _AUTO_BATCH_SIZE:
            return configured_batch_size
        if not requests:
            return 1

        with self._generation_lock:
            stats = self._batch_size_stats(requests)
            cache_key = (
                stats["row_count"],
                stats["min_prompt_tokens"],
                stats["avg_prompt_tokens"],
                stats["max_prompt_tokens"],
                stats["max_new_tokens"],
                stats["max_num_beams"],
                stats["dtype_name"],
                stats["dtype_bytes"],
                stats["total_vram_gib"],
                stats["parameter_count_billions"],
            )
            with self._state_lock:
                cached = self.auto_batch_size_cache.get(cache_key)
            if cached is not None:
                return cached

            resolved = self._estimate_auto_batch_size(stats)
            with self._state_lock:
                cached = self.auto_batch_size_cache.get(cache_key)
                if cached is not None:
                    return cached
                self.auto_batch_size_cache[cache_key] = resolved
            get_logger().info(
                "auto batch size resolved to %d for %d row(s); prompt_tokens(min/avg/max)=%d/%.1f/%d, "
                "max_new_tokens=%d, max_num_beams=%d, dtype=%s, total_vram_gib=%.1f",
                resolved,
                stats["row_count"],
                stats["min_prompt_tokens"],
                stats["avg_prompt_tokens"],
                stats["max_prompt_tokens"],
                stats["max_new_tokens"],
                stats["max_num_beams"],
                stats["dtype_name"],
                stats["total_vram_gib"],
            )
            return resolved

    # Cache per-stop-string stopping criteria when the installed transformers build exposes them.
    def _get_stop_criteria(self, stop_strings: list[str]) -> Any | None:
        cache_key = tuple(stop_strings)
        with self._state_lock:
            criteria = self.stop_criteria_cache.get(cache_key)
        if criteria is None:
            criteria = _build_stop_criteria(self.tokenizer, list(cache_key))
            if criteria is None:
                return None
            with self._state_lock:
                cached = self.stop_criteria_cache.get(cache_key)
                if cached is not None:
                    return cached
                self.stop_criteria_cache[cache_key] = criteria
        return criteria

    # Gather the prompt and memory statistics used by the auto batch-size estimator.
    def _batch_size_stats(self, requests: list[GenerationRequest]) -> dict[str, Any]:
        prompt_lengths = [
            len(request.input_ids)
            if request.input_ids is not None
            else None
            for request in requests
        ]
        if any(length is None for length in prompt_lengths):
            with self._tokenizer_lock:
                rendered_prompts = [
                    self._render_request(request)
                    for request in requests
                    if request.input_ids is None
                ]
                encoded = self.tokenizer(rendered_prompts, padding=False)
            fallback_lengths = iter(len(token_ids) for token_ids in encoded["input_ids"])
            prompt_lengths = [
                length if length is not None else next(fallback_lengths)
                for length in prompt_lengths
            ]
            del encoded
            del rendered_prompts

        max_new_tokens = max(
            request.max_new_tokens if request.max_new_tokens is not None else self.config.max_new_tokens
            for request in requests
        )
        max_num_beams = max(max(request.num_beams, 1) for request in requests)
        memory_profile = build_memory_profile(
            self.model,
            input_device=self.input_device,
            configured_dtype=self.config.dtype,
        )

        return {
            "row_count": len(requests),
            "min_prompt_tokens": min(prompt_lengths),
            "avg_prompt_tokens": float(mean(prompt_lengths)),
            "max_prompt_tokens": max(prompt_lengths),
            "max_new_tokens": max_new_tokens,
            "max_num_beams": max_num_beams,
            "dtype_name": memory_profile.dtype_name,
            "dtype_bytes": memory_profile.dtype_bytes,
            "total_vram_gib": memory_profile.total_vram_gib,
            "free_vram_gib": memory_profile.free_vram_gib,
            "parameter_count_billions": memory_profile.parameter_count_billions,
            "kv_cache_bytes_per_token": memory_profile.kv_cache_bytes_per_token,
        }

    # Estimate a friendly batch size from prompt lengths and the live memory profile.
    def _estimate_auto_batch_size(self, stats: dict[str, Any]) -> int:
        row_count = stats["row_count"]
        tokens_per_request = (
            stats["avg_prompt_tokens"] + stats["max_new_tokens"]
        ) * max(stats["max_num_beams"], 1)
        if tokens_per_request <= 0:
            return 1

        if self.input_device.type != "cuda":
            raw_batch_size = max(1, int(2048 / tokens_per_request))
            return min(row_count, _friendly_batch_size(raw_batch_size))

        dtype_scale = 2.0 / max(stats["dtype_bytes"], 1)
        total_token_budget = stats["total_vram_gib"] * 2048.0 * dtype_scale

        if stats["free_vram_gib"] > 0.0 and stats["total_vram_gib"] > 0.0:
            free_ratio = min(1.0, stats["free_vram_gib"] / stats["total_vram_gib"])
            total_token_budget *= max(0.35, free_ratio)

        spread_ratio = stats["max_prompt_tokens"] / max(stats["min_prompt_tokens"], 1)
        total_token_budget /= max(1.0, spread_ratio ** 0.25)
        max_from_tokens = max(1, int(total_token_budget / tokens_per_request))

        max_from_vram = row_count
        kv_cache_bytes_per_token = stats["kv_cache_bytes_per_token"]
        if kv_cache_bytes_per_token is not None and stats["total_vram_gib"] > 0.0:
            if stats["free_vram_gib"] > 0.0:
                available_budget_gib = min(
                    stats["total_vram_gib"] * 0.72,
                    stats["free_vram_gib"] * 0.90,
                )
            else:
                available_budget_gib = stats["total_vram_gib"] * 0.50
            request_bytes = (
                (stats["max_prompt_tokens"] + stats["max_new_tokens"])
                * kv_cache_bytes_per_token
                * 3.5
            )
            if request_bytes > 0:
                max_from_vram = max(
                    1,
                    int(gib_to_bytes(available_budget_gib) / request_bytes),
                )

        raw_batch_size = max(1, min(row_count, max_from_tokens, max_from_vram))
        return min(row_count, _friendly_batch_size(raw_batch_size))

    # Run standard `model.generate()` over one or more fixed batches.
    def _generate_standard(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int,
    ) -> list[GenerationOutput]:
        import torch

        outputs: list[GenerationOutput] = []
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            rendered_prompts = None
            encoded = None
            generated = None
            try:
                with self._tokenizer_lock:
                    rendered_prompts = [self._render_request(request) for request in batch]
                    encoded = self._encode_standard_batch(batch)
                encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
                generation_kwargs = self._build_generation_kwargs(batch)
                stopping_criteria = self._build_stopping_criteria(batch)
                if stopping_criteria is not None:
                    generation_kwargs["stopping_criteria"] = stopping_criteria

                with torch.inference_mode():
                    generated = self.model.generate(**encoded, **generation_kwargs)

                input_length = encoded["input_ids"].shape[1]
                with self._tokenizer_lock:
                    for index, token_ids in enumerate(generated):
                        generated_tokens = token_ids[input_length:]
                        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                        text = _truncate_at_stop(text, batch[index].stop).strip()
                        outputs.append(
                            GenerationOutput(
                                prompt=rendered_prompts[index],
                                text=text,
                                metadata=batch[index].metadata,
                            )
                        )
                        del generated_tokens
            finally:
                if generated is not None:
                    del generated
                if encoded is not None:
                    del encoded
                del batch
                if rendered_prompts is not None:
                    del rendered_prompts
        return outputs

    # Stream standard generation outputs in submission order while still honoring the continuous API shape.
    def _generate_standard_continuous(
        self,
        requests: Any,
        *,
        batch_size: int,
        stop_event: threading.Event | None = None,
    ) -> Any:
        batch: list[tuple[Any, GenerationRequest]] = []
        for item in requests:
            if stop_event is not None and stop_event.is_set():
                return
            batch.append(item)
            if len(batch) == batch_size:
                if stop_event is not None and stop_event.is_set():
                    return
                outputs = self._generate_standard(
                    [request for _, request in batch],
                    batch_size=len(batch),
                )
                for (request_key, _request), output in zip(batch, outputs, strict=True):
                    if stop_event is not None and stop_event.is_set():
                        return
                    yield request_key, output
                batch = []
        if batch:
            if stop_event is not None and stop_event.is_set():
                return
            outputs = self._generate_standard(
                [request for _, request in batch],
                batch_size=len(batch),
            )
            for (request_key, _request), output in zip(batch, outputs, strict=True):
                if stop_event is not None and stop_event.is_set():
                    return
                yield request_key, output

    # Build the per-batch generation kwargs shared by both transformer engines.
    def _build_generation_kwargs(self, batch: list[GenerationRequest]) -> dict[str, Any]:
        max_new_tokens = max(
            request.max_new_tokens if request.max_new_tokens is not None else self.config.max_new_tokens
            for request in batch
        )
        num_beams = max(max(request.num_beams, 1) for request in batch)
        do_sample = any(request.do_sample for request in batch)
        generation_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            generation_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        if do_sample:
            generation_kwargs.setdefault("temperature", batch[0].temperature)
        else:
            generation_kwargs["temperature"] = 1.0
            generation_kwargs["top_p"] = 1.0
            generation_kwargs.pop("top_k", None)
        return generation_kwargs

    # Prefer pretokenized ids when available so repeated prompt encoding stays off the hot path.
    def _encode_standard_batch(self, batch: list[GenerationRequest]) -> dict[str, Any]:
        if all(request.input_ids is not None for request in batch):
            return self.tokenizer.pad(
                {"input_ids": [list(request.input_ids) for request in batch]},
                return_tensors="pt",
                padding=True,
            )
        return self.tokenizer(
            [self._render_request(request) for request in batch],
            return_tensors="pt",
            padding=True,
        )

    # Mirror the active generation kwargs onto a GenerationConfig object when upstream APIs require it.
    def _build_generation_config(self, batch: list[GenerationRequest]) -> Any:
        from transformers import GenerationConfig

        generation_kwargs = self._build_generation_kwargs(batch)
        generation_config = GenerationConfig.from_model_config(self.model.config)
        for key, value in generation_kwargs.items():
            setattr(generation_config, key, value)
        common_stop_strings = _common_stop_strings(batch)
        generation_config.stop_strings = list(common_stop_strings) if common_stop_strings else None
        return generation_config

    # Build batch stopping criteria only when all requests share the same stop strings.
    def _build_stopping_criteria(self, batch: list[GenerationRequest]) -> Any | None:
        common_stop_strings = _common_stop_strings(batch)
        if not common_stop_strings:
            return None

        criteria = self._get_stop_criteria(common_stop_strings)
        if criteria is None:
            return None

        with suppress(Exception):
            from transformers import StoppingCriteriaList

            return StoppingCriteriaList([criteria])
        return None

    # Tokenize or reuse the ids needed for token-level continuation scoring.
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

    # Use tokenizer defaults for scored contexts while avoiding a doubled BOS prefix.
    def _tokenize_loglikelihood_context(self, text: str) -> list[int]:
        tokenizer_kwargs: dict[str, Any] = {}
        prefix_text = self._decoded_prefix_token_text()
        if prefix_text and text.startswith(prefix_text):
            tokenizer_kwargs["add_special_tokens"] = False
        return list(self.tokenizer(text, **tokenizer_kwargs)["input_ids"])

    # Reuse the auto batch heuristic by mapping scored tokens onto zero-generation requests.
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

    # Split long scored continuations into model-sized windows while preserving trailing context.
    def _build_loglikelihood_chunks(
        self,
        *,
        request_index: int,
        prefix_ids: list[int],
        target_ids: list[int],
        metadata: dict[str, Any],
    ) -> list[_ScoringChunk]:
        max_input_length = self._max_scoring_input_length()
        # Decoder-only scoring can carry one extra request token because the model input drops the
        # final token while still predicting the full continuation span in that window.
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

    # Expand one rolling-perplexity request into disjoint token windows.
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

    # Trim overlapping context tokens because the scorer already conditions continuation tokens on their prefix.
    def _make_disjoint_window(
        self,
        pair: tuple[list[int], list[int]],
    ) -> tuple[list[int], list[int]]:
        context_ids, continuation_ids = pair
        return context_ids[: len(context_ids) - (len(continuation_ids) - 1)], continuation_ids

    # Run padded forwards over chunk batches and gather the exact token log-probs being evaluated.
    def _score_chunks(
        self,
        chunks: list[_ScoringChunk],
        *,
        batch_size: int,
    ) -> list[LoglikelihoodOutput]:
        import torch

        scored_chunks: list[LoglikelihoodOutput] = []
        if not chunks:
            return scored_chunks

        progress_disabled = bool(
            chunks[0].metadata.get(_LOGLIKELIHOOD_DISABLE_CHUNK_PROGRESS_METADATA_KEY)
        )
        score_bar = None
        if not progress_disabled:
            progress_title = (
                loglikelihood_progress_title(chunks[0].metadata) or "loglikelihood: scoring continuations"
            )
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            score_bar = manual_progress(
                len(chunks),
                title=progress_title,
                subtitle=f"batch_size={batch_size}",
            )
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            if score_bar is not None:
                batch_index = (start // batch_size) + 1
                score_bar.subtitle(f"batch={batch_index}/{total_batches} batch_size={batch_size}")
            encoded = None
            logits = None
            try:
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = self._prefix_token_id()
                padded_length = max(len(chunk.input_ids) for chunk in batch)
                padded_rows = [
                    list(chunk.input_ids) + ([int(pad_token_id)] * (padded_length - len(chunk.input_ids)))
                    for chunk in batch
                ]
                encoded = {
                    "input_ids": torch.tensor(
                        padded_rows,
                        dtype=torch.long,
                        device=self.input_device,
                    )
                }

                with self._scoring_attention_context():
                    with torch.inference_mode():
                        outputs = self.model(input_ids=encoded["input_ids"])
                logits = outputs.logits
                shift_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                shift_labels = encoded["input_ids"][:, 1:]

                for row_index, chunk in enumerate(batch):
                    target_start = chunk.score_start
                    shift_start = target_start - 1
                    shift_end = shift_start + chunk.score_count
                    sample_log_probs = shift_log_probs[row_index, shift_start:shift_end, :]
                    sample_targets = shift_labels[row_index, shift_start:shift_end]
                    gathered = sample_log_probs.gather(-1, sample_targets.unsqueeze(-1)).squeeze(-1)
                    greedy_tokens = sample_log_probs.argmax(dim=-1)
                    scored_chunks.append(
                        LoglikelihoodOutput(
                            logprob=float(gathered.sum().item()),
                            is_greedy=bool(torch.equal(greedy_tokens, sample_targets)),
                            token_count=chunk.score_count,
                            metadata=dict(chunk.metadata),
                        )
                    )
                    if score_bar is not None:
                        score_bar.next().draw()
            finally:
                if logits is not None:
                    del logits
                if encoded is not None:
                    del encoded
        return scored_chunks

    # Reuse the shared chunk scorer for both eager and continuous log-likelihood submission paths.
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
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key != _LOGLIKELIHOOD_DISABLE_CHUNK_PROGRESS_METADATA_KEY
                },
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

    # Flush one submitted scoring batch and publish request-level outputs back to the caller.
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

    # Pick the token used as synthetic left context when a scored request has no explicit prefix.
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

    # Decode the synthetic prefix token once so callers can detect explicit BOS-prefixed text.
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

    # Resolve the finite scoring window the model/tokenizer expose, with a conservative fallback.
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

    # Standard and compat engines do not need to alter attention kernels during token scoring.
    @contextmanager
    def _scoring_attention_context(self) -> Any:
        yield

    # Log the generation backend once per suite so repeated batches do not spam the output.
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

    # Choose the lock that matches the tokenizer instance currently doing the work.
    def _lock_for_tokenizer(self, tokenizer: Any) -> threading.RLock:
        if tokenizer is not None and tokenizer is self.prepare_tokenizer and tokenizer is not self.tokenizer:
            return self._prepare_tokenizer_lock
        return self._tokenizer_lock

    # Render and tokenize one generation request for APIs that consume explicit prompt ids.
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


# Load model/tokenizer state once so modern and compat sessions share the same initialization path.
def load_transformer_runtime(
    config: _TransformersCommonConfig,
    model_config: Model,
) -> _LoadedTransformerRuntime:
    import torch
    from transformers import AutoModelForCausalLM
    _patch_transformers_no_tie_weights_context_once()

    _seed_transformer_runtime(config.seed)

    trust_remote_code = (
        config.trust_remote_code
        if config.trust_remote_code is not None
        else model_config.trust_remote_code
    )
    tokenizer_source = _resolve_tokenizer_source(model_config)
    tokenizer = _load_tokenizer_from_model(
        tokenizer_source,
        revision=model_config.revision,
        trust_remote_code=trust_remote_code,
        **model_config.tokenizer_kwargs,
    )
    tokenizer.padding_side = config.padding_side

    load_kwargs = {
        **model_config.model_kwargs,
        "revision": model_config.revision,
        "trust_remote_code": trust_remote_code,
    }
    resolved_dtype = resolve_dtype(config.dtype)
    if resolved_dtype is not None and (
        config.dtype != "auto" or ("dtype" not in load_kwargs and "torch_dtype" not in load_kwargs)
    ):
        load_kwargs["dtype"] = resolved_dtype
    if config.dtype != "auto" and "dtype" in load_kwargs:
        load_kwargs.pop("torch_dtype", None)
    raw_attn_implementation = config.attn_implementation
    attn_implementation = _base_attn_implementation(raw_attn_implementation)
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    if config.device_map is not None:
        load_kwargs["device_map"] = config.device_map

    model = _load_model_with_compat_fallback(
        AutoModelForCausalLM,
        model_config.path,
        load_kwargs,
    )
    _seed_with_internal_apis(model, config.seed)
    freeze = getattr(model, "requires_grad_", None)
    if callable(freeze):
        freeze(False)
    model.eval()

    if config.device_map is None:
        device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            model.to(device)
        except NotImplementedError as exc:
            if "Cannot copy out of meta tensor; no data!" not in str(exc):
                raise

            get_logger().warning(
                "transformers returned a meta-initialized model for %s; reloading with device_map=%s",
                model_config.path,
                device,
            )
            model = _load_model_with_compat_fallback(
                AutoModelForCausalLM,
                model_config.path,
                {
                    **load_kwargs,
                    "device_map": device,
                },
            )
            freeze = getattr(model, "requires_grad_", None)
            if callable(freeze):
                freeze(False)
            model.eval()
        input_device = torch.device(device)
    else:
        input_device = _resolve_input_device(model, prefer=config.device)

    _normalize_tokenizer_special_tokens(tokenizer=tokenizer, model=model)

    requested_attn_implementation = (
        raw_attn_implementation
        or getattr(model.config, "_attn_implementation", None)
        or getattr(model.config, "attn_implementation", None)
    )

    return _LoadedTransformerRuntime(
        model=model,
        tokenizer=tokenizer,
        prepare_tokenizer=_clone_prepare_tokenizer(
            tokenizer=tokenizer,
            model_config=model_config,
            trust_remote_code=trust_remote_code,
            model=model,
        ),
        input_device=input_device,
        requested_attn_implementation=requested_attn_implementation,
    )


def _patch_transformers_no_tie_weights_context_once() -> None:
    """Make transformers.no_tie_weights execution-scoped instead of process-global.

    Upstream transformers currently swaps `PreTrainedModel.tie_weights` at the class level inside
    `initialization.no_tie_weights()`. Concurrent model loads can therefore suppress `tie_weights()`
    on the wrong model/thread. Evalution loads models concurrently in compare tests, so mirror the
    upstream context-scoped stop-gap locally until the equivalent transformers fix is available.
    """
    try:
        import transformers.initialization as initialization
        import transformers.modeling_utils as modeling_utils
    except Exception:
        return

    PreTrainedModel = getattr(modeling_utils, "PreTrainedModel", None)
    current_no_tie_weights = getattr(initialization, "no_tie_weights", None)
    current_tie_weights = getattr(PreTrainedModel, "tie_weights", None)
    if PreTrainedModel is None or not callable(current_no_tie_weights) or not callable(current_tie_weights):
        return

    with _TRANSFORMERS_NO_TIE_WEIGHTS_PATCH_LOCK:
        current_no_tie_weights = getattr(initialization, "no_tie_weights", None)
        current_tie_weights = getattr(PreTrainedModel, "tie_weights", None)
        if not callable(current_no_tie_weights) or not callable(current_tie_weights):
            return
        if getattr(current_no_tie_weights, "__evalution_context_patch__", False):
            return

        original_tie_weights = current_tie_weights

        def _context_scoped_tie_weights(self: Any, *args: Any, **kwargs: Any) -> Any:
            if _TRANSFORMERS_NO_TIE_WEIGHTS_ACTIVE.get():
                return None
            return original_tie_weights(self, *args, **kwargs)

        _context_scoped_tie_weights.__evalution_context_patch__ = True
        _context_scoped_tie_weights.__wrapped__ = original_tie_weights
        PreTrainedModel.tie_weights = _context_scoped_tie_weights

        @contextmanager
        def _context_scoped_no_tie_weights():
            token = _TRANSFORMERS_NO_TIE_WEIGHTS_ACTIVE.set(True)
            try:
                yield
            finally:
                _TRANSFORMERS_NO_TIE_WEIGHTS_ACTIVE.reset(token)

        _context_scoped_no_tie_weights.__evalution_context_patch__ = True
        _context_scoped_no_tie_weights.__wrapped__ = current_no_tie_weights
        initialization.no_tie_weights = _context_scoped_no_tie_weights


def _resolve_tokenizer_source(model_config: Model) -> Any:
    if model_config.tokenizer is not None:
        return model_config.tokenizer
    if model_config.tokenizer_path is not None:
        return model_config.tokenizer_path
    return model_config.path


def _load_tokenizer_from_model(
    pretrained_model_name_or_path_or_tokenizer: Any,
    **kwargs: Any,
) -> Any:
    return Tokenicer.load(pretrained_model_name_or_path_or_tokenizer, strict=False, **kwargs)


def _normalize_tokenizer_special_tokens(tokenizer: Any, *, model: Any | None = None) -> None:
    if tokenizer is None:
        return

    auto_fix_pad_token = getattr(tokenizer, "auto_fix_pad_token", None)
    if not callable(auto_fix_pad_token):
        return

    if model is None:
        try:
            auto_fix_pad_token(strict=False)
        except Exception:
            return
        return

    with suppress(Exception):
        auto_fix_pad_token(model, strict=False)


def _seed_transformer_runtime(seed: int | None) -> None:
    if seed is None:
        return

    with suppress(Exception):
        from transformers import set_seed as transformers_set_seed

        transformers_set_seed(seed)

    random.seed(seed)

    with suppress(Exception):
        import numpy as np

        np.random.seed(seed)

    with suppress(Exception):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def _seed_with_internal_apis(obj: Any, seed: int | None) -> None:
    if seed is None:
        return
    for method_name in ("set_seed", "manual_seed", "seed", "reset_rng"):
        method = getattr(obj, method_name, None)
        if callable(method):
            with suppress(Exception):
                method(seed)
                return


# Guard modern paged-attention engines with the package version that first shipped continuous batching.
def transformers_continuous_batching_support() -> tuple[bool, str]:
    import transformers

    installed_version = Version(transformers.__version__)
    if installed_version < _CONTINUOUS_BATCHING_MIN_TRANSFORMERS_VERSION:
        return (
            False,
            (
                f"transformers {installed_version} is older than "
                f"{_CONTINUOUS_BATCHING_MIN_TRANSFORMERS_VERSION}, which is the first release "
                "that ships generation/continuous_batching"
            ),
        )

    with suppress(Exception):
        from transformers import ContinuousBatchingManager  # noqa: F401

        return True, f"transformers {installed_version} provides ContinuousBatchingManager"

    return False, f"transformers {installed_version} does not expose ContinuousBatchingManager"


# Retry without newer load kwargs when older transformers releases reject them at model load time.
def _load_model_with_compat_fallback(
    loader: Any,
    model_path: str,
    load_kwargs: dict[str, Any],
) -> Any:
    attempt_kwargs = dict(load_kwargs)
    while True:
        try:
            return loader.from_pretrained(model_path, **attempt_kwargs)
        except TypeError as exc:
            fallback_kwargs, retry_action = _loader_kwargs_compat_fallback(attempt_kwargs, exc)
            if fallback_kwargs is None or retry_action is None:
                raise
            get_logger().warning(
                "transformers model loader rejected %s; retrying %s",
                sorted(_unexpected_loader_kwargs(exc)),
                retry_action,
            )
            attempt_kwargs = fallback_kwargs


# Detect specific loader kwargs that older transformers builds report as unexpected.
def _unexpected_loader_kwargs(exc: TypeError) -> set[str]:
    return set(_UNEXPECTED_LOADER_KWARG_PATTERN.findall(str(exc)))


def _loader_kwargs_compat_fallback(
    load_kwargs: dict[str, Any],
    exc: TypeError,
) -> tuple[dict[str, Any] | None, str | None]:
    unexpected_kwargs = _unexpected_loader_kwargs(exc)
    if not unexpected_kwargs:
        return None, None

    fallback_kwargs = dict(load_kwargs)
    actions: list[str] = []

    if "dtype" in unexpected_kwargs and "dtype" in fallback_kwargs:
        dtype_value = fallback_kwargs.pop("dtype")
        fallback_kwargs.setdefault("torch_dtype", dtype_value)
        actions.append("with torch_dtype compatibility alias")

    for key in ("attn_implementation", "torch_dtype"):
        if key in unexpected_kwargs and key in fallback_kwargs:
            fallback_kwargs.pop(key)
            actions.append(f"without {key}")

    if fallback_kwargs == load_kwargs or not actions:
        return None, None

    return fallback_kwargs, ", then ".join(actions)


def _resolve_input_device(model: Any, *, prefer: str | None = None) -> Any:
    import torch

    if prefer is not None:
        return torch.device(prefer)

    hf_device_map = getattr(model, "hf_device_map", {})
    for device in hf_device_map.values():
        if device in {"cpu", "disk"}:
            continue
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        return torch.device(str(device))

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _clone_prepare_tokenizer(
    *,
    tokenizer: Any,
    model_config: Model,
    trust_remote_code: bool | None,
    model: Any | None = None,
) -> Any | None:
    with suppress(Exception):
        resolved_trust_remote_code = (
            trust_remote_code if trust_remote_code is not None else False
        )
        tokenizer_source = _resolve_tokenizer_source(model_config)
        prepare_tokenizer = _load_tokenizer_from_model(
            tokenizer_source,
            revision=model_config.revision,
            trust_remote_code=resolved_trust_remote_code,
            **model_config.tokenizer_kwargs,
        )
        prepare_tokenizer.padding_side = tokenizer.padding_side
        _normalize_tokenizer_special_tokens(tokenizer=prepare_tokenizer, model=model)
        return prepare_tokenizer
    return None


# Build stop-string criteria only when the installed transformers release exposes that API.
def _build_stop_criteria(tokenizer: Any, stop_strings: list[str]) -> Any | None:
    with suppress(Exception):
        from transformers import StopStringCriteria

        return StopStringCriteria(tokenizer, stop_strings)
    return None


def _common_stop_strings(batch: list[GenerationRequest]) -> list[str] | None:
    if not batch:
        return None

    first = batch[0].stop
    if all(request.stop == first for request in batch):
        return first
    return None


def _truncate_at_stop(text: str, stop_strings: list[str]) -> str:
    if not stop_strings:
        return text

    cut_points = [text.find(stop) for stop in stop_strings if stop and stop in text]
    if not cut_points:
        return text
    return text[: min(cut_points)]


def _base_attn_implementation(attn_implementation: str | None) -> str | None:
    if attn_implementation is None:
        return None
    if attn_implementation.startswith("paged|"):
        return attn_implementation.split("paged|", maxsplit=1)[1]
    return attn_implementation


def _requests_paged_attention(attn_implementation: str | None) -> bool:
    return isinstance(attn_implementation, str) and attn_implementation.startswith("paged|")


def _fallback_batch_size(batch_size: int) -> int:
    if batch_size <= 1:
        return 1
    return _friendly_batch_size(max(1, batch_size // 2))


def _normalize_batch_size(batch_size: int | str) -> int | str:
    if batch_size == _AUTO_BATCH_SIZE:
        return _AUTO_BATCH_SIZE
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or 'auto'")
    return batch_size


def _friendly_batch_size(raw_batch_size: int) -> int:
    friendly = 1
    for candidate in _AUTO_BATCH_LADDER:
        if candidate > raw_batch_size:
            break
        friendly = candidate
    return max(1, friendly)
