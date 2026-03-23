# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import gc
import inspect
import threading
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from statistics import mean
from typing import Any

import pcre
from packaging.version import Version

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
from evalution.engines.memory import build_memory_profile, gib_to_bytes, resolve_dtype
from evalution.logbar import get_logger

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
            chunks: list[_ScoringChunk] = []
            for request_index, (prefix_ids, target_ids, metadata) in enumerate(prepared_requests):
                chunks.extend(
                    self._build_loglikelihood_chunks(
                        request_index=request_index,
                        prefix_ids=prefix_ids,
                        target_ids=target_ids,
                        metadata=metadata,
                    )
                )

            chunk_scores = self._score_chunks(chunks, batch_size=effective_batch_size)
            totals = [
                LoglikelihoodOutput(
                    logprob=0.0,
                    is_greedy=True,
                    token_count=0,
                    metadata=dict(request.metadata),
                )
                for request in requests
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

    # Reuse continuation scoring to evaluate rolling text likelihood for perplexity-style suites.
    def loglikelihood_rolling(
        self,
        requests: list[RollingLoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]:
        if not requests:
            return []

        scoring_requests = [
            LoglikelihoodRequest(
                context="",
                continuation=request.text,
                continuation_input_ids=list(request.input_ids) if request.input_ids is not None else None,
                metadata=dict(request.metadata),
            )
            for request in requests
        ]
        scored = self.loglikelihood(scoring_requests, batch_size=batch_size)
        return [
            RollingLoglikelihoodOutput(
                logprob=output.logprob,
                token_count=output.token_count,
                metadata=output.metadata,
            )
            for output in scored
        ]

    # Emulate continuous refill on top of fixed batches so suite execution code can stay unchanged.
    def generate_continuous(
        self,
        requests: Any,
        *,
        batch_size: int | None = None,
    ) -> Any:
        def iterator() -> Any:
            request_items = list(requests)
            if not request_items:
                return

            effective_batch_size = batch_size or self.resolve_batch_size(
                [request for _, request in request_items]
            )
            with self._generation_lock:
                with self._state_lock:
                    standard_batch_size_cap = self.standard_batch_size_cap
                if standard_batch_size_cap is not None:
                    effective_batch_size = min(effective_batch_size, standard_batch_size_cap)
                self._log_generation_execution()
                yield from self._generate_standard_continuous(
                    request_items,
                    batch_size=effective_batch_size,
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
                "max_new_tokens=%d, dtype=%s, total_vram_gib=%.1f",
                resolved,
                stats["row_count"],
                stats["min_prompt_tokens"],
                stats["avg_prompt_tokens"],
                stats["max_prompt_tokens"],
                stats["max_new_tokens"],
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
        tokens_per_request = stats["avg_prompt_tokens"] + stats["max_new_tokens"]
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
    ) -> Any:
        batch: list[tuple[Any, GenerationRequest]] = []
        for item in requests:
            batch.append(item)
            if len(batch) == batch_size:
                outputs = self._generate_standard(
                    [request for _, request in batch],
                    batch_size=len(batch),
                )
                for (request_key, _request), output in zip(batch, outputs, strict=True):
                    yield request_key, output
                batch = []
        if batch:
            outputs = self._generate_standard(
                [request for _, request in batch],
                batch_size=len(batch),
            )
            for (request_key, _request), output in zip(batch, outputs, strict=True):
                yield request_key, output

    # Build the per-batch generation kwargs shared by both transformer engines.
    def _build_generation_kwargs(self, batch: list[GenerationRequest]) -> dict[str, Any]:
        max_new_tokens = max(
            request.max_new_tokens if request.max_new_tokens is not None else self.config.max_new_tokens
            for request in batch
        )
        do_sample = any(request.do_sample for request in batch)
        generation_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
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
                prefix_ids = self.tokenizer(
                    request.context,
                    add_special_tokens=False,
                )["input_ids"]
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
        history_ids = list(prefix_ids)
        if not history_ids:
            history_ids = [self._prefix_token_id()]

        chunks: list[_ScoringChunk] = []
        cursor = 0
        while cursor < len(target_ids):
            remaining = len(target_ids) - cursor
            target_count = min(remaining, max_input_length - 1)
            context_count = min(len(history_ids), max_input_length - target_count)
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

    # Run padded forwards over chunk batches and gather the exact token log-probs being evaluated.
    def _score_chunks(
        self,
        chunks: list[_ScoringChunk],
        *,
        batch_size: int,
    ) -> list[LoglikelihoodOutput]:
        import torch

        scored_chunks: list[LoglikelihoodOutput] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            encoded = None
            logits = None
            try:
                with self._tokenizer_lock:
                    encoded = self.tokenizer.pad(
                        {"input_ids": [list(chunk.input_ids) for chunk in batch]},
                        return_tensors="pt",
                        padding=True,
                    )
                encoded = {key: value.to(self.input_device) for key, value in encoded.items()}

                with self._scoring_attention_context():
                    with torch.inference_mode():
                        outputs = self.model(**encoded)
                logits = outputs.logits
                shift_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                shift_labels = encoded["input_ids"][:, 1:]
                padded_length = int(encoded["input_ids"].shape[1])
                padding_side = getattr(self.tokenizer, "padding_side", "right")

                for row_index, chunk in enumerate(batch):
                    sample_length = len(chunk.input_ids)
                    pad_offset = padded_length - sample_length if padding_side == "left" else 0
                    target_start = pad_offset + chunk.score_start
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
            finally:
                if logits is not None:
                    del logits
                if encoded is not None:
                    del encoded
        return scored_chunks

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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trust_remote_code = (
        config.trust_remote_code
        if config.trust_remote_code is not None
        else model_config.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_path or model_config.path,
        revision=model_config.revision,
        trust_remote_code=trust_remote_code,
        **model_config.tokenizer_kwargs,
    )
    tokenizer.padding_side = config.padding_side
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("tokenizer must define either a pad_token, eos_token, or unk_token")

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
        ),
        input_device=input_device,
        requested_attn_implementation=requested_attn_implementation,
    )


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
) -> Any | None:
    with suppress(Exception):
        from transformers import AutoTokenizer

        prepare_tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer_path or model_config.path,
            revision=model_config.revision,
            trust_remote_code=trust_remote_code,
            **model_config.tokenizer_kwargs,
        )
        prepare_tokenizer.padding_side = tokenizer.padding_side
        if prepare_tokenizer.pad_token_id is None:
            if tokenizer.pad_token is not None:
                prepare_tokenizer.pad_token = tokenizer.pad_token
            elif tokenizer.eos_token is not None:
                prepare_tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.unk_token is not None:
                prepare_tokenizer.pad_token = tokenizer.unk_token
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
