from __future__ import annotations

import gc
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from itertools import chain, islice
from statistics import mean
from typing import Any

from evalution.config import Model
from evalution.engines.base import (
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
_AUTO_PAGED_ATTENTION = "auto"
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


def _truncate_at_stop(text: str, stop_strings: list[str]) -> str:
    if not stop_strings:
        return text

    cut_points = [text.find(stop) for stop in stop_strings if stop and stop in text]
    if not cut_points:
        return text
    return text[: min(cut_points)]


@dataclass(slots=True)
class Transformer:
    dtype: str | None = "auto"
    attn_implementation: str | None = None
    attention_impl: str | None = None
    device: str | None = None
    device_map: str | dict[str, Any] | None = None
    batch_size: int | str = _AUTO_BATCH_SIZE
    paged_attention: bool | str = _AUTO_PAGED_ATTENTION
    manual_eviction: bool = False
    allow_block_sharing: bool = True
    use_async_batching: bool | None = None
    q_padding_interval_size: int = 0
    kv_padding_interval_size: int = 0
    max_cached_graphs: int = 0
    max_new_tokens: int = 256
    trust_remote_code: bool | None = None
    padding_side: str = "left"
    load_kwargs: dict[str, Any] = field(default_factory=dict)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, model: Model) -> TransformerSession:
        return TransformerSession.from_config(self, model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TransformerSession:
    config: Transformer
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
    # Session-owned continuous batching state is reused while the request signature stays compatible.
    continuous_batching_manager: Any | None = field(default=None, repr=False)
    continuous_batching_signature: tuple[Any, ...] | None = field(default=None, repr=False)
    # Tracks completed request ids whose cache blocks are still retained when manual eviction is enabled.
    continuous_batching_completed_request_ids: set[str] = field(default_factory=set, repr=False)
    continuous_batching_request_counter: int = field(default=0, repr=False)
    _generation_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _tokenizer_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _prepare_tokenizer_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    @classmethod
    def from_config(cls, config: Transformer, model_config: Model) -> TransformerSession:
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
            **config.load_kwargs,
            "revision": model_config.revision,
            "trust_remote_code": trust_remote_code,
        }
        resolved_dtype = resolve_dtype(config.dtype)
        if resolved_dtype is not None:
            load_kwargs["dtype"] = resolved_dtype
        raw_attn_implementation = config.attention_impl or config.attn_implementation
        attn_implementation = _base_attn_implementation(raw_attn_implementation)
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        if config.device_map is not None:
            load_kwargs["device_map"] = config.device_map

        model = AutoModelForCausalLM.from_pretrained(model_config.path, **load_kwargs)
        freeze = getattr(model, "requires_grad_", None)
        if callable(freeze):
            freeze(False)
        model.eval()

        if config.device_map is None:
            device = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            input_device = torch.device(device)
        else:
            input_device = _resolve_input_device(model, prefer=config.device)

        requested_attn_implementation = (
            attn_implementation
            or getattr(model.config, "_attn_implementation", None)
            or getattr(model.config, "attn_implementation", None)
        )
        paged_attention_config = config.paged_attention
        if raw_attn_implementation is not None and raw_attn_implementation.startswith("paged|"):
            paged_attention_config = True
        paged_attention_enabled = _resolve_paged_attention(
            paged_attention=paged_attention_config,
            attn_implementation=requested_attn_implementation,
            model=model,
            input_device=input_device,
        )
        effective_attn_implementation = _effective_attn_implementation(
            requested_attn_implementation,
            paged_attention_enabled=paged_attention_enabled,
        )
        generation_backend = "continuous_batching" if paged_attention_enabled else "generate"
        get_logger().info(
            "transformer attention requested=%s effective=%s backend=%s paged_attention=%s",
            requested_attn_implementation,
            effective_attn_implementation,
            generation_backend,
            paged_attention_enabled,
        )

        return cls(
            config=config,
            model_config=model_config,
            model=model,
            tokenizer=tokenizer,
            prepare_tokenizer=_clone_prepare_tokenizer(
                tokenizer=tokenizer,
                model_config=model_config,
                trust_remote_code=trust_remote_code,
            ),
            input_device=input_device,
            requested_attn_implementation=requested_attn_implementation,
            effective_attn_implementation=effective_attn_implementation,
            paged_attention_enabled=paged_attention_enabled,
            generation_backend=generation_backend,
        )

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
                paged_attention_enabled = self.paged_attention_enabled
                standard_batch_size_cap = self.standard_batch_size_cap
            if not paged_attention_enabled and standard_batch_size_cap is not None:
                effective_batch_size = min(effective_batch_size, standard_batch_size_cap)
            self._log_generation_execution()
            if paged_attention_enabled:
                try:
                    return self._generate_paged(requests, batch_size=effective_batch_size)
                except Exception as exc:
                    fallback_batch_size = _fallback_batch_size(effective_batch_size)
                    self._disable_paged_attention(exc, fallback_batch_size=fallback_batch_size)
                    return self._generate_standard(requests, batch_size=fallback_batch_size)
            return self._generate_standard(requests, batch_size=effective_batch_size)

    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]:
        # Score short continuation spans with direct causal-LM forwards and token log-prob gathers.
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

    def loglikelihood_rolling(
        self,
        requests: list[RollingLoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]:
        # Reuse the continuation scorer by treating the whole text as the continuation from BOS/prefix.
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

    def generate_continuous(
        self,
        requests: Iterable[tuple[Any, GenerationRequest]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        def iterator() -> Iterator[tuple[Any, GenerationOutput]]:
            request_iter = iter(requests)
            preview_items = list(islice(request_iter, 64))
            if not preview_items:
                return

            effective_batch_size = batch_size or self.resolve_batch_size(
                [request for _, request in preview_items]
            )
            items = chain(preview_items, request_iter)

            with self._generation_lock:
                with self._state_lock:
                    paged_attention_enabled = self.paged_attention_enabled
                    standard_batch_size_cap = self.standard_batch_size_cap
                if not paged_attention_enabled and standard_batch_size_cap is not None:
                    effective_batch_size = min(effective_batch_size, standard_batch_size_cap)
                self._log_generation_execution()
                if paged_attention_enabled:
                    try:
                        yield from self._generate_paged_continuous(items, batch_size=effective_batch_size)
                        return
                    except Exception as exc:
                        fallback_batch_size = _fallback_batch_size(effective_batch_size)
                        self._disable_paged_attention(exc, fallback_batch_size=fallback_batch_size)
                        effective_batch_size = fallback_batch_size
                yield from self._generate_standard_continuous(items, batch_size=effective_batch_size)

        return iterator()

    def close(self) -> None:
        with self._generation_lock:
            self._stop_continuous_batching_manager()
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

    # Stop session-owned generation state, drop per-suite caches, and return unused allocator memory.
    def gc(self) -> None:
        with self._generation_lock:
            self._stop_continuous_batching_manager()
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
                "standard_batch_size_cap": self.standard_batch_size_cap,
            }

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

    def _get_stop_criteria(self, stop_strings: list[str]) -> Any:
        from transformers import StopStringCriteria

        cache_key = tuple(stop_strings)
        with self._state_lock:
            criteria = self.stop_criteria_cache.get(cache_key)
        if criteria is None:
            with self._tokenizer_lock:
                criteria = StopStringCriteria(self.tokenizer, list(cache_key))
            with self._state_lock:
                cached = self.stop_criteria_cache.get(cache_key)
                if cached is not None:
                    return cached
                self.stop_criteria_cache[cache_key] = criteria
        return criteria

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

    def _generate_standard(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int,
    ) -> list[GenerationOutput]:
        import torch
        from transformers import StoppingCriteriaList

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
                common_stop_strings = _common_stop_strings(batch)
                if common_stop_strings:
                    generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                        [self._get_stop_criteria(common_stop_strings)]
                    )

                with torch.inference_mode():
                    generated = self.model.generate(**encoded, **generation_kwargs)

                # `generate()` returns the full padded prompt plus new tokens. With left padding,
                # slicing by the unpadded token count leaks prompt-tail tokens into the decode.
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

    def _generate_paged(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int,
    ) -> list[GenerationOutput]:
        outputs_by_position: list[GenerationOutput | None] = [None] * len(requests)
        for position, output in self._generate_paged_continuous(
            enumerate(requests),
            batch_size=batch_size,
        ):
            outputs_by_position[int(position)] = output
        if any(output is None for output in outputs_by_position):
            raise RuntimeError("continuous batching returned incomplete results")
        outputs = [output for output in outputs_by_position if output is not None]
        return outputs

    def _generate_standard_continuous(
        self,
        requests: Iterable[tuple[Any, GenerationRequest]],
        *,
        batch_size: int,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
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

    def _generate_paged_continuous(
        self,
        requests: Iterable[tuple[Any, GenerationRequest]],
        *,
        batch_size: int,
    ) -> Iterator[tuple[Any, GenerationOutput]]:
        request_iter = iter(requests)
        inflight_requests: dict[str, tuple[Any, str, list[str], dict[str, Any]]] = {}
        source_exhausted = False
        expected_signature: tuple[Any, ...] | None = None
        manager: Any | None = None

        def submit_one() -> bool:
            nonlocal source_exhausted
            nonlocal expected_signature
            if source_exhausted:
                return False
            try:
                request_key, request = next(request_iter)
            except StopIteration:
                source_exhausted = True
                return False

            request_signature = _continuous_request_signature(request)
            if expected_signature is None:
                expected_signature = request_signature
            elif request_signature != expected_signature:
                raise ValueError(
                    "continuous batching requires shared stop strings and sampling settings "
                    "within a generation stream"
                )

            if manager is None:
                raise RuntimeError("continuous batching manager was not initialized")

            rendered_prompt, input_ids = self._prepare_request_for_generation(request)
            request_id = self._next_continuous_batching_request_id()
            manager.add_request(
                input_ids,
                request_id=request_id,
                max_new_tokens=request.max_new_tokens,
                streaming=False,
            )
            inflight_requests[request_id] = (
                request_key,
                rendered_prompt,
                list(request.stop),
                dict(request.metadata),
            )
            return True

        preview_items = list(islice(request_iter, max(1, batch_size)))
        if not preview_items:
            return
        request_iter = iter(chain(preview_items, request_iter))

        first_request = preview_items[0][1]
        expected_signature = _continuous_request_signature(first_request)
        manager = self._ensure_continuous_batching_manager(
            request_signature=expected_signature,
            request=first_request,
        )

        while len(inflight_requests) < batch_size and submit_one():
            continue

        while inflight_requests:
            request_output = manager.get_result(timeout=0.1)
            if request_output is None:
                if not manager.is_running():
                    raise RuntimeError(
                        "continuous batching manager stopped before all requests completed"
                    )
                continue

            request_id = request_output.request_id
            request_state = inflight_requests.get(request_id)
            if request_state is None:
                raise RuntimeError(
                    f"continuous batching returned unknown request_id={request_id!r}"
                )
            if request_output.error is not None:
                raise RuntimeError(
                    f"continuous batching request {request_id!r} failed: {request_output.error}"
                )
            if not request_output.is_finished():
                continue

            request_key, rendered_prompt, stop_strings, metadata = inflight_requests.pop(request_id)
            if self.config.manual_eviction:
                with self._state_lock:
                    self.continuous_batching_completed_request_ids.add(request_id)
            with self._tokenizer_lock:
                text = self.tokenizer.decode(
                    request_output.generated_tokens,
                    skip_special_tokens=False,
                )
            text = _truncate_at_stop(text, stop_strings).strip()
            yield request_key, GenerationOutput(
                prompt=rendered_prompt,
                text=text,
                metadata=metadata,
            )

            while len(inflight_requests) < batch_size and submit_one():
                continue

    def _ensure_continuous_batching_manager(
        self,
        *,
        request_signature: tuple[Any, ...],
        request: GenerationRequest,
    ) -> Any:
        from transformers import ContinuousBatchingManager

        generation_config = self._build_generation_config([request])

        with self._state_lock:
            manager = self.continuous_batching_manager
            manager_signature = self.continuous_batching_signature
            if (
                manager is not None
                and manager_signature == request_signature
                and manager.is_running()
            ):
                return manager

        self._stop_continuous_batching_manager()

        manager = ContinuousBatchingManager(
            self.model,
            generation_config=generation_config,
            manual_eviction=self.config.manual_eviction,
            q_padding_interval_size=self.config.q_padding_interval_size,
            kv_padding_interval_size=self.config.kv_padding_interval_size,
            max_cached_graphs=self.config.max_cached_graphs,
            allow_block_sharing=self.config.allow_block_sharing,
            use_async_batching=self.config.use_async_batching,
        )
        manager.start()

        with self._state_lock:
            self.continuous_batching_manager = manager
            self.continuous_batching_signature = request_signature
            return manager

    def _next_continuous_batching_request_id(self) -> str:
        with self._state_lock:
            request_id = f"req_{self.continuous_batching_request_counter}"
            self.continuous_batching_request_counter += 1
            return request_id

    def _stop_continuous_batching_manager(self) -> None:
        with self._state_lock:
            manager = self.continuous_batching_manager
            retained_request_ids = set(self.continuous_batching_completed_request_ids)
            self.continuous_batching_manager = None
            self.continuous_batching_signature = None
            self.continuous_batching_completed_request_ids.clear()

        if manager is None:
            return

        if self.config.manual_eviction:
            for request_id in sorted(retained_request_ids):
                with suppress(Exception):
                    manager.evict_request_from_cache(request_id)

        with suppress(Exception):
            manager.stop(block=True)

    def _build_generation_kwargs(self, batch: list[GenerationRequest]) -> dict[str, Any]:
        max_new_tokens = max(
            request.max_new_tokens if request.max_new_tokens is not None else self.config.max_new_tokens
            for request in batch
        )
        do_sample = any(request.do_sample for request in batch)
        generation_kwargs = {
            **self.config.generation_kwargs,
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

    def _build_generation_config(self, batch: list[GenerationRequest]) -> Any:
        from transformers import GenerationConfig

        generation_kwargs = self._build_generation_kwargs(batch)
        generation_config = GenerationConfig.from_model_config(self.model.config)
        for key, value in generation_kwargs.items():
            setattr(generation_config, key, value)
        common_stop_strings = _common_stop_strings(batch)
        generation_config.stop_strings = list(common_stop_strings) if common_stop_strings else None
        return generation_config

    def _prepare_loglikelihood_request(
        self,
        request: LoglikelihoodRequest,
    ) -> tuple[list[int], list[int], dict[str, Any]]:
        # Tokenize or reuse the context/continuation ids needed for token-level scoring.
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

    def _resolve_scoring_batch_size(
        self,
        requests: list[tuple[list[int], list[int], dict[str, Any]]],
    ) -> int:
        # Reuse the auto batch-size heuristic by mapping scored tokens onto zero-generation requests.
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
        # Split long scored continuations into model-sized windows while preserving trailing context.
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

    def _score_chunks(
        self,
        chunks: list[_ScoringChunk],
        *,
        batch_size: int,
    ) -> list[LoglikelihoodOutput]:
        # Run padded forwards over chunk batches and gather the exact token log-probs being evaluated.
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

    def _prefix_token_id(self) -> int:
        # Pick the model's scoring prefix token used when a request has no explicit context tokens.
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

    def _max_scoring_input_length(self) -> int:
        # Resolve the finite scoring window the model/tokenizer expose, with a conservative fallback.
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

    @contextmanager
    def _scoring_attention_context(self) -> Iterator[None]:
        # Temporarily restore the base attention kernel when token-level scoring runs on a paged-attention session.
        active_attention = self.effective_attn_implementation or self.requested_attn_implementation
        if not isinstance(active_attention, str) or not active_attention.startswith("paged|"):
            yield
            return

        setter = getattr(self.model, "set_attn_implementation", None)
        base_attention = _base_attn_implementation(active_attention)
        if not callable(setter) or base_attention is None:
            yield
            return

        setter(base_attention)
        try:
            yield
        finally:
            setter(active_attention)

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

    def _disable_paged_attention(self, exc: Exception, *, fallback_batch_size: int) -> None:
        logger = get_logger()
        self._stop_continuous_batching_manager()
        with self._state_lock:
            previous_attention = self.effective_attn_implementation or self.requested_attn_implementation
            requested_attention = self.requested_attn_implementation
        base_attention = _base_attn_implementation(requested_attention)
        setter = getattr(self.model, "set_attn_implementation", None)
        if callable(setter) and base_attention is not None:
            with suppress(Exception):
                setter(base_attention)

        with self._state_lock:
            self.paged_attention_enabled = False
            self.generation_backend = "generate"
            self.effective_attn_implementation = base_attention or requested_attention
            self.standard_batch_size_cap = fallback_batch_size
            self.execution_logged = False
            effective_attention = self.effective_attn_implementation
            generation_backend = self.generation_backend
        logger.warning(
            "paged attention failed for attention=%s: %s; falling back to backend=%s attention=%s batch_size_cap=%d",
            previous_attention,
            exc,
            generation_backend,
            effective_attention,
            fallback_batch_size,
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


def _common_stop_strings(batch: list[GenerationRequest]) -> list[str] | None:
    if not batch:
        return None

    first = batch[0].stop
    if all(request.stop == first for request in batch):
        return first
    return None


def _continuous_request_signature(request: GenerationRequest) -> tuple[Any, ...]:
    return (
        tuple(request.stop),
        request.do_sample,
        request.temperature if request.do_sample else None,
    )


def _base_attn_implementation(attn_implementation: str | None) -> str | None:
    if attn_implementation is None:
        return None
    if attn_implementation.startswith("paged|"):
        return attn_implementation.split("paged|", maxsplit=1)[1]
    return attn_implementation


def _effective_attn_implementation(
    attn_implementation: str | None,
    *,
    paged_attention_enabled: bool,
) -> str | None:
    if attn_implementation is None:
        return None
    if paged_attention_enabled and not attn_implementation.startswith("paged|"):
        return f"paged|{attn_implementation}"
    return attn_implementation


def _resolve_paged_attention(
    *,
    paged_attention: bool | str,
    attn_implementation: str | None,
    model: Any,
    input_device: Any,
) -> bool:
    normalized = _normalize_paged_attention(paged_attention)
    if normalized is False:
        return False

    can_use_paged_attention = (
        getattr(input_device, "type", None) == "cuda"
        and callable(getattr(model, "generate_batch", None))
        and callable(getattr(model, "set_attn_implementation", None))
    )
    if normalized == _AUTO_PAGED_ATTENTION:
        return can_use_paged_attention and _supports_auto_paged_attention(attn_implementation)
    if normalized and not can_use_paged_attention:
        get_logger().warning(
            "paged attention requested but unsupported on this session; falling back to standard generate()"
        )
        return False
    return bool(normalized)


def _normalize_paged_attention(paged_attention: bool | str) -> bool | str:
    if paged_attention == _AUTO_PAGED_ATTENTION:
        return _AUTO_PAGED_ATTENTION
    if not isinstance(paged_attention, bool):
        raise ValueError("paged_attention must be a boolean or 'auto'")
    return paged_attention


def _supports_auto_paged_attention(attn_implementation: str | None) -> bool:
    return _base_attn_implementation(attn_implementation) == "flash_attention_2"


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
