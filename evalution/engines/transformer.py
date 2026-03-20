from __future__ import annotations

import gc
from statistics import mean
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from evalution.config import Model
from evalution.engines.base import GenerationOutput, GenerationRequest
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

        effective_batch_size = batch_size or self.resolve_batch_size(requests)
        if not self.paged_attention_enabled and self.standard_batch_size_cap is not None:
            effective_batch_size = min(effective_batch_size, self.standard_batch_size_cap)
        self._log_generation_execution()
        if self.paged_attention_enabled:
            try:
                return self._generate_paged(requests, batch_size=effective_batch_size)
            except Exception as exc:
                fallback_batch_size = _fallback_batch_size(effective_batch_size)
                self._disable_paged_attention(exc, fallback_batch_size=fallback_batch_size)
                return self._generate_standard(requests, batch_size=fallback_batch_size)
        return self._generate_standard(requests, batch_size=effective_batch_size)

    def close(self) -> None:
        self.stop_criteria_cache.clear()
        self.auto_batch_size_cache.clear()
        with suppress(Exception):
            del self.model
        with suppress(Exception):
            del self.tokenizer
        with suppress(Exception):
            del self.prepare_tokenizer
        gc.collect()
        with suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        return {
            "requested_attn_implementation": self.requested_attn_implementation,
            "effective_attn_implementation": self.effective_attn_implementation,
            "paged_attention": self.paged_attention_enabled,
            "generation_backend": self.generation_backend,
            "standard_batch_size_cap": self.standard_batch_size_cap,
        }

    def prepare_requests(self, requests: list[GenerationRequest]) -> list[GenerationRequest]:
        tokenizer = self.prepare_tokenizer or self.tokenizer
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
        cached = self.auto_batch_size_cache.get(cache_key)
        if cached is not None:
            return cached

        resolved = self._estimate_auto_batch_size(stats)
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
        criteria = self.stop_criteria_cache.get(cache_key)
        if criteria is None:
            criteria = StopStringCriteria(self.tokenizer, list(cache_key))
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
            rendered_prompts = [self._render_request(request) for request in requests if request.input_ids is None]
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
        outputs: list[GenerationOutput] = []
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            rendered_prompts = None
            encoded = None
            generated = None
            generation_config = None
            try:
                rendered_prompts = [self._render_request(request) for request in batch]
                encoded = [
                    list(request.input_ids)
                    if request.input_ids is not None
                    else self.tokenizer(
                        rendered_prompts[index],
                        add_special_tokens=False,
                    )["input_ids"]
                    for index, request in enumerate(batch)
                ]
                generation_config = self._build_generation_config(batch)
                generated = self.model.generate_batch(
                    encoded,
                    generation_config=generation_config,
                    progress_bar=False,
                )
                missing_keys = [
                    f"req_{index}"
                    for index in range(len(batch))
                    if f"req_{index}" not in generated
                ]
                if missing_keys:
                    raise RuntimeError(
                        "continuous batching returned incomplete results: "
                        f"missing {len(missing_keys)}/{len(batch)} request(s); "
                        f"first_missing={missing_keys[0]}"
                    )

                for index in range(len(batch)):
                    request_output = generated[f"req_{index}"]
                    generated_tokens = request_output.generated_tokens
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
                    del request_output
            finally:
                if generated is not None:
                    del generated
                if generation_config is not None:
                    del generation_config
                if encoded is not None:
                    del encoded
                del batch
                if rendered_prompts is not None:
                    del rendered_prompts
        return outputs

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

    def _log_generation_execution(self) -> None:
        if self.execution_logged:
            return
        get_logger().info(
            "transformer generation using backend=%s attention=%s batch_size_cap=%s",
            self.generation_backend,
            self.effective_attn_implementation or self.requested_attn_implementation,
            self.standard_batch_size_cap,
        )
        self.execution_logged = True

    def _disable_paged_attention(self, exc: Exception, *, fallback_batch_size: int) -> None:
        logger = get_logger()
        previous_attention = self.effective_attn_implementation or self.requested_attn_implementation
        base_attention = _base_attn_implementation(self.requested_attn_implementation)
        setter = getattr(self.model, "set_attn_implementation", None)
        if callable(setter) and base_attention is not None:
            with suppress(Exception):
                setter(base_attention)

        self.paged_attention_enabled = False
        self.generation_backend = "generate"
        self.effective_attn_implementation = base_attention or self.requested_attn_implementation
        self.standard_batch_size_cap = fallback_batch_size
        self.execution_logged = False
        logger.warning(
            "paged attention failed for attention=%s: %s; falling back to backend=%s attention=%s batch_size_cap=%d",
            previous_attention,
            exc,
            self.generation_backend,
            self.effective_attn_implementation,
            fallback_batch_size,
        )


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
