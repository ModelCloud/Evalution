from __future__ import annotations

import gc
from statistics import mean
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from typing import Any

from evalution.config import Model
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.engines.memory import build_memory_profile, gib_to_bytes, resolve_dtype
from evalution.logbar import get_logger

_AUTO_BATCH_SIZE = "auto"
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
    stop_criteria_cache: dict[tuple[str, ...], Any] = field(default_factory=dict, repr=False)
    auto_batch_size_cache: dict[tuple[Any, ...], int] = field(default_factory=dict, repr=False)

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
        attn_implementation = config.attention_impl or config.attn_implementation
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

        return cls(
            config=config,
            model_config=model_config,
            model=model,
            tokenizer=tokenizer,
            input_device=input_device,
        )

    def generate(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int | None = None,
    ) -> list[GenerationOutput]:
        import torch
        from transformers import StoppingCriteriaList

        if not requests:
            return []

        effective_batch_size = batch_size or self.resolve_batch_size(requests)
        outputs: list[GenerationOutput] = []

        for start in range(0, len(requests), effective_batch_size):
            batch = requests[start : start + effective_batch_size]
            rendered_prompts = None
            encoded = None
            generated = None
            try:
                rendered_prompts = [self._render_request(request) for request in batch]
                encoded = self.tokenizer(
                    rendered_prompts,
                    return_tensors="pt",
                    padding=True,
                )
                encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
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
            finally:
                if generated is not None:
                    del generated
                if encoded is not None:
                    del encoded
                del batch
                if rendered_prompts is not None:
                    del rendered_prompts

        return outputs

    def close(self) -> None:
        self.stop_criteria_cache.clear()
        self.auto_batch_size_cache.clear()
        with suppress(Exception):
            del self.model
        with suppress(Exception):
            del self.tokenizer
        gc.collect()
        with suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _render_request(self, request: GenerationRequest) -> str:
        if request.messages is not None:
            return self.tokenizer.apply_chat_template(
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
        import torch

        rendered_prompts = [self._render_request(request) for request in requests]
        encoded = self.tokenizer(rendered_prompts, padding=False)
        prompt_lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
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


def _common_stop_strings(batch: list[GenerationRequest]) -> list[str] | None:
    if not batch:
        return None

    first = batch[0].stop
    if all(request.stop == first for request in batch):
        return first
    return None


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
