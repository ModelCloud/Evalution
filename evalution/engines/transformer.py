from __future__ import annotations

import gc
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from typing import Any

from evalution.config import Model
from evalution.engines.base import GenerationOutput, GenerationRequest


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
    batch_size: int = 1
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
        resolved_dtype = _resolve_dtype(config.dtype)
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

        effective_batch_size = batch_size or self.config.batch_size
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

    def _get_stop_criteria(self, stop_strings: list[str]) -> Any:
        from transformers import StopStringCriteria

        cache_key = tuple(stop_strings)
        criteria = self.stop_criteria_cache.get(cache_key)
        if criteria is None:
            criteria = StopStringCriteria(self.tokenizer, list(cache_key))
            self.stop_criteria_cache[cache_key] = criteria
        return criteria


def _resolve_dtype(dtype: str | None) -> Any:
    if dtype is None:
        return None
    if dtype == "auto":
        return "auto"

    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype}") from exc


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
