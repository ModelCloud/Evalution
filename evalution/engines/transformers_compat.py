# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass

from evalution.config import Model
from evalution.engines.transformers_common import (
    BaseTransformerSession,
    _TransformersCommonConfig,
    _requests_paged_attention,
    load_transformer_runtime,
)


@dataclass(slots=True)
class TransformersCompat(_TransformersCommonConfig):
    # Use the fixed-batch compatibility engine for transformers releases that predate continuous batching.

    # Build the compatibility session that emulates the continuous-refill API on top of standard generate().
    def build(self, model: Model) -> BaseTransformerSession:
        if _requests_paged_attention(self.attn_implementation):
            raise ValueError("TransformersCompat does not support paged attn_implementation")
        self.resolved_engine = "TransformersCompat"
        return TransformersCompatSession.from_config(self, model)

    @classmethod
    def from_transformers(cls, engine: object) -> TransformersCompat:
        # Copy only the shared controls from the modern engine when auto-falling back to compat mode.
        return cls(
            dtype=getattr(engine, "dtype", "auto"),
            attn_implementation=getattr(engine, "attn_implementation", None),
            device=getattr(engine, "device", None),
            device_map=getattr(engine, "device_map", None),
            seed=getattr(engine, "seed", None),
            batch_size=getattr(engine, "batch_size", "auto"),
            max_new_tokens=getattr(engine, "max_new_tokens", 256),
            trust_remote_code=getattr(engine, "trust_remote_code", None),
            padding_side=getattr(engine, "padding_side", "left"),
        )


@dataclass(slots=True)
class TransformersCompatSession(BaseTransformerSession):
    # Run the shared transformer session logic without paged attention or an upstream batching manager.

    @classmethod
    def from_config(
        cls,
        config: TransformersCompat,
        model_config: Model,
    ) -> TransformersCompatSession:
        runtime = load_transformer_runtime(config, model_config)
        return cls(
            config=config,
            model_config=model_config,
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            prepare_tokenizer=runtime.prepare_tokenizer,
            input_device=runtime.input_device,
            requested_attn_implementation=runtime.requested_attn_implementation,
            effective_attn_implementation=runtime.requested_attn_implementation,
            paged_attention_enabled=False,
            generation_backend="generate_compat",
        )
