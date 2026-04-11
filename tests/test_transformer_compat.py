# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evalution.config import Model
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.engines.transformers_compat import TransformersCompat, TransformersCompatSession


def test_transformer_compat_defaults_batch_size_to_auto() -> None:
    """Verify transformer compat defaults batch size to auto."""
    engine = TransformersCompat()

    assert engine.batch_size == "auto"
    assert engine.to_dict()["batch_size"] == "auto"
    assert "resolved_engine" in engine.to_dict()


def test_transformer_compat_session_describes_generate_compat_backend() -> None:
    """Verify transformer compat session describes generate compat backend."""
    session = TransformersCompatSession(
        config=TransformersCompat(),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(dtype="bfloat16"),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="generate_compat",
    )

    assert session.describe_execution() == {
        "requested_attn_implementation": None,
        "effective_attn_implementation": None,
        "paged_attention": False,
        "generation_backend": "generate_compat",
        "standard_batch_size_cap": None,
    }


def test_transformer_compat_rejects_paged_attn_implementation() -> None:
    """Verify transformer compat rejects paged attn implementation."""
    with pytest.raises(ValueError, match="TransformersCompat does not support paged attn_implementation"):
        TransformersCompat(attn_implementation="paged|flash_attention_2").build(
            Model(path="/tmp/model")
        )


def test_transformer_compat_session_emulates_continuous_generation(monkeypatch) -> None:
    """Verify transformer compat session emulates continuous generation."""
    session = TransformersCompatSession(
        config=TransformersCompat(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(dtype="bfloat16"),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="generate_compat",
    )

    def fake_generate(self, requests, *, batch_size):
        """Support the surrounding tests with fake generate."""
        assert batch_size in {1, 2}
        return [
            GenerationOutput(
                prompt=request.prompt or "",
                text=f"out::{request.prompt}",
                metadata=dict(request.metadata),
            )
            for request in requests
        ]

    monkeypatch.setattr(
        TransformersCompatSession,
        "_generate_standard",
        fake_generate,
    )

    outputs = list(
        session.generate_continuous(
            [
                (10, GenerationRequest(prompt="alpha")),
                (11, GenerationRequest(prompt="beta")),
                (12, GenerationRequest(prompt="gamma")),
            ],
            batch_size=2,
        )
    )

    assert outputs == [
        (10, GenerationOutput(prompt="alpha", text="out::alpha", metadata={})),
        (11, GenerationOutput(prompt="beta", text="out::beta", metadata={})),
        (12, GenerationOutput(prompt="gamma", text="out::gamma", metadata={})),
    ]
