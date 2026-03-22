# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from types import SimpleNamespace

from evalution.config import Model
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.engines.transformer_compat import TransformerCompat, TransformerCompatSession


def test_transformer_compat_defaults_batch_size_to_auto() -> None:
    engine = TransformerCompat()

    assert engine.batch_size == "auto"
    assert engine.to_dict()["batch_size"] == "auto"
    assert "resolved_engine" in engine.to_dict()


def test_transformer_compat_session_describes_generate_compat_backend() -> None:
    session = TransformerCompatSession(
        config=TransformerCompat(),
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


def test_transformer_compat_session_emulates_continuous_generation(monkeypatch) -> None:
    session = TransformerCompatSession(
        config=TransformerCompat(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(dtype="bfloat16"),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="generate_compat",
    )

    def fake_generate(self, requests, *, batch_size):
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
        TransformerCompatSession,
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
