from __future__ import annotations

from types import SimpleNamespace

from evalution.config import Model
from evalution.engines.base import GenerationRequest
from evalution.engines.transformer import Transformer, TransformerSession


def test_transformer_defaults_batch_size_to_auto() -> None:
    engine = Transformer()

    assert engine.batch_size == "auto"
    assert engine.to_dict()["batch_size"] == "auto"


def test_transformer_session_resolves_auto_batch_size_once_per_suite(monkeypatch) -> None:
    session = TransformerSession(
        config=Transformer(batch_size="auto"),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(dtype="bfloat16"),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cpu"),
    )
    requests = [GenerationRequest(prompt="Question: 1 + 1\nAnswer:")]
    stats = {
        "row_count": 1,
        "min_prompt_tokens": 12,
        "avg_prompt_tokens": 12.0,
        "max_prompt_tokens": 12,
        "max_new_tokens": 32,
        "dtype_name": "bfloat16",
        "dtype_bytes": 2,
        "total_vram_gib": 0.0,
        "parameter_count_billions": 1.0,
    }
    calls = {"estimate": 0}

    monkeypatch.setattr(
        TransformerSession,
        "_batch_size_stats",
        lambda self, batch: stats,
    )

    def fake_estimate(self, batch_stats):
        calls["estimate"] += 1
        assert batch_stats is stats
        return 16

    monkeypatch.setattr(
        TransformerSession,
        "_estimate_auto_batch_size",
        fake_estimate,
    )

    assert session.resolve_batch_size(requests) == 16
    assert session.resolve_batch_size(requests) == 16
    assert calls["estimate"] == 1
