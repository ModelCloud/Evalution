from __future__ import annotations

from types import SimpleNamespace

from transformers import PretrainedConfig

from evalution.config import Model
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.engines.transformer import Transformer, TransformerSession


def test_transformer_defaults_batch_size_to_auto() -> None:
    engine = Transformer()

    assert engine.batch_size == "auto"
    assert engine.paged_attention == "auto"
    assert engine.to_dict()["batch_size"] == "auto"
    assert engine.to_dict()["paged_attention"] == "auto"


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


def test_transformer_session_describes_auto_paged_attention_on_cuda_like_session() -> None:
    session = TransformerSession(
        config=Transformer(attn_implementation="flash_attention_2", paged_attention="auto"),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="flash_attention_2"),
            generate_batch=lambda *args, **kwargs: {},
            set_attn_implementation=lambda value: None,
        ),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    assert session.describe_execution() == {
        "requested_attn_implementation": "flash_attention_2",
        "effective_attn_implementation": "paged|flash_attention_2",
        "paged_attention": True,
        "generation_backend": "continuous_batching",
        "standard_batch_size_cap": None,
    }


def test_transformer_session_generate_uses_generate_batch_when_paged_attention_is_enabled() -> None:
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, prompts, *, add_special_tokens=False, **kwargs):
            assert add_special_tokens is False
            del kwargs
            if isinstance(prompts, str):
                prompts = [prompts]
            return {"input_ids": [[11, 12, 13] for _ in prompts]}

        def decode(self, token_ids, *, skip_special_tokens=False):
            del token_ids, skip_special_tokens
            return "The answer is 42."

    class FakeGenerateBatchOutput:
        def __init__(self, tokens):
            self.generated_tokens = tokens

    class FakeModel:
        def __init__(self) -> None:
            self.config = PretrainedConfig()
            self.config._attn_implementation = "flash_attention_2"
            self.calls: list[dict[str, object]] = []

        def generate_batch(self, inputs, *, generation_config=None, progress_bar=False):
            self.calls.append(
                {
                    "inputs": inputs,
                    "generation_config": generation_config,
                    "progress_bar": progress_bar,
                }
            )
            return {"req_0": FakeGenerateBatchOutput([101, 102, 103])}

        def set_attn_implementation(self, value: str) -> None:
            self.config._attn_implementation = value

    model = FakeModel()
    session = TransformerSession(
        config=Transformer(attn_implementation="flash_attention_2", paged_attention="auto"),
        model_config=Model(path="/tmp/model"),
        model=model,
        tokenizer=FakeTokenizer(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    outputs = session.generate(
        [GenerationRequest(prompt="Q: 40 + 2\nA:", stop=["Q:"])],
        batch_size=1,
    )

    assert len(outputs) == 1
    assert outputs[0].text == "The answer is 42."
    assert model.calls
    call = model.calls[0]
    assert call["inputs"] == [[11, 12, 13]]
    assert call["progress_bar"] is False
    assert call["generation_config"].stop_strings == ["Q:"]


def test_transformer_session_falls_back_to_standard_generate_when_paged_generation_fails(monkeypatch) -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa"),
        set_attn_implementation=lambda value: None,
    )
    session = TransformerSession(
        config=Transformer(attn_implementation="sdpa", paged_attention=True),
        model_config=Model(path="/tmp/model"),
        model=model,
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="sdpa",
        effective_attn_implementation="paged|sdpa",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )
    requests = [GenerationRequest(prompt=f"Q: {index}\nA:") for index in range(4)]
    calls: dict[str, object] = {}

    def fake_generate_paged(self, batch, *, batch_size):
        del self, batch, batch_size
        raise RuntimeError("paged failure")

    def fake_generate_standard(self, batch, *, batch_size):
        calls["batch_size"] = batch_size
        return [
            GenerationOutput(
                prompt=request.prompt or "",
                text="The answer is 42.",
            )
            for request in batch
        ]

    monkeypatch.setattr(TransformerSession, "_generate_paged", fake_generate_paged)
    monkeypatch.setattr(TransformerSession, "_generate_standard", fake_generate_standard)

    outputs = session.generate(requests, batch_size=4)

    assert len(outputs) == 4
    assert calls["batch_size"] == 2
    assert session.paged_attention_enabled is False
    assert session.generation_backend == "generate"
    assert session.effective_attn_implementation == "sdpa"
    assert session.standard_batch_size_cap == 2
