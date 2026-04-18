# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalution.config import Model
from evalution.engines.base import GenerationRequest, LoglikelihoodOutput, LoglikelihoodRequest
from evalution.engines.tinygrad_engine import (
    _TinygradRuntimeProfile,
    Tinygrad,
    TinygradSession,
    _import_tinygrad_modules,
    _load_tinygrad_runtime,
    _resolve_tinygrad_device,
    _resolve_tinygrad_runtime_profile,
)


class FakeTinygradTokenizer:
    """Small tinygrad-tokenizer test double used by the prompt-shaping tests."""

    def __init__(self) -> None:
        """Keep the tokenizer state explicit for the surrounding assertions."""

        self.encoded_messages: list[str] = []

    def prefix(self) -> list[int]:
        """Return the synthetic BOS prefix used by the fake GGUF tokenizer."""

        return [1]

    def role(self, role: str) -> list[int]:
        """Encode the chat role into one deterministic sentinel id."""

        return [100 if role == "user" else 200]

    def end_turn(self) -> list[int]:
        """Return the synthetic end-of-turn token."""

        return [9]

    def is_end(self, token_id: int) -> bool:
        """Treat token id zero as the synthetic end token."""

        return token_id == 0

    def encode(self, text: str) -> list[int]:
        """Encode text deterministically so the tests can assert exact token ids."""

        self.encoded_messages.append(text)
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode the fake token ids back into deterministic text."""

        mapping = {65: "abc", 66: "STOP", 67: "tail"}
        return "".join(mapping.get(token_id, chr(token_id)) for token_id in token_ids)


class FakePrepareTokenizer:
    """Small chat-template tokenizer double used by the rendering tests."""

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        """Render chat messages into one deterministic prompt string."""

        rendered = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        rendered = rendered + ("|assistant:" if add_generation_prompt else "")
        if tokenize:
            return {"input_ids": [len(rendered)]}
        return rendered

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text deterministically so the tests can assert exact token ids."""

        prefix = [999] if add_special_tokens else []
        return prefix + [len(text)]


def _build_session(
    *,
    tokenizer=None,
    prepare_tokenizer=None,
    load_format: str = "gguf",
    compute_device: str = "CPU",
    runtime_profile: _TinygradRuntimeProfile | None = None,
) -> TinygradSession:
    """Construct one minimal tinygrad session test double."""

    fake_modules = SimpleNamespace(
        helpers=SimpleNamespace(Context=lambda **kwargs: nullcontext()),
        Tensor=None,
        dtypes=SimpleNamespace(int32="int32"),
    )
    return TinygradSession(
        config=Tinygrad(device="cpu", batch_size=2),
        model_config=Model(path="/tmp/model.gguf"),
        modules=fake_modules,
        model=SimpleNamespace(blk=[]),
        tokenizer=tokenizer or FakeTinygradTokenizer(),
        prepare_tokenizer=prepare_tokenizer,
        load_format=load_format,
        model_type="llama",
        compute_device=compute_device,
        max_context=32,
        runtime_profile=runtime_profile or _TinygradRuntimeProfile(name="test", jit=None, jitbeam=None),
    )


def test_tinygrad_engine_defaults_batch_size_to_auto() -> None:
    """Verify the tinygrad engine exposes stable defaults for config serialization."""

    engine = Tinygrad()

    assert engine.batch_size == "auto"
    assert engine.max_context is None
    assert engine.tinygrad_path is None
    assert engine.jit is None
    assert engine.jitbeam is None
    assert engine.to_dict()["resolved_engine"] is None


def test_resolve_tinygrad_device_normalizes_evalution_device_strings() -> None:
    """Verify Evalution-style device overrides are normalized into tinygrad device names."""

    assert _resolve_tinygrad_device("auto") is None
    assert _resolve_tinygrad_device("cuda:0") == "CUDA"
    assert _resolve_tinygrad_device("cuda:3") == "CUDA:3"
    assert _resolve_tinygrad_device("cpu") == "CPU"
    assert _resolve_tinygrad_device("nv:1") == "NV:1"


def test_import_tinygrad_uses_checkout_fallback(monkeypatch, tmp_path: Path) -> None:
    """Verify local checkout discovery is used when importing tinygrad initially fails."""

    checkout = tmp_path / "tinygrad"
    checkout.mkdir()
    fake_modules = object()

    def fake_load():
        """Support the surrounding tests with a path-sensitive import helper."""

        if str(checkout) not in __import__("sys").path:
            raise ModuleNotFoundError("No module named 'tinygrad'")
        return fake_modules

    monkeypatch.setattr("evalution.engines.tinygrad_engine._load_tinygrad_modules", fake_load)

    imported = _import_tinygrad_modules(str(checkout))

    assert imported is fake_modules


def test_tinygrad_prepare_requests_renders_messages_through_builtin_tokenizer() -> None:
    """Verify GGUF chat requests fall back to tinygrad's built-in tokenizer when no HF tokenizer is supplied."""

    tokenizer = FakeTinygradTokenizer()
    session = _build_session(tokenizer=tokenizer)

    prepared = session.prepare_requests(
        [
            GenerationRequest(
                messages=[{"role": "user", "content": "hi"}],
                add_generation_prompt=True,
            )
        ]
    )

    assert prepared[0].input_ids == [1, 100, ord("h"), ord("i"), 9, 200]
    assert prepared[0].rendered_prompt is None


def test_tinygrad_prepare_requests_uses_chat_template_tokenizer_when_available() -> None:
    """Verify message rendering prefers tokenizer.apply_chat_template when a prepare tokenizer exists."""

    session = _build_session(
        tokenizer=FakeTinygradTokenizer(),
        prepare_tokenizer=FakePrepareTokenizer(),
    )

    prepared = session.prepare_requests(
        [
            GenerationRequest(
                messages=[{"role": "user", "content": "hi"}],
                add_generation_prompt=True,
            )
        ]
    )

    assert prepared[0].rendered_prompt == "user:hi|assistant:"
    assert prepared[0].input_ids == [18]


def test_tinygrad_generate_truncates_stop_strings(monkeypatch) -> None:
    """Verify generated text is truncated at the first configured stop string."""

    session = _build_session()
    generated_steps = iter(([65], [66]))
    monkeypatch.setattr(session, "_reset_model_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(session, "_batched_next_token_ids", lambda *args, **kwargs: next(generated_steps))

    outputs = session.generate(
        [
            GenerationRequest(
                prompt="hello",
                input_ids=[1, 2, 3],
                stop=["STOP"],
                max_new_tokens=2,
            )
        ]
    )

    assert outputs[0].text == "abc"
    assert outputs[0].metadata["finish_reason"] == "stop"
    assert outputs[0].metadata["completion_token_count"] == 2


def test_tinygrad_generate_batches_same_signature_requests_together(monkeypatch) -> None:
    """Verify same-shape requests are coalesced into one low-level tinygrad batch."""

    session = _build_session()
    seen_batches: list[list[str | None]] = []

    def fake_generate_batch(requests: list[GenerationRequest]) -> list:
        seen_batches.append([request.prompt for request in requests])
        return [
            SimpleNamespace(
                prompt=request.prompt or "",
                text=f"out:{request.prompt}",
                metadata={},
            )
            for request in requests
        ]

    monkeypatch.setattr(session, "_generate_batch", fake_generate_batch)

    outputs = session.generate(
        [
            GenerationRequest(prompt="one", input_ids=[1, 2, 3], max_new_tokens=4),
            GenerationRequest(prompt="two", input_ids=[4, 5, 6], max_new_tokens=4),
        ],
        batch_size=2,
    )

    assert seen_batches == [["one", "two"]]
    assert [output.text for output in outputs] == ["out:one", "out:two"]


def test_tinygrad_generate_batches_mixed_prompt_lengths_together(monkeypatch) -> None:
    """Verify static batching now keeps mixed prompt lengths together when decode settings match."""

    session = _build_session()
    seen_batches: list[list[str | None]] = []

    def fake_generate_batch(requests: list[GenerationRequest]) -> list:
        seen_batches.append([request.prompt for request in requests])
        return [
            SimpleNamespace(
                prompt=request.prompt or "",
                text=f"out:{request.prompt}",
                metadata={},
            )
            for request in requests
        ]

    monkeypatch.setattr(session, "_generate_batch", fake_generate_batch)

    outputs = session.generate(
        [
            GenerationRequest(prompt="same-a", input_ids=[1, 2, 3], max_new_tokens=4),
            GenerationRequest(prompt="different-len", input_ids=[4, 5], max_new_tokens=4),
            GenerationRequest(prompt="same-b", input_ids=[6, 7, 8], max_new_tokens=4),
        ],
        batch_size=4,
    )

    assert seen_batches == [["same-a", "different-len", "same-b"]]
    assert [output.text for output in outputs] == ["out:same-a", "out:different-len", "out:same-b"]


def test_tinygrad_generate_splits_requests_by_batch_signature(monkeypatch) -> None:
    """Verify static batching still splits requests when sampling controls diverge."""

    session = _build_session()
    seen_batches: list[list[str | None]] = []

    def fake_generate_batch(requests: list[GenerationRequest]) -> list:
        seen_batches.append([request.prompt for request in requests])
        return [
            SimpleNamespace(
                prompt=request.prompt or "",
                text=f"out:{request.prompt}",
                metadata={},
            )
            for request in requests
        ]

    monkeypatch.setattr(session, "_generate_batch", fake_generate_batch)

    outputs = session.generate(
        [
            GenerationRequest(prompt="greedy-a", input_ids=[1, 2, 3], max_new_tokens=4),
            GenerationRequest(prompt="sampled", input_ids=[4, 5], max_new_tokens=4, do_sample=True, temperature=0.8),
            GenerationRequest(prompt="greedy-b", input_ids=[6, 7, 8], max_new_tokens=4),
        ],
        batch_size=4,
    )

    assert seen_batches == [["greedy-a", "greedy-b"], ["sampled"]]
    assert [output.text for output in outputs] == ["out:greedy-a", "out:sampled", "out:greedy-b"]


def test_tinygrad_cuda_execution_metadata_reports_runtime_profile() -> None:
    """Verify CUDA execution reports the resolved tinygrad runtime controls explicitly."""

    session = _build_session(
        load_format="gguf",
        compute_device="CUDA",
        runtime_profile=_TinygradRuntimeProfile(name="rtx4090", jit=2, jitbeam=0),
    )

    assert session.describe_execution()["jit"] == 2
    assert session.describe_execution()["jitbeam"] == 0
    assert session.describe_execution()["jit_profile"] == "rtx4090"


def test_tinygrad_runtime_profile_uses_profiled_cuda_defaults(monkeypatch) -> None:
    """Verify CUDA sessions pick the profiled JIT settings when the user leaves them unset."""

    monkeypatch.setattr(
        "evalution.engines.tinygrad_engine._visible_cuda_profile_bucket",
        lambda: "a100",
    )

    profile = _resolve_tinygrad_runtime_profile(Tinygrad(), "CUDA")

    assert profile.name == "a100"
    assert profile.jit == 2
    assert profile.jitbeam == 0


def test_tinygrad_runtime_profile_respects_explicit_user_overrides(monkeypatch) -> None:
    """Verify explicit engine controls override the profiled CUDA defaults field by field."""

    monkeypatch.setattr(
        "evalution.engines.tinygrad_engine._visible_cuda_profile_bucket",
        lambda: "rtx4090",
    )

    profile = _resolve_tinygrad_runtime_profile(Tinygrad(jit=0, jitbeam=3), "CUDA")

    assert profile.name == "rtx4090"
    assert profile.jit == 0
    assert profile.jitbeam == 3


def test_tinygrad_runtime_rejects_non_gguf_model_paths() -> None:
    """Verify the public tinygrad engine rejects removed dense-checkpoint execution clearly."""

    with pytest.raises(ValueError, match="GGUF checkpoints only"):
        _load_tinygrad_runtime(
            Tinygrad(),
            Model(path="/tmp/model"),
        )


def test_tinygrad_loglikelihood_aggregates_chunk_scores(monkeypatch) -> None:
    """Verify per-chunk scores are folded back into one request-level loglikelihood result."""

    session = _build_session()
    monkeypatch.setattr(
        session,
        "_prepare_loglikelihood_request",
        lambda request: ([1], [2, 3], dict(request.metadata)),
    )
    monkeypatch.setattr(
        session,
        "_build_loglikelihood_chunks",
        lambda **kwargs: [
            ([1, 2], 1, 1, {"chunk": 0}),
            ([1, 2, 3], 2, 1, {"chunk": 1}),
        ],
    )
    monkeypatch.setattr(
        session,
        "_score_loglikelihood_chunk",
        lambda **kwargs: LoglikelihoodOutput(
            logprob=-0.5,
            is_greedy=kwargs["metadata"]["chunk"] == 0,
            token_count=1,
            metadata=kwargs["metadata"],
        ),
    )

    outputs = session.loglikelihood(
        [
            LoglikelihoodRequest(
                context="q",
                continuation="a",
                metadata={"name": "sample"},
            )
        ]
    )

    assert outputs[0].logprob == -1.0
    assert outputs[0].is_greedy is False
    assert outputs[0].token_count == 2
    assert outputs[0].metadata == {"name": "sample"}
