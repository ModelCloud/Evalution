# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evalution.config import Model
from evalution.engines.base import (
    GenerationOutput,
    GenerationRequest,
    LoglikelihoodOutput,
    LoglikelihoodRequest,
)
from evalution.engines.llama_cpp_engine import (
    LlamaCpp,
    LlamaCppSession,
    _import_llama_cpp,
)


class FakePrepareTokenizer:
    """Small chat-template test double used by the prompt-rendering tests."""

    def __init__(self) -> None:
        """Keep the tokenizer state explicit for surrounding assertions."""

        self.padding_side = "left"

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        """Render chat messages into one deterministic prompt string."""

        del tokenize, add_generation_prompt
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


class FakeLlamaRuntime:
    """Minimal llama.cpp stand-in for generation and scoring tests."""

    def __init__(self) -> None:
        """Initialize the fake runtime and capture calls for assertions."""

        self.create_completion_calls: list[dict[str, object]] = []
        self.create_chat_completion_calls: list[dict[str, object]] = []
        self.reset_calls = 0
        self.eval_calls: list[list[int]] = []
        self._scores = np.zeros((1, 4), dtype=np.single)

    def create_completion(self, **payload):
        """Return one deterministic completion payload."""

        self.create_completion_calls.append(payload)
        prompt = payload["prompt"]
        if isinstance(prompt, list):
            prompt_text = "".join(chr(token_id) for token_id in prompt if token_id > 1)
        else:
            prompt_text = str(prompt)
        return {
            "choices": [
                {
                    "text": f"{prompt_text} -> completion",
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_text),
                "completion_tokens": 2,
            },
        }

    def create_chat_completion(self, **payload):
        """Return one deterministic chat-completion payload."""

        self.create_chat_completion_calls.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": "chat completion",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
            },
        }

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        """Encode bytes by mapping characters to ordinal ids and optionally prepending BOS."""

        del special
        tokens = [ord(char) for char in text.decode("utf-8")]
        return ([1] + tokens) if add_bos else tokens

    def reset(self) -> None:
        """Track session resets between generation and scoring calls."""

        self.reset_calls += 1

    def eval(self, tokens: list[int]) -> None:
        """Populate one deterministic log-probability table aligned with the scored tokens."""

        self.eval_calls.append(list(tokens))
        vocab_size = max(tokens) + 2
        scores = np.full((len(tokens), vocab_size), -10.0, dtype=np.single)
        for row_index, token_id in enumerate(tokens[1:]):
            scores[row_index, token_id] = -0.25
            scores[row_index, 0] = -1.0
        self._scores = scores

    def token_bos(self) -> int:
        """Return the synthetic prefix token used by the session scorer."""

        return 1

    def n_ctx(self) -> int:
        """Expose the fake runtime context length to the session."""

        return 8

    def close(self) -> None:
        """Release the resources owned by this object."""

        return None


class FakeLlamaModule:
    """Small module-shaped object that mimics the parts of llama_cpp we use."""

    def __init__(self, *, gpu_offload_supported: bool) -> None:
        """Expose GPU support and a capturing Llama constructor for the build tests."""

        self.init_kwargs: list[dict[str, object]] = []
        self.llama_cpp = SimpleNamespace(
            llama_supports_gpu_offload=lambda: gpu_offload_supported,
        )

        module = self

        class CapturingLlama:
            """Capture constructor kwargs and expose the static logprob helper."""

            def __init__(self, **kwargs) -> None:
                """Record the runtime constructor kwargs used by the session builder."""

                module.init_kwargs.append(kwargs)

            @staticmethod
            def logits_to_logprobs(logits):
                """Pass through test logits that are already expressed as log-probabilities."""

                return logits

        self.Llama = CapturingLlama


def _build_session(
    *,
    prepare_tokenizer=None,
    device: str | None = None,
    continuous_batching: bool = False,
) -> LlamaCppSession:
    """Construct one minimal llama.cpp session test double."""

    fake_llm = FakeLlamaRuntime()
    fake_module = SimpleNamespace(
        Llama=SimpleNamespace(logits_to_logprobs=staticmethod(lambda logits: logits)),
    )
    return LlamaCppSession(
        config=LlamaCpp(
            device=device,
            batch_size=2,
            continuous_batching=continuous_batching,
        ),
        model_config=Model(path="/tmp/model.gguf"),
        llm=fake_llm,
        llama_module=fake_module,
        prepare_tokenizer=prepare_tokenizer,
        requested_device=device or "auto",
        effective_device=device or "cpu",
        gpu_offload_supported=device == "cuda",
        effective_n_gpu_layers=-1 if device == "cuda" else 0,
    )


def test_llama_cpp_engine_defaults_batch_size_to_auto() -> None:
    """Verify the llama.cpp engine exposes stable defaults for config serialization."""

    engine = LlamaCpp()

    assert engine.batch_size == "auto"
    assert engine.continuous_batching is True
    assert engine.n_ctx == 4096
    assert engine.logits_all is True
    assert engine.to_dict()["resolved_engine"] is None


def test_import_llama_cpp_uses_native_python_import(monkeypatch) -> None:
    """Verify llama_cpp import reads from the active Python environment only."""
    fake_module = object()

    def fake_import_module(name: str):
        """Support the surrounding tests with a fake import hook."""

        assert name == "llama_cpp"
        return fake_module

    monkeypatch.setattr(
        "evalution.engines.llama_cpp_engine.importlib.import_module",
        fake_import_module,
    )

    imported = _import_llama_cpp()

    assert imported is fake_module


def test_llama_cpp_build_enables_full_gpu_offload_when_available(monkeypatch) -> None:
    """Verify session construction defaults to full GPU offload on CUDA-capable builds."""

    fake_module = FakeLlamaModule(gpu_offload_supported=True)
    monkeypatch.setattr(
        "evalution.engines.llama_cpp_engine._import_llama_cpp",
        lambda: fake_module,
    )
    monkeypatch.setattr(
        "evalution.engines.llama_cpp_engine._maybe_load_prepare_tokenizer",
        lambda **kwargs: None,
    )

    session = LlamaCpp(device="cuda").build(Model(path="/tmp/model.gguf"))

    assert isinstance(session, LlamaCppSession)
    assert fake_module.init_kwargs == [
        {
            "model_path": "/tmp/model.gguf",
            "n_ctx": 4096,
            "n_batch": 512,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "main_gpu": 0,
            "split_mode": 1,
            "tensor_split": None,
            "flash_attn": False,
            "offload_kqv": True,
            "use_mmap": True,
            "use_mlock": False,
            "chat_format": None,
            "verbose": False,
            "logits_all": True,
        }
    ]
    assert session.describe_execution()["requested_device"] == "cuda"
    assert session.describe_execution()["device"] == "cuda"
    assert session.describe_execution()["n_gpu_layers"] == -1
    assert session.describe_execution()["continuous_batching"] is True
    assert session.describe_execution()["generation_backend"] == "continuous_batching"


def test_llama_cpp_build_falls_back_to_cpu_without_gpu_offload(monkeypatch) -> None:
    """Verify CUDA requests degrade to CPU when the installed llama.cpp build is CPU-only."""

    fake_module = FakeLlamaModule(gpu_offload_supported=False)
    monkeypatch.setattr(
        "evalution.engines.llama_cpp_engine._import_llama_cpp",
        lambda: fake_module,
    )
    monkeypatch.setattr(
        "evalution.engines.llama_cpp_engine._maybe_load_prepare_tokenizer",
        lambda **kwargs: None,
    )

    session = LlamaCpp(device="cuda", n_gpu_layers=12).build(Model(path="/tmp/model.gguf"))

    assert session.describe_execution()["requested_device"] == "cuda"
    assert session.describe_execution()["device"] == "cpu"
    assert session.describe_execution()["n_gpu_layers"] == 0
    assert fake_module.init_kwargs[0]["n_gpu_layers"] == 0


def test_llama_cpp_build_accepts_mlx_device_when_gpu_offload_is_available(monkeypatch) -> None:
    """Verify MLX-style device requests are accepted as a GPU-backed llama.cpp mode."""

    fake_module = FakeLlamaModule(gpu_offload_supported=True)
    monkeypatch.setattr(
        "evalution.engines.llama_cpp_engine._import_llama_cpp",
        lambda: fake_module,
    )
    monkeypatch.setattr(
        "evalution.engines.llama_cpp_engine._maybe_load_prepare_tokenizer",
        lambda **kwargs: None,
    )

    session = LlamaCpp(device="mlx").build(Model(path="/tmp/model.gguf"))

    assert session.describe_execution()["requested_device"] == "mlx"
    assert session.describe_execution()["device"] == "mlx"
    assert session.describe_execution()["n_gpu_layers"] == -1


def test_llama_cpp_session_generate_uses_native_continuous_batching_when_enabled(monkeypatch) -> None:
    """Verify batched generation dispatches through the native continuous-batching runner."""

    session = _build_session(continuous_batching=True)
    seen_max_concurrency: list[int] = []

    def fake_native_runner(self, request_items, *, max_concurrency, put_result):
        """Support the surrounding tests with a deterministic native continuous-batching stub."""

        seen_max_concurrency.append(max_concurrency)
        for request_key, request in request_items:
            put_result(
                request_key,
                GenerationOutput(
                    prompt=request.prompt or "",
                    text=f"native::{request.prompt}",
                    metadata={},
                ),
            )

    monkeypatch.setattr(
        LlamaCppSession,
        "_run_native_continuous_generation",
        fake_native_runner,
    )

    outputs = session.generate(
        [
            GenerationRequest(prompt="alpha"),
            GenerationRequest(prompt="beta"),
            GenerationRequest(prompt="gamma"),
        ],
        batch_size=2,
    )

    assert seen_max_concurrency == [2]
    assert outputs == [
        GenerationOutput(prompt="alpha", text="native::alpha", metadata={}),
        GenerationOutput(prompt="beta", text="native::beta", metadata={}),
        GenerationOutput(prompt="gamma", text="native::gamma", metadata={}),
    ]


def test_llama_cpp_session_generate_uses_completion_and_chat_paths() -> None:
    """Verify text prompts use create_completion while chat messages fall back to chat completion."""

    session = _build_session()

    outputs = session.generate(
        [
            GenerationRequest(prompt="Hello", metadata={"slot": 1}),
            GenerationRequest(
                messages=[{"role": "user", "content": "Hi"}],
                metadata={"slot": 2},
            ),
        ]
    )

    assert session.llm.create_completion_calls == [
        {
            "prompt": "Hello",
            "max_tokens": 256,
            "temperature": 0.0,
            "stop": None,
            "seed": None,
            "stream": False,
        }
    ]
    assert session.llm.create_chat_completion_calls == [
        {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 256,
            "temperature": 0.0,
            "stop": None,
            "seed": None,
            "stream": False,
            "logprobs": False,
        }
    ]
    assert outputs == [
        GenerationOutput(
            prompt="Hello",
            text="Hello -> completion",
            metadata={
                "slot": 1,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        ),
        GenerationOutput(
            prompt="user: Hi",
            text="chat completion",
            metadata={
                "slot": 2,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        ),
    ]


def test_llama_cpp_session_generate_renders_messages_with_prepare_tokenizer() -> None:
    """Verify chat requests use the supplied tokenizer template when one is available."""

    session = _build_session(prepare_tokenizer=FakePrepareTokenizer())

    outputs = session.generate(
        [
            GenerationRequest(
                messages=[{"role": "user", "content": "Hi"}],
            )
        ]
    )

    assert session.llm.create_chat_completion_calls == []
    assert session.llm.create_completion_calls == [
        {
            "prompt": "user: Hi",
            "max_tokens": 256,
            "temperature": 0.0,
            "stop": None,
            "seed": None,
            "stream": False,
        }
    ]
    assert outputs[0].prompt == "user: Hi"


def test_llama_cpp_session_generate_continuous_uses_native_scheduler_when_enabled(monkeypatch) -> None:
    """Verify continuous generation routes through the native llama.cpp scheduler by default."""

    session = _build_session(continuous_batching=True)
    seen_max_concurrency: list[int] = []

    def fake_native_runner(self, request_items, *, max_concurrency, put_result):
        """Support the surrounding tests with a deterministic native continuous-batching stub."""

        seen_max_concurrency.append(max_concurrency)
        for request_key, request in request_items:
            put_result(
                request_key,
                GenerationOutput(
                    prompt=request.prompt or "",
                    text=f"out::{request.prompt}",
                    metadata={},
                ),
            )

    monkeypatch.setattr(
        LlamaCppSession,
        "_run_native_continuous_generation",
        fake_native_runner,
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

    assert seen_max_concurrency == [2]
    assert outputs == [
        (10, GenerationOutput(prompt="alpha", text="out::alpha", metadata={})),
        (11, GenerationOutput(prompt="beta", text="out::beta", metadata={})),
        (12, GenerationOutput(prompt="gamma", text="out::gamma", metadata={})),
    ]


def test_llama_cpp_session_generate_continuous_uses_fixed_batches_when_disabled(monkeypatch) -> None:
    """Verify disabling native continuous batching falls back to the fixed-size queue path."""

    session = _build_session(continuous_batching=False)
    seen_batch_sizes: list[int] = []

    def fake_generate(self, requests, *, batch_size=None):
        """Support the surrounding tests with a deterministic generation stub."""

        seen_batch_sizes.append(int(batch_size or 0))
        return [
            GenerationOutput(prompt=request.prompt or "", text=f"out::{request.prompt}", metadata={})
            for request in requests
        ]

    monkeypatch.setattr(LlamaCppSession, "generate", fake_generate)

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

    assert seen_batch_sizes == [2, 1]
    assert outputs == [
        (10, GenerationOutput(prompt="alpha", text="out::alpha", metadata={})),
        (11, GenerationOutput(prompt="beta", text="out::beta", metadata={})),
        (12, GenerationOutput(prompt="gamma", text="out::gamma", metadata={})),
    ]


def test_llama_cpp_native_generation_budget_respects_context_window() -> None:
    """Verify native continuous batching clamps completion length to the live context budget."""

    session = _build_session(continuous_batching=True)

    max_completion_tokens, reserved_tokens = session._resolve_native_generation_budget(
        prompt_tokens=[1, 2, 3, 4, 5, 6, 7],
        requested_max_new_tokens=8,
        max_context_tokens=8,
    )

    assert max_completion_tokens == 1
    assert reserved_tokens == 8


def test_llama_cpp_native_generation_budget_rejects_oversized_prompts() -> None:
    """Verify prompts larger than the runtime context fail fast before native scheduling starts."""

    session = _build_session(continuous_batching=True)

    with pytest.raises(ValueError, match="generation prompt exceeds llama.cpp context window"):
        session._resolve_native_generation_budget(
            prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            requested_max_new_tokens=4,
            max_context_tokens=8,
        )


def test_llama_cpp_session_loglikelihood_scores_continuation_tokens() -> None:
    """Verify log-likelihood scoring gathers continuation token log-probabilities from eval logits."""

    session = _build_session()

    outputs = session.loglikelihood(
        [
            LoglikelihoodRequest(
                context="ab",
                continuation="cd",
                metadata={"suite": "demo"},
            )
        ]
    )

    assert session.llm.eval_calls == [[1, ord("a"), ord("b"), ord("c"), ord("d")]]
    assert outputs == [
        LoglikelihoodOutput(
            logprob=-0.5,
            is_greedy=True,
            token_count=2,
            metadata={"suite": "demo"},
        )
    ]
