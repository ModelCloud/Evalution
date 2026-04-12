# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evalution.config import Model
from evalution.engines.base import GenerationRequest, LoglikelihoodRequest
from evalution.engines.tensorrt_llm_engine import (
    TensorRTLLM,
    TensorRTLLMSession,
    _import_tensorrt_llm,
)

# Keep integration tests anchored to the shared local model store used elsewhere in this repo.
_LOCAL_MODEL_ROOT = Path("/monster/data/model")
# Prefer the smallest local checkpoints so the real runtime smoke test stays fast and predictable.
_LOCAL_TENSORRT_LLM_MODEL_CANDIDATES = (
    "TinyLlama-1.1B-Chat-v1.0",
    "Llama-3.2-1B-Instruct",
)
_HAS_TENSORRT_LLM_RUNTIME = importlib.util.find_spec("tensorrt_llm") is not None


def _local_tensorrt_llm_model() -> Path:
    """Return a known local CUDA smoke-test model and fail loudly when none is available."""

    for candidate in _LOCAL_TENSORRT_LLM_MODEL_CANDIDATES:
        path = _LOCAL_MODEL_ROOT / candidate
        if path.exists():
            return path
    pytest.fail(
        f"missing TensorRT-LLM smoke-test model under {_LOCAL_MODEL_ROOT}; "
        f"looked for {', '.join(_LOCAL_TENSORRT_LLM_MODEL_CANDIDATES)}"
    )


def _integration_log(capfd: pytest.CaptureFixture[str], message: str) -> None:
    """Emit a progress line directly to the terminal so pytest capture does not hide it."""

    with capfd.disabled():
        print(f"[test_tensorrt_llm_engine] {message}", flush=True)


class FakeLogprob:
    """Minimal prompt-logprob record that matches the engine's scoring expectations."""

    def __init__(self, logprob: float, rank: int | None = None) -> None:
        self.logprob = logprob
        self.rank = rank


class FakeSamplingParams:
    """Small stand-in for TensorRT-LLM SamplingParams that preserves passed kwargs."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeTokenizer:
    """Deterministic tokenizer stub used to keep prompt rendering and scoring tests simple."""

    # Expose common special-token ids so synthetic-prefix scoring paths behave like a real tokenizer.
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    model_max_length = 32

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(token_id) for token_id in ids)


class FakeLLM:
    """Blocking TensorRT-LLM stand-in that records generation calls for assertions."""

    def __init__(self, tokenizer=None) -> None:
        # Mimic the small runtime surface the engine inspects for metadata and lifecycle hooks.
        self._tokenizer = tokenizer or FakeTokenizer()
        self.args = SimpleNamespace(backend="trt")
        self.generate_calls: list[tuple[list[object], list[FakeSamplingParams]]] = []
        self.reset_prefix_cache_calls = 0
        self.shutdown_calls = 0

    def get_tokenizer(self):
        return self._tokenizer

    def generate(self, prompts, sampling_params, use_tqdm=False):
        del use_tqdm
        prompts = list(prompts)
        sampling_params = list(sampling_params)
        self.generate_calls.append((prompts, sampling_params))
        results = []
        for prompt, params in zip(prompts, sampling_params, strict=True):
            if isinstance(prompt, dict):
                prompt_token_ids = list(prompt["prompt_token_ids"])
                prompt_text = prompt.get("prompt", "")
            else:
                prompt_text = str(prompt)
                prompt_token_ids = self._tokenizer.encode(prompt_text, add_special_tokens=False)
            if params.kwargs.get("prompt_logprobs"):
                prompt_logprobs = [None]
                for token_id in prompt_token_ids[1:]:
                    prompt_logprobs.append(
                        {
                            token_id: FakeLogprob(-0.25, rank=1),
                            token_id + 1: FakeLogprob(-1.50, rank=2),
                        }
                    )
                results.append(
                    SimpleNamespace(
                        prompt=prompt_text,
                        prompt_token_ids=prompt_token_ids,
                        prompt_logprobs=prompt_logprobs,
                        outputs=[],
                    )
                )
                continue

            results.append(
                SimpleNamespace(
                    prompt=prompt_text,
                    prompt_token_ids=prompt_token_ids,
                    prompt_logprobs=None,
                    outputs=[
                        SimpleNamespace(
                            text="AB stop here",
                            token_ids=[65, 66, 67],
                            finish_reason="stop",
                            stop_reason="stop",
                        )
                    ],
                )
            )
        return results

    def reset_prefix_cache(self):
        self.reset_prefix_cache_calls += 1
        return True

    def shutdown(self):
        self.shutdown_calls += 1


class FakeContinuousEngine:
    """Request-level scheduler stub that can complete prompts out of submission order."""

    def __init__(self, *, prompt_delays: dict[str, int]) -> None:
        # Track in-flight prompts so continuous batching assertions can verify refill behavior.
        self.prompt_delays = dict(prompt_delays)
        self.inflight: dict[str, dict[str, object]] = {}
        self.max_inflight = 0
        self.abort_calls: list[tuple[list[str], bool]] = []

    def add_request(self, request_id, prompt, params, prompt_text=None, **kwargs):
        del params, kwargs
        prompt_token_ids = list(prompt["prompt_token_ids"])
        rendered_prompt = prompt_text or prompt.get("prompt") or ""
        self.inflight[request_id] = {
            "prompt": rendered_prompt,
            "prompt_token_ids": prompt_token_ids,
            "remaining_steps": self.prompt_delays.get(rendered_prompt, 1),
        }
        self.max_inflight = max(self.max_inflight, len(self.inflight))
        return request_id

    def step(self):
        finished = []
        for request_id, state in list(self.inflight.items()):
            state["remaining_steps"] = int(state["remaining_steps"]) - 1
            if int(state["remaining_steps"]) > 0:
                continue
            del self.inflight[request_id]
            prompt_text = str(state["prompt"])
            finished.append(
                SimpleNamespace(
                    request_id=request_id,
                    finished=True,
                    prompt_token_ids=list(state["prompt_token_ids"]),
                    outputs=[
                        SimpleNamespace(
                            text=f"{prompt_text}-done",
                            token_ids=[ord(char) for char in f"{prompt_text}!"],
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ],
                )
            )
        return sorted(finished, key=lambda output: output.request_id, reverse=True)

    def has_unfinished_requests(self):
        return bool(self.inflight)

    def abort_request(self, request_ids, internal=False):
        request_ids = list(request_ids)
        self.abort_calls.append((request_ids, internal))
        for request_id in request_ids:
            self.inflight.pop(request_id, None)

    def shutdown(self):
        self.inflight.clear()


class FakeContinuousLLM(FakeLLM):
    """Fake LLM whose runtime exposes the request-level scheduler used by continuous tests."""

    def __init__(self, *, prompt_delays: dict[str, int], tokenizer=None) -> None:
        super().__init__(tokenizer=tokenizer)
        self.llm_engine = FakeContinuousEngine(prompt_delays=prompt_delays)


def test_tensorrt_llm_engine_defaults_batch_size_to_auto() -> None:
    engine = TensorRTLLM()

    assert engine.batch_size == "auto"
    assert engine.padding_side == "left"
    assert engine.tensor_parallel_size == 1
    assert engine.runtime_backend is None
    assert engine.tensorrt_llm_path is None
    assert engine.to_dict()["resolved_engine"] is None


def test_import_tensorrt_llm_uses_checkout_fallback(monkeypatch, tmp_path) -> None:
    fake_module = object()

    def fake_import_module(name: str):
        assert name == "tensorrt_llm"
        if str(tmp_path) not in sys.path:
            raise ModuleNotFoundError("No module named 'tensorrt_llm'")
        return fake_module

    monkeypatch.setattr(
        "evalution.engines.tensorrt_llm_engine.importlib.import_module",
        fake_import_module,
    )
    monkeypatch.setattr("evalution.engines.tensorrt_llm_engine.sys.path", list(sys.path))

    imported = _import_tensorrt_llm(str(tmp_path))

    assert imported is fake_module
    assert sys.path[0] == str(tmp_path)


def test_tensorrt_llm_session_generates_and_scores() -> None:
    llm = FakeLLM()
    session = TensorRTLLMSession(
        config=TensorRTLLM(batch_size=2),
        model_config=Model(path="/tmp/model"),
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        prepare_tokenizer=llm.get_tokenizer(),
        sampling_params_cls=FakeSamplingParams,
    )

    generation_outputs = session.generate(
        [
            GenerationRequest(prompt="Hello", max_new_tokens=2, stop=[" stop"]),
            GenerationRequest(messages=[{"role": "user", "content": "Hi"}], max_new_tokens=2),
        ]
    )
    scores = session.loglikelihood(
        [
            LoglikelihoodRequest(context="A", continuation="B"),
            LoglikelihoodRequest(context="", continuation="C"),
        ]
    )

    assert [output.text for output in generation_outputs] == ["AB", "AB stop here"]
    assert generation_outputs[0].prompt == "Hello"
    assert generation_outputs[1].prompt == "user: Hi"
    assert scores[0].logprob == pytest.approx(-0.25)
    assert scores[0].is_greedy is True
    assert scores[0].token_count == 1
    assert scores[1].logprob == pytest.approx(-0.25)
    assert len(llm.generate_calls) == 2


def test_tensorrt_llm_generate_continuous_refills_request_slots() -> None:
    llm = FakeContinuousLLM(prompt_delays={"A": 2, "B": 1, "C": 1})
    session = TensorRTLLMSession(
        config=TensorRTLLM(batch_size=2),
        model_config=Model(path="/tmp/model"),
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        prepare_tokenizer=llm.get_tokenizer(),
        sampling_params_cls=FakeSamplingParams,
    )

    outputs = list(
        session.generate_continuous(
            [
                (10, GenerationRequest(prompt="A", max_new_tokens=2)),
                (11, GenerationRequest(prompt="B", max_new_tokens=2)),
                (12, GenerationRequest(prompt="C", max_new_tokens=2)),
            ]
        )
    )

    assert [item_id for item_id, _output in outputs] == [11, 12, 10]
    assert llm.llm_engine.max_inflight == 2
    assert [output.text for _item_id, output in outputs] == ["B-done", "C-done", "A-done"]


def test_tensorrt_llm_session_gc_and_close_release_runtime_state() -> None:
    llm = FakeLLM()
    session = TensorRTLLMSession(
        config=TensorRTLLM(batch_size=2),
        model_config=Model(path="/tmp/model"),
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        prepare_tokenizer=llm.get_tokenizer(),
        sampling_params_cls=FakeSamplingParams,
    )

    session.gc()
    session.close()

    assert llm.reset_prefix_cache_calls == 1
    assert llm.shutdown_calls == 1


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _HAS_TENSORRT_LLM_RUNTIME,
    reason="tensorrt_llm is required for the TensorRT-LLM engine integration test",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the TensorRT-LLM engine integration test",
)
def test_tensorrt_llm_engine_runs_real_cuda_smoke_path(capfd: pytest.CaptureFixture[str]) -> None:
    """Run the real TensorRT-LLM CUDA path and assert the current scoring limitation explicitly."""

    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    model_path = _local_tensorrt_llm_model()

    _integration_log(capfd, f"checking TensorRT-LLM runtime for model {model_path}")
    try:
        _import_tensorrt_llm(None)
    except Exception as exc:  # pragma: no cover - runtime availability is environment specific
        pytest.fail(f"TensorRT-LLM runtime unavailable: {exc}")

    _integration_log(capfd, "building TensorRT-LLM session")
    session = TensorRTLLM(
        batch_size=1,
        tensor_parallel_size=1,
        runtime_backend="pytorch",
        llm_kwargs={
            "gpus_per_node": 1,
            "attn_backend": "TRTLLM",
            "max_batch_size": 1,
            "max_input_len": 256,
            "max_num_tokens": 512,
            "max_seq_len": 512,
            "kv_cache_config": KvCacheConfig(
                max_tokens=2048,
                free_gpu_memory_fraction=0.05,
            ),
            "env_overrides": {
                "TLLM_WORKER_USE_SINGLE_PROCESS": "1",
                "TRTLLM_ENABLE_PDL": "0",
            },
        },
    ).build(Model(path=str(model_path)))

    try:
        _integration_log(capfd, "running generation smoke step")
        outputs = session.generate(
            [
                GenerationRequest(
                    prompt="The capital of France is",
                    max_new_tokens=8,
                )
            ],
            batch_size=1,
        )
        _integration_log(capfd, "running loglikelihood smoke step")
        with pytest.raises(
                RuntimeError,
                match="did not return prompt_logprobs",
        ):
            session.loglikelihood(
                [
                    LoglikelihoodRequest(
                        context="The capital of France is",
                        continuation=" Paris",
                    )
                ],
                batch_size=1,
            )
        _integration_log(capfd, "collecting execution metadata")
        execution = session.describe_execution()
    finally:
        _integration_log(capfd, "closing TensorRT-LLM session")
        session.close()

    print(f"outputs={outputs}")
    assert len(outputs) == 1
    assert outputs[0].prompt == "The capital of France is"
    assert isinstance(outputs[0].text, str)
    assert outputs[0].text
    assert outputs[0].metadata["completion_token_count"] > 0
    assert execution["tensor_parallel_size"] == 1
    assert execution["runtime_backend"] == "pytorch"
    assert execution["max_model_len"] == 512
