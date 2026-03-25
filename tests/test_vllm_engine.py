# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evalution.config import Model
from evalution.engines.base import GenerationRequest, LoglikelihoodRequest, RollingLoglikelihoodRequest
from evalution.engines.vllm_engine import VLLM, VLLMSession, _import_vllm

_TINYLLAMA_GPTQ_MODEL = Path("/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit")


class FakeLogprob:
    def __init__(self, logprob: float, rank: int | None = None) -> None:
        self.logprob = logprob
        self.rank = rank


class FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeTokenizer:
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
    def __init__(self, tokenizer=None) -> None:
        self._tokenizer = tokenizer or FakeTokenizer()
        self.generate_calls: list[tuple[list[dict[str, object]], list[FakeSamplingParams]]] = []
        self.reset_prefix_cache_calls = 0
        self.llm_engine = SimpleNamespace(
            shutdown=lambda: None,
            vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=32)),
        )

    def get_tokenizer(self):
        return self._tokenizer

    def generate(self, prompts, sampling_params, use_tqdm=False):
        del use_tqdm
        prompts = list(prompts)
        sampling_params = list(sampling_params)
        self.generate_calls.append((prompts, sampling_params))
        results = []
        for prompt, params in zip(prompts, sampling_params, strict=True):
            prompt_token_ids = list(prompt["prompt_token_ids"])
            prompt_text = prompt.get("prompt")
            if params.kwargs.get("max_tokens") == 0:
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

            generated_tokens = [65, 66]
            results.append(
                SimpleNamespace(
                    prompt=prompt_text,
                    prompt_token_ids=prompt_token_ids,
                    prompt_logprobs=None,
                    outputs=[
                        SimpleNamespace(
                            text="AB",
                            token_ids=generated_tokens,
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ],
                )
            )
        return results

    def reset_prefix_cache(self):
        self.reset_prefix_cache_calls += 1
        return True


def test_vllm_engine_defaults_batch_size_to_auto() -> None:
    engine = VLLM()

    assert engine.batch_size == "auto"
    assert engine.padding_side == "left"
    assert engine.tensor_parallel_size == 1
    assert engine.gpu_memory_utilization == 0.9
    assert engine.vllm_path is None
    assert engine.to_dict()["resolved_engine"] is None


def test_import_vllm_uses_checkout_fallback(monkeypatch, tmp_path) -> None:
    fake_module = object()

    def fake_import_module(name: str):
        assert name == "vllm"
        if str(tmp_path) not in sys.path:
            raise ModuleNotFoundError("No module named 'vllm'")
        return fake_module

    monkeypatch.setattr("evalution.engines.vllm_engine.importlib.import_module", fake_import_module)
    monkeypatch.setattr("evalution.engines.vllm_engine.sys.path", list(sys.path))

    imported = _import_vllm(str(tmp_path))

    assert imported is fake_module
    assert sys.path[0] == str(tmp_path)


def test_vllm_session_generates_and_scores() -> None:
    llm = FakeLLM()
    session = VLLMSession(
        config=VLLM(batch_size=2),
        model_config=Model(path="/tmp/model"),
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        prepare_tokenizer=llm.get_tokenizer(),
        sampling_params_cls=FakeSamplingParams,
    )

    generation_outputs = session.generate(
        [
            GenerationRequest(prompt="Hello", max_new_tokens=2),
            GenerationRequest(
                messages=[{"role": "user", "content": "Hi"}],
                max_new_tokens=2,
            ),
        ]
    )
    scores = session.loglikelihood(
        [
            LoglikelihoodRequest(context="A", continuation="B"),
            LoglikelihoodRequest(context="", continuation="C"),
        ]
    )

    assert [output.text for output in generation_outputs] == ["AB", "AB"]
    assert generation_outputs[0].prompt == "Hello"
    assert generation_outputs[1].prompt == "user: Hi"
    assert scores[0].logprob == pytest.approx(-0.25)
    assert scores[0].is_greedy is True
    assert scores[0].token_count == 1
    assert scores[1].logprob == pytest.approx(-0.25)
    assert len(llm.generate_calls) == 2
    assert llm.generate_calls[1][1][0].prompt_logprobs == 1


def test_vllm_session_scores_rolling_requests() -> None:
    llm = FakeLLM()
    session = VLLMSession(
        config=VLLM(batch_size=4, max_model_len=4),
        model_config=Model(path="/tmp/model"),
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        prepare_tokenizer=llm.get_tokenizer(),
        sampling_params_cls=FakeSamplingParams,
    )

    outputs = session.loglikelihood_rolling(
        [RollingLoglikelihoodRequest(text="ABCD")],
        batch_size=2,
    )

    assert len(outputs) == 1
    assert outputs[0].token_count == 4
    assert outputs[0].logprob == pytest.approx(-1.0)


def test_vllm_session_rejects_beam_search_requests() -> None:
    llm = FakeLLM()
    session = VLLMSession(
        config=VLLM(),
        model_config=Model(path="/tmp/model"),
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        prepare_tokenizer=llm.get_tokenizer(),
        sampling_params_cls=FakeSamplingParams,
    )

    with pytest.raises(ValueError, match="num_beams=1"):
        session.generate([GenerationRequest(prompt="Hello", num_beams=2)])


def test_vllm_session_gc_resets_prefix_cache() -> None:
    llm = FakeLLM()
    session = VLLMSession(
        config=VLLM(),
        model_config=Model(path="/tmp/model"),
        llm=llm,
        tokenizer=llm.get_tokenizer(),
        prepare_tokenizer=llm.get_tokenizer(),
        sampling_params_cls=FakeSamplingParams,
    )

    session.gc()

    assert llm.reset_prefix_cache_calls == 1


def test_vllm_build_loads_local_checkout(monkeypatch) -> None:
    fake_llm = FakeLLM()

    class FakeVLLMModule:
        SamplingParams = FakeSamplingParams

        @staticmethod
        def LLM(**kwargs):
            assert kwargs["model"] == "/tmp/model"
            assert kwargs["tokenizer"] == "/tmp/model"
            assert kwargs["tensor_parallel_size"] == 2
            return fake_llm

    monkeypatch.setattr("evalution.engines.vllm_engine._import_vllm", lambda path: FakeVLLMModule)

    session = VLLM(tensor_parallel_size=2).build(Model(path="/tmp/model"))

    assert isinstance(session, VLLMSession)
    assert session.describe_execution()["generation_backend"] == "vllm_generate"


def test_vllm_build_accepts_custom_tokenizer_object(monkeypatch) -> None:
    fake_llm = FakeLLM()
    custom_tokenizer = FakeTokenizer()

    class FakeVLLMModule:
        SamplingParams = FakeSamplingParams

        @staticmethod
        def LLM(**kwargs):
            assert kwargs["tokenizer"] == "/tmp/model"
            return fake_llm

    monkeypatch.setattr("evalution.engines.vllm_engine._import_vllm", lambda path: FakeVLLMModule)
    monkeypatch.setattr(
        "evalution.engines.vllm_engine._load_tokenizer_from_model",
        lambda source, **kwargs: custom_tokenizer,
    )

    session = VLLM().build(Model(path="/tmp/model", tokenizer=custom_tokenizer))

    assert session.prepare_tokenizer is custom_tokenizer


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _TINYLLAMA_GPTQ_MODEL.exists(),
    reason="local TinyLlama GPTQ weights are not available",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the vLLM engine integration test",
)
@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="vLLM is required for the vLLM engine integration test",
)
def test_vllm_engine_can_generate_and_score_on_cuda() -> None:
    try:
        import vllm._C  # noqa: F401
    except Exception as exc:  # pragma: no cover - runtime availability is environment specific
        pytest.skip(f"vllm runtime unavailable: {exc}")

    session = VLLM(
        batch_size=1,
        gpu_memory_utilization=0.8,
        quantization="gptq",
    ).build(Model(path=str(_TINYLLAMA_GPTQ_MODEL)))

    try:
        outputs = session.generate(
            [
                GenerationRequest(
                    prompt="The capital of France is",
                    max_new_tokens=8,
                )
            ],
            batch_size=1,
        )
        scores = session.loglikelihood(
            [
                LoglikelihoodRequest(
                    context="The capital of France is",
                    continuation=" Paris",
                )
            ],
            batch_size=1,
        )
    finally:
        session.close()

    assert len(outputs) == 1
    assert outputs[0].prompt == "The capital of France is"
    assert isinstance(outputs[0].text, str)
    assert len(scores) == 1
    assert scores[0].token_count > 0
    assert math.isfinite(scores[0].logprob)
    execution = session.describe_execution()
    assert execution["generation_backend"] == "vllm_generate"
    assert execution["tensor_parallel_size"] == 1
    assert execution["gpu_memory_utilization"] == pytest.approx(0.8)
    assert execution["quantization"] == "gptq"
    assert execution["max_model_len"] is not None
