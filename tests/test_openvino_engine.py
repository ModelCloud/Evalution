# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from evalution.config import Model
from evalution.engines.base import GenerationRequest, LoglikelihoodRequest
from evalution.engines.openvino_engine import (
    OpenVINO,
    OpenVINOSession,
    _import_openvino_optimum,
    load_openvino_runtime,
)

_TINYLLAMA_GPTQ_MODEL = Path("/monster/data/model/TinyLlama-1.1B-Chat-v1.0")


def _openvino_runtime_available() -> bool:
    try:
        return (
            importlib.util.find_spec("optimum.intel.openvino") is not None
            and importlib.util.find_spec("openvino") is not None
        )
    except ModuleNotFoundError:
        return False


def test_openvino_engine_defaults_batch_size_to_auto() -> None:
    engine = OpenVINO()

    assert engine.batch_size == "auto"
    assert engine.seed is None
    assert engine.device is None
    assert engine.ov_config is None
    assert engine.to_dict()["batch_size"] == "auto"
    assert engine.to_dict()["seed"] is None
    assert engine.to_dict()["ov_config"] is None


def test_import_openvino_optimum_requires_optional_dependency(monkeypatch) -> None:
    def fake_import_module(name: str):
        assert name == "optimum.intel.openvino"
        raise ModuleNotFoundError("No module named 'optimum'")

    monkeypatch.setattr(
        "evalution.engines.openvino_engine.importlib.import_module",
        fake_import_module,
    )

    with pytest.raises(
        ModuleNotFoundError,
        match="OpenVINO engine requires the optional `optimum\\[openvino\\]`.*",
    ):
        _import_openvino_optimum()


def test_load_openvino_runtime_seeds_runtime(monkeypatch) -> None:
    class FakeTokenizer:
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeModel:
        def __init__(self) -> None:
            self.config = PretrainedConfig()

        def requires_grad_(self, value: bool):
            return self

        def eval(self):
            return self

    seed_calls: list[int | None] = []

    monkeypatch.setattr(
        "evalution.engines.openvino_engine._seed_transformer_runtime",
        lambda seed: seed_calls.append(seed),
    )
    monkeypatch.setattr(
        "evalution.engines.openvino_engine._load_tokenizer_from_model",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "evalution.engines.openvino_engine._clone_prepare_tokenizer",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "evalution.engines.openvino_engine._import_openvino_optimum",
        lambda: SimpleNamespace(
            OVModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeModel())
        ),
    )

    load_openvino_runtime(
        OpenVINO(device="cuda", seed=2026),
        Model(path="/tmp/model"),
    )

    assert seed_calls == [2026]


def test_load_openvino_runtime_uses_optimum_loader(monkeypatch) -> None:
    class FakeTokenizer:
        pad_token_id = None
        pad_token = None
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeModel:
        def __init__(self) -> None:
            self.config = PretrainedConfig()
            self.seed_calls: list[int] = []
            self.requires_grad_calls: list[bool] = []
            self.eval_calls = 0

        def requires_grad_(self, value: bool):
            self.requires_grad_calls.append(value)
            return self

        def eval(self):
            self.eval_calls += 1
            return self

        def set_seed(self, value: int) -> None:
            self.seed_calls.append(value)

    fake_model = FakeModel()
    load_calls: list[tuple[str, dict[str, object]]] = []

    def fake_from_pretrained(model_path: str, **kwargs):
        load_calls.append((model_path, kwargs))
        return fake_model

    monkeypatch.setattr(
        "evalution.engines.openvino_engine._load_tokenizer_from_model",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "evalution.engines.openvino_engine._clone_prepare_tokenizer",
        lambda **kwargs: "prepare-tokenizer",
    )
    monkeypatch.setattr(
        "evalution.engines.openvino_engine._import_openvino_optimum",
        lambda: SimpleNamespace(
            OVModelForCausalLM=SimpleNamespace(from_pretrained=fake_from_pretrained)
        ),
    )

    runtime = load_openvino_runtime(
        OpenVINO(
            device="cuda",
            ov_config={"PERFORMANCE_HINT": "LATENCY"},
            seed=17,
        ),
        Model(
            path="/tmp/model",
            revision="main",
            trust_remote_code=True,
            model_kwargs={"foo": "bar"},
        ),
    )

    assert load_calls == [
        (
            "/tmp/model",
            {
                "foo": "bar",
                "revision": "main",
                "trust_remote_code": True,
                "device": "cuda",
                "ov_config": {"PERFORMANCE_HINT": "LATENCY"},
            },
        )
    ]
    assert runtime.model is fake_model
    assert runtime.prepare_tokenizer == "prepare-tokenizer"
    assert runtime.input_device.type == "cuda"
    assert fake_model.requires_grad_calls == [False]
    assert fake_model.eval_calls == 1
    assert fake_model.seed_calls == [17]


def test_openvino_session_describes_backend() -> None:
    session = OpenVINOSession(
        config=OpenVINO(),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(dtype="float16"),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="openvino_generate",
    )

    assert session.describe_execution() == {
        "requested_attn_implementation": None,
        "effective_attn_implementation": None,
        "paged_attention": False,
        "generation_backend": "openvino_generate",
        "standard_batch_size_cap": None,
    }


def test_openvino_build_constructs_session(monkeypatch) -> None:
    fake_session = object()

    monkeypatch.setattr(
        OpenVINOSession,
        "from_config",
        classmethod(lambda cls, config, model_config: fake_session),
    )

    engine = OpenVINO(device="cuda")

    assert engine.build(Model(path="/tmp/model")) is fake_session
    assert engine.resolved_engine == "OpenVINO"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _TINYLLAMA_GPTQ_MODEL.exists(),
    reason="local TinyLlama GPTQ weights are not available",
)
@pytest.mark.skipif(
    not _openvino_runtime_available(),
    reason="optimum-intel[openvino] is required for the OpenVINO engine integration test",
)
def test_openvino_engine_can_generate_and_score_on_tinyllama_checkpoint() -> None:
    session = OpenVINO(
        device="cpu",
        batch_size=1,
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
    assert session.input_device.type == "cpu"
    execution = session.describe_execution()
    assert execution["generation_backend"] == "openvino_generate"
