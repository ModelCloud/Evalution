# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evalution.config import Model
from evalution.engines.base import GenerationRequest, LoglikelihoodRequest
from evalution.engines.gptqmodel_engine import (
    GPTQModel,
    GPTQModelSession,
    _import_gptqmodel,
    _validate_gptqmodel_backend,
    load_gptqmodel_runtime,
)

_TINYLLAMA_GPTQ_MODEL = Path("/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit")


def test_gptqmodel_engine_defaults_batch_size_to_auto() -> None:
    engine = GPTQModel()

    assert engine.batch_size == "auto"
    assert engine.backend == "auto"
    assert engine.gptqmodel_path == "/root/gptqmodel"
    assert engine.paged_attention == "auto"
    assert engine.manual_eviction is False
    assert engine.allow_block_sharing is True
    assert engine.use_async_batching is None
    assert engine.q_padding_interval_size == 0
    assert engine.kv_padding_interval_size == 0
    assert engine.max_cached_graphs == 0
    assert engine.to_dict()["batch_size"] == "auto"
    assert engine.to_dict()["backend"] == "auto"
    assert engine.to_dict()["gptqmodel_path"] == "/root/gptqmodel"
    assert engine.to_dict()["paged_attention"] == "auto"
    assert engine.to_dict()["manual_eviction"] is False
    assert engine.to_dict()["allow_block_sharing"] is True
    assert engine.to_dict()["use_async_batching"] is None
    assert engine.to_dict()["q_padding_interval_size"] == 0
    assert engine.to_dict()["kv_padding_interval_size"] == 0
    assert engine.to_dict()["max_cached_graphs"] == 0


def test_gptqmodel_session_describes_quantized_backend() -> None:
    session = GPTQModelSession(
        config=GPTQModel(),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(dtype="float16"),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cuda"),
        generation_backend="gptqmodel_generate",
        resolved_backend="gptq_triton",
        quant_method="gptq",
        runtime_format="gptq",
    )

    assert session.describe_execution() == {
        "requested_attn_implementation": None,
        "effective_attn_implementation": None,
        "paged_attention": False,
        "generation_backend": "gptqmodel_generate",
        "standard_batch_size_cap": None,
        "quantized_backend": "gptq_triton",
        "quant_method": "gptq",
        "runtime_format": "gptq",
    }


def test_gptqmodel_engine_rejects_external_backends() -> None:
    with pytest.raises(ValueError, match="only supports native GPTQModel/HF-style backends"):
        _validate_gptqmodel_backend("vllm")


def test_import_gptqmodel_uses_checkout_fallback(monkeypatch, tmp_path) -> None:
    fake_module = object()

    def fake_import_module(name: str):
        assert name == "gptqmodel"
        if str(tmp_path) not in sys.path:
            raise ModuleNotFoundError("No module named 'gptqmodel'")
        return fake_module

    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine.importlib.import_module",
        fake_import_module,
    )
    monkeypatch.setattr("evalution.engines.gptqmodel_engine.sys.path", list(sys.path))

    imported = _import_gptqmodel(str(tmp_path))

    assert imported is fake_module
    assert sys.path[0] == str(tmp_path)


def test_load_gptqmodel_runtime_uses_quantized_loader(monkeypatch) -> None:
    import transformers

    class FakeTokenizer:
        pad_token_id = None
        pad_token = None
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeInnerModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(_attn_implementation="flash_attention_2")

        def modules(self):
            yield SimpleNamespace(backend=SimpleNamespace(value="gptq_triton"))

    class FakeWrapper:
        def __init__(self) -> None:
            self.model = FakeInnerModel()
            self.quantize_config = SimpleNamespace(
                method=SimpleNamespace(value="gptq"),
                runtime_format=SimpleNamespace(value="gptq"),
            )
            self.eval_called = False
            self.requires_grad_value: bool | None = None

        def requires_grad_(self, value: bool):
            self.requires_grad_value = value
            return self

        def eval(self):
            self.eval_called = True
            return self

    fake_wrapper = FakeWrapper()

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._clone_prepare_tokenizer",
        lambda **kwargs: None,
    )

    load_calls: list[dict[str, object]] = []

    def fake_load(**kwargs):
        load_calls.append(kwargs)
        return fake_wrapper

    fake_module = SimpleNamespace(GPTQModel=SimpleNamespace(load=fake_load))
    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._import_gptqmodel",
        lambda path: fake_module,
    )

    runtime = load_gptqmodel_runtime(
        GPTQModel(device="cpu", dtype="bfloat16", backend="auto"),
        Model(path="/tmp/model"),
    )

    assert load_calls == [
        {
            "model_id_or_path": "/tmp/model",
            "device_map": None,
            "device": "cpu",
            "backend": "auto",
            "trust_remote_code": False,
            "revision": None,
            "dtype": torch.bfloat16,
        }
    ]
    assert fake_wrapper.requires_grad_value is False
    assert fake_wrapper.eval_called is True
    assert runtime.model is fake_wrapper.model
    assert runtime.input_device.type == "cpu"
    assert runtime.requested_attn_implementation == "flash_attention_2"
    assert runtime.resolved_backend == "gptq_triton"
    assert runtime.quant_method == "gptq"
    assert runtime.runtime_format == "gptq"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _TINYLLAMA_GPTQ_MODEL.exists(),
    reason="local TinyLlama GPTQ weights are not available",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the GPTQModel engine integration test",
)
def test_gptqmodel_engine_can_generate_and_score_on_cuda() -> None:
    session = GPTQModel(
        device="cuda:0",
        batch_size=1,
        attn_implementation="flash_attention_2",
        paged_attention=True,
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
    assert session.input_device.type == "cuda"
    execution = session.describe_execution()
    assert execution["generation_backend"] == "continuous_batching"
    assert execution["effective_attn_implementation"] == "paged|flash_attention_2"
    assert execution["paged_attention"] is True
    assert execution["quant_method"] == "gptq"
    assert execution["runtime_format"] is not None
    assert execution["quantized_backend"] is not None
