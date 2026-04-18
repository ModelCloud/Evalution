# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers import PretrainedConfig

from evalution.config import Model
from evalution.engines.base import GenerationOutput, GenerationRequest, LoglikelihoodRequest
from evalution.engines.gptqmodel_engine import (
    GPTQModel,
    GPTQModelSession,
    _CUDA_CUSPARSE_HEADER,
    _import_gptqmodel,
    _validate_gptqmodel_backend,
    load_gptqmodel_runtime,
)

# Keep shared test fixtures and expectations explicit at module scope.
_TINYLLAMA_GPTQ_MODEL = Path("/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit")


def test_gptqmodel_engine_defaults_batch_size_to_auto() -> None:
    """Verify gptqmodel engine defaults batch size to auto."""
    engine = GPTQModel()

    assert engine.batch_size == "auto"
    assert engine.seed is None
    assert engine.backend == "auto"
    assert engine.manual_eviction is False
    assert engine.allow_block_sharing is True
    assert engine.max_blocks_per_request is None
    assert engine.use_async_batching is None
    assert engine.use_cuda_graph is None
    assert engine.q_padding_interval_size == 0
    assert engine.kv_padding_interval_size == 0
    assert engine.max_cached_graphs == 0
    assert engine.to_dict()["batch_size"] == "auto"
    assert engine.to_dict()["seed"] is None
    assert engine.to_dict()["backend"] == "auto"
    assert engine.to_dict()["manual_eviction"] is False
    assert engine.to_dict()["allow_block_sharing"] is True
    assert engine.to_dict()["max_blocks_per_request"] is None
    assert engine.to_dict()["use_async_batching"] is None
    assert engine.to_dict()["use_cuda_graph"] is None
    assert engine.to_dict()["q_padding_interval_size"] == 0
    assert engine.to_dict()["kv_padding_interval_size"] == 0
    assert engine.to_dict()["max_cached_graphs"] == 0

def test_load_gptqmodel_runtime_seeds_runtime(monkeypatch) -> None:
    """Verify load gptqmodel runtime seeds runtime."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeInnerModel:
        """Provide the fake inner model helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        config = SimpleNamespace()

    class FakeWrapper:
        """Provide the fake wrapper helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.model = FakeInnerModel()

        def requires_grad_(self, value: bool):
            """Implement requires grad for fake wrapper."""
            return self

        def eval(self):
            """Implement eval for fake wrapper."""
            return self

    seed_calls: list[int | None] = []

    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._seed_transformer_runtime",
        lambda seed: seed_calls.append(seed),
    )
    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._load_tokenizer_from_model",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._clone_prepare_tokenizer",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._import_gptqmodel",
        lambda: SimpleNamespace(
            GPTQModel=SimpleNamespace(load=lambda **kwargs: FakeWrapper())
        ),
    )

    load_gptqmodel_runtime(
        GPTQModel(device="cpu", seed=4321),
        Model(path="/tmp/model"),
    )

    assert seed_calls == [4321]


def test_gptqmodel_session_describes_quantized_backend() -> None:
    """Verify gptqmodel session describes quantized backend."""
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


def test_gptqmodel_session_generate_continuous_refills_paged_manager_while_caller_is_paused(
    monkeypatch,
) -> None:
    """Verify gptqmodel session generate continuous refills paged manager while caller is paused. Preserve the fallback order expected by the surrounding caller."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, prompts, *, add_special_tokens=False, **kwargs):
            """Implement call for fake tokenizer."""
            assert add_special_tokens is False
            del kwargs
            if isinstance(prompts, str):
                return {
                    "input_ids": torch.tensor([[11, 12, 13]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }
            batch_size = len(prompts)
            return {
                "input_ids": torch.tensor([[11, 12, 13] for _ in range(batch_size)]),
                "attention_mask": torch.tensor([[1, 1, 1] for _ in range(batch_size)]),
            }

        def decode(self, token_ids, *, skip_special_tokens=False):
            """Implement decode for fake tokenizer."""
            del token_ids, skip_special_tokens
            return "The answer is 42."

    class FakeContinuousOutput:
        """Provide the fake continuous output helper used by the surrounding tests."""
        def __init__(self, request_id, tokens):
            """Initialize this object."""
            self.request_id = request_id
            self.generated_tokens = tokens
            self.error = None

        def is_finished(self) -> bool:
            """Implement is finished for fake continuous output."""
            return True

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = PretrainedConfig()
            self.config._attn_implementation = "flash_attention_2"
            self.device = "cuda"

        def eval(self):
            """Implement eval for fake model."""
            return self

        def _get_logits_processor(self, generation_config):
            """Get logits processor."""
            del generation_config
            return transformers.LogitsProcessorList()
        def set_attn_implementation(self, value: str) -> None:
            """Implement set attn implementation for fake model."""
            self.config._attn_implementation = value

    class FakeContinuousBatchingManager:
        """Provide the fake continuous batching manager helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        last_instance: FakeContinuousBatchingManager | None = None

        def __init__(self, model, generation_config, **kwargs):
            """Initialize this object."""
            del model, generation_config, kwargs
            self.started = False
            self.stopped = False
            self.stop_calls = 0
            self.result_queue: list[FakeContinuousOutput] = []
            self.added_request_ids: list[str] = []
            self.third_request_added = threading.Event()
            FakeContinuousBatchingManager.last_instance = self

        def start(self) -> None:
            """Implement start for fake continuous batching manager."""
            self.started = True

        def add_request(self, input_ids, *, request_id=None, max_new_tokens=None, streaming=False):
            """Implement add request for fake continuous batching manager."""
            del input_ids, max_new_tokens, streaming
            assert request_id is not None
            self.added_request_ids.append(request_id)
            if len(self.added_request_ids) == 2:
                self.result_queue.append(FakeContinuousOutput("req_0", [101, 102, 103]))
            elif len(self.added_request_ids) == 3:
                self.third_request_added.set()

        def get_result(self, timeout=None):
            """Get result."""
            if not self.result_queue:
                if timeout:
                    threading.Event().wait(min(timeout, 0.01))
                return None
            return self.result_queue.pop(0)

        def is_running(self) -> bool:
            """Implement is running for fake continuous batching manager."""
            return self.started and not self.stopped

        def stop(self, block=True):
            """Implement stop for fake continuous batching manager."""
            assert block is True
            self.stopped = True
            self.stop_calls += 1

    monkeypatch.setattr(
        transformers,
        "ContinuousBatchingManager",
        FakeContinuousBatchingManager,
    )

    session = GPTQModelSession(
        config=GPTQModel(attn_implementation="paged|flash_attention_2"),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
        resolved_backend="gptq_triton",
        quant_method="gptq",
        runtime_format="gptq",
    )

    iterator = session.generate_continuous(
        [
            (10, GenerationRequest(prompt="Q: 40 + 2\nA:", stop=["Q:"])),
            (11, GenerationRequest(prompt="Q: 41 + 2\nA:", stop=["Q:"])),
            (12, GenerationRequest(prompt="Q: 42 + 2\nA:", stop=["Q:"])),
        ],
        batch_size=2,
    )

    first_output = next(iterator)
    manager = FakeContinuousBatchingManager.last_instance

    assert first_output == (
        10,
        GenerationOutput(prompt="Q: 40 + 2\nA:", text="The answer is 42.", metadata={}),
    )
    assert manager is not None
    assert manager.third_request_added.wait(timeout=1.0)
    assert manager.added_request_ids == ["req_0", "req_1", "req_2"]
    assert session.continuous_batching_manager is manager

    iterator.close()

    assert manager.stopped is True
    assert manager.stop_calls == 1
    assert session.continuous_batching_manager is None


def test_gptqmodel_engine_rejects_external_backends() -> None:
    """Verify gptqmodel engine rejects external backends."""
    with pytest.raises(ValueError, match="only supports native GPTQModel/HF-style backends"):
        _validate_gptqmodel_backend("vllm")


def test_gptqmodel_rejects_paged_attn_when_continuous_batching_is_unavailable(monkeypatch) -> None:
    """Verify gptqmodel rejects paged attn when continuous batching is unavailable."""
    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine.transformers_continuous_batching_support",
        lambda: (False, "transformers 4.55.4 is older than 4.56.0"),
    )

    with pytest.raises(
        ValueError,
        match="paged attn_implementation requires a transformers build with continuous batching support",
    ):
        GPTQModel(attn_implementation="paged|flash_attention_2").build(Model(path="/tmp/model"))


def test_import_gptqmodel_uses_native_python_import(monkeypatch) -> None:
    """Verify import gptqmodel reads from the active Python environment only."""
    fake_module = object()

    def fake_import_module(name: str):
        """Support the surrounding tests with fake import module."""
        assert name == "gptqmodel"
        return fake_module

    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine.importlib.import_module",
        fake_import_module,
    )
    imported = _import_gptqmodel()

    assert imported is fake_module


def test_load_gptqmodel_runtime_uses_quantized_loader(monkeypatch) -> None:
    """Verify load gptqmodel runtime uses quantized loader."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = None
        pad_token = None
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeInnerModel:
        """Provide the fake inner model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = SimpleNamespace(_attn_implementation="flash_attention_2")

        def modules(self):
            """Implement modules for fake inner model."""
            yield SimpleNamespace(backend=SimpleNamespace(value="gptq_triton"))

    class FakeWrapper:
        """Provide the fake wrapper helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.model = FakeInnerModel()
            self.quantize_config = SimpleNamespace(
                method=SimpleNamespace(value="gptq"),
                runtime_format=SimpleNamespace(value="gptq"),
            )
            self.eval_called = False
            self.requires_grad_value: bool | None = None

        def requires_grad_(self, value: bool):
            """Implement requires grad for fake wrapper."""
            self.requires_grad_value = value
            return self

        def eval(self):
            """Implement eval for fake wrapper."""
            self.eval_called = True
            return self

    fake_wrapper = FakeWrapper()

    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._load_tokenizer_from_model",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._clone_prepare_tokenizer",
        lambda **kwargs: None,
    )

    load_calls: list[dict[str, object]] = []

    def fake_load(**kwargs):
        """Support the surrounding tests with fake load."""
        load_calls.append(kwargs)
        return fake_wrapper

    fake_module = SimpleNamespace(GPTQModel=SimpleNamespace(load=fake_load))
    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._import_gptqmodel",
        lambda: fake_module,
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


def test_load_gptqmodel_runtime_prefers_triton_when_marlin_headers_are_missing(monkeypatch) -> None:
    """Force GPTQModel auto backend away from Marlin when CUDA developer headers are absent."""

    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = None
        pad_token = None
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeInnerModel:
        """Provide the fake inner model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = SimpleNamespace(_attn_implementation="flash_attention_2")

        def modules(self):
            """Implement modules for fake inner model."""
            return iter(())

    class FakeWrapper:
        """Provide the fake wrapper helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.model = FakeInnerModel()
            self.quantize_config = None

        def requires_grad_(self, value: bool):
            """Implement requires grad for fake wrapper."""
            del value
            return self

        def eval(self):
            """Implement eval for fake wrapper."""
            return self

    load_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._load_tokenizer_from_model",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._clone_prepare_tokenizer",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "evalution.engines.gptqmodel_engine._import_gptqmodel",
        lambda: SimpleNamespace(
            GPTQModel=SimpleNamespace(
                load=lambda **kwargs: load_calls.append(kwargs) or FakeWrapper()
            )
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(type(_CUDA_CUSPARSE_HEADER), "exists", lambda self: False)

    load_gptqmodel_runtime(
        GPTQModel(device="cuda:0", backend="auto"),
        Model(path="/tmp/model"),
    )

    assert load_calls[0]["backend"] == "gptq_triton"


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
    """Verify gptqmodel engine can generate and score on cuda. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    session = GPTQModel(
        device="cuda:0",
        batch_size=1,
        attn_implementation="paged|flash_attention_2",
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
