# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import threading
import sys
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers import PretrainedConfig

from evalution.config import Model
from evalution.engines.base import (
    GenerationOutput,
    GenerationRequest,
    LoglikelihoodOutput,
    LoglikelihoodRequest,
    RollingLoglikelihoodRequest,
)
from evalution.engines.transformers_common import _resolve_input_device, _seed_transformer_runtime
from evalution.engines.transformers import Transformers, TransformersSession
from evalution.engines.transformers_compat import TransformersCompat


def test_transformer_defaults_batch_size_to_auto() -> None:
    """Verify transformer defaults batch size to auto."""
    engine = Transformers()

    assert engine.batch_size == "auto"
    assert engine.seed is None
    assert engine.manual_eviction is False
    assert engine.continuous_batching is True
    assert engine.allow_block_sharing is True
    assert engine.max_blocks_per_request is None
    assert engine.use_async_batching is None
    assert engine.use_cuda_graph is None
    assert engine.q_padding_interval_size == 0
    assert engine.kv_padding_interval_size == 0
    assert engine.max_cached_graphs == 0
    assert engine.to_dict()["batch_size"] == "auto"
    assert engine.to_dict()["seed"] is None
    assert engine.to_dict()["manual_eviction"] is False
    assert engine.to_dict()["continuous_batching"] is True
    assert engine.to_dict()["allow_block_sharing"] is True
    assert engine.to_dict()["max_blocks_per_request"] is None
    assert engine.to_dict()["use_async_batching"] is None
    assert engine.to_dict()["use_cuda_graph"] is None
    assert engine.to_dict()["q_padding_interval_size"] == 0
    assert engine.to_dict()["kv_padding_interval_size"] == 0
    assert engine.to_dict()["max_cached_graphs"] == 0


def test_resolve_input_device_keeps_cpu_only_hf_device_maps_on_cpu() -> None:
    """Verify resolve input device keeps cpu only hf device maps on cpu."""
    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        hf_device_map = {"": "cpu"}

    assert _resolve_input_device(FakeModel()) == torch.device("cpu")


def test_resolve_input_device_uses_model_device_when_no_device_map_exists() -> None:
    """Verify resolve input device uses model device when no device map exists."""
    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        device = torch.device("cpu")

    assert _resolve_input_device(FakeModel()) == torch.device("cpu")


def test_transformer_session_from_config_seeds_runtime(monkeypatch) -> None:
    """Verify transformer session from config seeds runtime."""
    import transformers

    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = PretrainedConfig()

        def requires_grad_(self, value: bool):
            """Implement requires grad for fake model."""
            return self

        def eval(self):
            """Implement eval for fake model."""
            return self

        def to(self, device):
            """Implement to for fake model."""
            return self

    seed_calls: list[int | None] = []

    monkeypatch.setattr(
        "evalution.engines.transformers_common._seed_transformer_runtime",
        lambda seed: seed_calls.append(seed),
    )
    monkeypatch.setattr(
        "evalution.engines.transformers_common._load_tokenizer_from_model",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        "evalution.engines.transformers_common._clone_prepare_tokenizer",
        lambda **kwargs: None,
    )

    TransformersSession.from_config(
        Transformers(device="cpu", seed=1234),
        Model(path="/tmp/model"),
    )

    assert seed_calls == [1234]


def test_seed_transformer_runtime_syncs_transformers_python_numpy_torch_and_cuda(
    monkeypatch,
) -> None:
    """Verify seed transformer runtime syncs transformers python numpy torch and cuda."""
    import numpy as np
    import transformers

    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(transformers, "set_seed", lambda seed: calls.append(("transformers", seed)))
    monkeypatch.setattr("evalution.engines.transformers_common.random.seed", lambda seed: calls.append(("python", seed)))
    monkeypatch.setattr(np.random, "seed", lambda seed: calls.append(("numpy", seed)))
    monkeypatch.setattr(torch, "manual_seed", lambda seed: calls.append(("torch", seed)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: calls.append(("cuda", seed)))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(("cuda_all", seed)))

    _seed_transformer_runtime(99)

    assert calls == [
        ("transformers", 99),
        ("python", 99),
        ("numpy", 99),
        ("torch", 99),
        ("cuda", 99),
        ("cuda_all", 99),
    ]


def test_transformer_session_resolves_auto_batch_size_once_per_suite(monkeypatch) -> None:
    """Verify transformer session resolves auto batch size once per suite."""
    session = TransformersSession(
        config=Transformers(batch_size="auto"),
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
        "max_num_beams": 1,
        "dtype_name": "bfloat16",
        "dtype_bytes": 2,
        "total_vram_gib": 0.0,
        "free_vram_gib": 0.0,
        "parameter_count_billions": 1.0,
        "kv_cache_bytes_per_token": None,
    }
    calls = {"estimate": 0}

    monkeypatch.setattr(
        TransformersSession,
        "_batch_size_stats",
        lambda self, batch: stats,
    )

    def fake_estimate(self, batch_stats):
        """Support the surrounding tests with fake estimate."""
        calls["estimate"] += 1
        assert batch_stats is stats
        return 16

    monkeypatch.setattr(
        TransformersSession,
        "_estimate_auto_batch_size",
        fake_estimate,
    )

    assert session.resolve_batch_size(requests) == 16
    assert session.resolve_batch_size(requests) == 16
    assert calls["estimate"] == 1


def test_transformer_session_describes_paged_attention_on_cuda_like_session() -> None:
    """Verify transformer session describes paged attention on cuda like session."""
    session = TransformersSession(
        config=Transformers(attn_implementation="paged|flash_attention_2"),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="flash_attention_2"),
            generate_batch=lambda *args, **kwargs: {},
            set_attn_implementation=lambda value: None,
        ),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    assert session.describe_execution() == {
        "requested_attn_implementation": "paged|flash_attention_2",
        "effective_attn_implementation": "paged|flash_attention_2",
        "paged_attention": True,
        "generation_backend": "continuous_batching",
        "standard_batch_size_cap": None,
    }


def test_transformer_session_gc_clears_caches_and_releases_cuda_allocator(monkeypatch) -> None:
    """Verify transformer session gc clears caches and releases cuda allocator."""
    ipc_collect_calls = {"count": 0}
    empty_cache_calls = {"count": 0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.__setitem__("count", empty_cache_calls["count"] + 1),
    )
    monkeypatch.setattr(
        torch.cuda,
        "ipc_collect",
        lambda: ipc_collect_calls.__setitem__("count", ipc_collect_calls["count"] + 1),
    )

    session = TransformersSession(
        config=Transformers(),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(dtype="bfloat16"),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cuda"),
        stop_criteria_cache={("A",): object()},
        auto_batch_size_cache={("stats",): 16},
        execution_logged=True,
    )

    session.gc()

    assert session.stop_criteria_cache == {}
    assert session.auto_batch_size_cache == {}
    assert session.execution_logged is False
    assert empty_cache_calls["count"] == 1
    assert ipc_collect_calls["count"] == 1


def test_transformer_session_gc_stops_continuous_batching_manager(monkeypatch) -> None:
    """Verify transformer session gc stops continuous batching manager."""
    empty_cache_calls = {"count": 0}
    ipc_collect_calls = {"count": 0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.__setitem__("count", empty_cache_calls["count"] + 1),
    )
    monkeypatch.setattr(
        torch.cuda,
        "ipc_collect",
        lambda: ipc_collect_calls.__setitem__("count", ipc_collect_calls["count"] + 1),
    )

    class FakeContinuousBatchingManager:
        """Provide the fake continuous batching manager helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.evicted_request_ids: list[str] = []
            self.stop_calls = 0

        def evict_request_from_cache(self, request_id: str) -> None:
            """Implement evict request from cache for fake continuous batching manager."""
            self.evicted_request_ids.append(request_id)

        def stop(self, block=True) -> None:
            """Implement stop for fake continuous batching manager."""
            assert block is True
            self.stop_calls += 1

    manager = FakeContinuousBatchingManager()
    session = TransformersSession(
        config=Transformers(manual_eviction=True),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(dtype="bfloat16"),
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cuda"),
        continuous_batching_manager=manager,
        continuous_batching_signature=(("A",), False, None),
        continuous_batching_completed_request_ids={"req_2", "req_1"},
        continuous_batching_request_counter=3,
    )

    session.gc()

    assert manager.evicted_request_ids == ["req_1", "req_2"]
    assert manager.stop_calls == 1
    assert session.continuous_batching_manager is None
    assert session.continuous_batching_signature is None
    assert session.continuous_batching_completed_request_ids == set()
    assert empty_cache_calls["count"] == 1
    assert ipc_collect_calls["count"] == 1


def test_transformer_session_from_config_freezes_model_and_calls_eval(monkeypatch) -> None:
    """Verify transformer session from config freezes model and calls eval."""
    import transformers

    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = PretrainedConfig()
            self.eval_called = False
            self.requires_grad_value: bool | None = None
            self.moved_to: str | None = None

        def requires_grad_(self, value: bool):
            """Implement requires grad for fake model."""
            self.requires_grad_value = value
            return self

        def eval(self):
            """Implement eval for fake model."""
            self.eval_called = True
            return self

        def to(self, device):
            """Implement to for fake model."""
            self.moved_to = str(device)
            return self

    fake_model = FakeModel()
    monkeypatch.setattr(
        "evalution.engines.transformers_common._load_tokenizer_from_model",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: fake_model,
    )
    monkeypatch.setattr(
        "evalution.engines.transformers_common._clone_prepare_tokenizer",
        lambda **kwargs: None,
    )

    session = TransformersSession.from_config(
        Transformers(device="cpu"),
        Model(path="/tmp/model"),
    )

    assert fake_model.requires_grad_value is False
    assert fake_model.eval_called is True
    assert fake_model.moved_to == "cpu"
    assert session.model is fake_model


def test_transformer_session_from_config_prefers_dtype_loader_kwarg(monkeypatch) -> None:
    """Verify transformer session from config prefers dtype loader kwarg."""
    import transformers

    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = PretrainedConfig()

        def requires_grad_(self, value: bool):
            """Implement requires grad for fake model."""
            return self

        def eval(self):
            """Implement eval for fake model."""
            return self

        def to(self, device):
            """Implement to for fake model."""
            return self

    loader_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "evalution.engines.transformers_common._load_tokenizer_from_model",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    def fake_from_pretrained(*args, **kwargs):
        """Support the surrounding tests with fake from pretrained."""
        loader_calls.append(kwargs)
        return FakeModel()

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "evalution.engines.transformers_common._clone_prepare_tokenizer",
        lambda **kwargs: None,
    )

    TransformersSession.from_config(
        Transformers(device="cpu", dtype="bfloat16"),
        Model(path="/tmp/model"),
    )

    assert len(loader_calls) == 1
    assert loader_calls[0]["dtype"] == torch.bfloat16
    assert "torch_dtype" not in loader_calls[0]


def test_transformer_session_from_config_uses_preinitialized_tokenizer(
    monkeypatch,
) -> None:
    """Verify transformer session from config uses preinitialized tokenizer."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = PretrainedConfig()

        def requires_grad_(self, value: bool):
            """Implement requires grad for fake model."""
            return self

        def eval(self):
            """Implement eval for fake model."""
            return self

        def to(self, device):
            """Implement to for fake model."""
            return self

    load_calls: list[object] = []
    provided_tokenizer = FakeTokenizer()

    def fake_load_tokenizer(source: object, **kwargs: object) -> object:
        """Support the surrounding tests with fake load tokenizer."""
        load_calls.append(source)
        return provided_tokenizer

    monkeypatch.setattr(
        "evalution.engines.transformers_common._load_tokenizer_from_model",
        fake_load_tokenizer,
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        "evalution.engines.transformers_common._clone_prepare_tokenizer",
        lambda **kwargs: provided_tokenizer,
    )

    TransformersSession.from_config(
        Transformers(device="cpu"),
        Model(path="/tmp/model", tokenizer=provided_tokenizer),
    )

    assert load_calls == [provided_tokenizer]


def test_transformer_session_from_config_remaps_dtype_for_older_loader(monkeypatch) -> None:
    """Verify transformer session from config remaps dtype for older loader."""
    import transformers

    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = PretrainedConfig()

        def requires_grad_(self, value: bool):
            """Implement requires grad for fake model."""
            return self

        def eval(self):
            """Implement eval for fake model."""
            return self

        def to(self, device):
            """Implement to for fake model."""
            return self

    loader_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "evalution.engines.transformers_common._load_tokenizer_from_model",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    def fake_from_pretrained(*args, **kwargs):
        """Support the surrounding tests with fake from pretrained."""
        loader_calls.append(kwargs)
        if "dtype" in kwargs:
            raise TypeError("from_pretrained() got an unexpected keyword argument 'dtype'")
        return FakeModel()

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "evalution.engines.transformers_common._clone_prepare_tokenizer",
        lambda **kwargs: None,
    )

    TransformersSession.from_config(
        Transformers(device="cpu", dtype="bfloat16"),
        Model(path="/tmp/model"),
    )

    assert len(loader_calls) == 2
    assert loader_calls[0]["dtype"] == torch.bfloat16
    assert "torch_dtype" not in loader_calls[0]
    assert loader_calls[1]["torch_dtype"] == torch.bfloat16
    assert "dtype" not in loader_calls[1]


def test_transformer_session_from_config_reloads_with_device_map_after_meta_to_error(monkeypatch) -> None:
    """Verify transformer session from config reloads with device map after meta to error."""
    import transformers

    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self, *, fail_on_to: bool) -> None:
            """Initialize this object."""
            self.config = PretrainedConfig()
            self.fail_on_to = fail_on_to
            self.eval_called = False
            self.requires_grad_value: bool | None = None
            self.moved_to: str | None = None

        def requires_grad_(self, value: bool):
            """Implement requires grad for fake model."""
            self.requires_grad_value = value
            return self

        def eval(self):
            """Implement eval for fake model."""
            self.eval_called = True
            return self

        def to(self, device):
            """Implement to for fake model."""
            if self.fail_on_to:
                raise NotImplementedError("Cannot copy out of meta tensor; no data!")
            self.moved_to = str(device)
            return self

    loader_calls: list[dict[str, object]] = []
    models = [
        FakeModel(fail_on_to=True),
        FakeModel(fail_on_to=False),
    ]

    monkeypatch.setattr(
        "evalution.engines.transformers_common._load_tokenizer_from_model",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    def fake_from_pretrained(*args, **kwargs):
        """Support the surrounding tests with fake from pretrained."""
        loader_calls.append(kwargs)
        return models.pop(0)

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "evalution.engines.transformers_common._clone_prepare_tokenizer",
        lambda **kwargs: None,
    )

    session = TransformersSession.from_config(
        Transformers(device="cpu"),
        Model(path="/tmp/model"),
    )

    assert len(loader_calls) == 2
    assert "device_map" not in loader_calls[0]
    assert loader_calls[1]["device_map"] == "cpu"
    assert session.model.moved_to is None
    assert session.input_device.type == "cpu"


def test_transformer_build_falls_back_to_compat_engine_when_continuous_batching_is_unavailable(
    monkeypatch,
) -> None:
    """Verify transformer build falls back to compat engine when continuous batching is unavailable."""
    fake_session = object()

    class FakeCompatEngine:
        """Provide the fake compat engine helper used by the surrounding tests."""
        def build(self, model):
            """Build build."""
            assert model.path == "/tmp/model"
            return fake_session

    compat_engine = FakeCompatEngine()

    monkeypatch.setattr(
        "evalution.engines.transformers.transformers_continuous_batching_support",
        lambda: (False, "transformers 4.55.4 is older than 4.56.0"),
    )
    monkeypatch.setattr(
        TransformersCompat,
        "from_transformers",
        classmethod(lambda cls, engine: compat_engine),
    )

    engine = Transformers(device="cpu", attn_implementation="flash_attention_2")
    session = engine.build(Model(path="/tmp/model"))

    assert session is fake_session
    assert engine.resolved_engine == "TransformersCompat"


def test_transformer_compat_uses_preinitialized_tokenizer(monkeypatch) -> None:
    """Verify transformer compat uses preinitialized tokenizer."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        pad_token = "<pad>"
        eos_token = "</s>"
        unk_token = "<unk>"
        padding_side = "right"

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = PretrainedConfig()

        def requires_grad_(self, value: bool):
            """Implement requires grad for fake model."""
            return self

        def eval(self):
            """Implement eval for fake model."""
            return self

        def to(self, device):
            """Implement to for fake model."""
            return self

    load_calls: list[object] = []
    provided_tokenizer = FakeTokenizer()

    def fake_load_tokenizer(source: object, **kwargs: object) -> object:
        """Support the surrounding tests with fake load tokenizer."""
        load_calls.append(source)
        return provided_tokenizer

    monkeypatch.setattr(
        "evalution.engines.transformers_common._load_tokenizer_from_model",
        fake_load_tokenizer,
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )
    monkeypatch.setattr(
        "evalution.engines.transformers_common._clone_prepare_tokenizer",
        lambda **kwargs: provided_tokenizer,
    )

    engine = TransformersCompat(
        device="cpu",
        trust_remote_code=False,
    )
    session = engine.build(Model(path="/tmp/model", tokenizer=provided_tokenizer))

    assert session.model is not None
    assert load_calls == [provided_tokenizer]


def test_transformer_build_rejects_paged_attn_when_continuous_batching_is_unavailable(
    monkeypatch,
) -> None:
    """Verify transformer build rejects paged attn when continuous batching is unavailable."""
    monkeypatch.setattr(
        "evalution.engines.transformers.transformers_continuous_batching_support",
        lambda: (False, "transformers 4.55.4 is older than 4.56.0"),
    )

    with pytest.raises(
        ValueError,
        match="paged attn_implementation requires a transformers build with continuous batching support",
    ):
        Transformers(
            device="cpu",
            attn_implementation="paged|flash_attention_2",
        ).build(Model(path="/tmp/model"))


def test_transformer_build_warns_once_about_pending_nogil_transformers_pr(monkeypatch) -> None:
    """Verify transformer build warns once about pending no-GIL transformers pr."""
    import evalution.engines.transformers as transformer_module

    warnings: list[tuple[str, tuple[object, ...]]] = []

    class FakeLogger:
        """Provide the fake logger helper used by the surrounding tests."""
        def warning(self, message, *args):
            """Implement warning for fake logger."""
            warnings.append((message, args))

    fake_session = object()

    monkeypatch.setattr(
        "evalution.engines.transformers.transformers_continuous_batching_support",
        lambda: (True, "ok"),
    )
    monkeypatch.setattr(
        TransformersSession,
        "from_config",
        classmethod(lambda cls, engine, model_config: fake_session),
    )
    monkeypatch.setattr("evalution.engines.transformers.get_logger", lambda: FakeLogger())
    monkeypatch.setattr(transformer_module, "_PENDING_NOGIL_TRANSFORMERS_PR_WARNED", False)
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False, raising=False)

    engine = Transformers(device="cpu")

    assert engine.build(Model(path="/tmp/model")) is fake_session
    assert engine.build(Model(path="/tmp/model")) is fake_session
    assert len(warnings) == 1
    assert "PR #44924" in warnings[0][0]
    assert warnings[0][1] == (transformer_module._PENDING_NOGIL_TRANSFORMERS_PR_URL,)


def test_transformer_build_skips_pending_nogil_pr_warning_when_gil_is_enabled(monkeypatch) -> None:
    """Verify transformer build skips pending no-GIL pr warning when gil is enabled."""
    import evalution.engines.transformers as transformer_module

    warnings: list[tuple[str, tuple[object, ...]]] = []

    class FakeLogger:
        """Provide the fake logger helper used by the surrounding tests."""
        def warning(self, message, *args):
            """Implement warning for fake logger."""
            warnings.append((message, args))

    fake_session = object()

    monkeypatch.setattr(
        "evalution.engines.transformers.transformers_continuous_batching_support",
        lambda: (True, "ok"),
    )
    monkeypatch.setattr(
        TransformersSession,
        "from_config",
        classmethod(lambda cls, engine, model_config: fake_session),
    )
    monkeypatch.setattr("evalution.engines.transformers.get_logger", lambda: FakeLogger())
    monkeypatch.setattr(transformer_module, "_PENDING_NOGIL_TRANSFORMERS_PR_WARNED", False)
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True, raising=False)

    assert Transformers(device="cpu").build(Model(path="/tmp/model")) is fake_session
    assert warnings == []


def test_transformer_monkey_patches_continuous_batching_generation_loop_once(monkeypatch) -> None:
    """Verify transformer monkey patches continuous batching generation loop once."""
    import evalution.engines.transformers as transformer_module

    entered_devices: list[object] = []
    calls: list[str] = []

    class FakeCudaDeviceContext:
        """Provide the fake cuda device context helper used by the surrounding tests."""
        def __init__(self, device: object) -> None:
            """Initialize this object."""
            self.device = device

        def __enter__(self) -> None:
            """Enter the managed context for this object."""
            entered_devices.append(self.device)

        def __exit__(self, exc_type, exc, tb) -> bool:
            """Exit the managed context for this object."""
            del exc_type, exc, tb
            return False

    class FakeContinuousBatchingManager:
        """Provide the fake continuous batching manager helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.model = SimpleNamespace(device=SimpleNamespace(type="cuda", index=3))

        def _run_generation_loop(self) -> str:
            """Run generation loop."""
            calls.append("run")
            return "ok"

    original = FakeContinuousBatchingManager._run_generation_loop
    monkeypatch.setattr(torch.cuda, "device", lambda device: FakeCudaDeviceContext(device))

    # The stop-gap is class-level because the real continuous-batching work runs on the manager
    # thread. Re-applying it should therefore keep a single stable wrapper on that entrypoint.
    transformer_module._patch_continuous_batching_manager_cuda_context_once(FakeContinuousBatchingManager)
    patched = FakeContinuousBatchingManager._run_generation_loop
    transformer_module._patch_continuous_batching_manager_cuda_context_once(FakeContinuousBatchingManager)

    assert patched is not original
    assert FakeContinuousBatchingManager._run_generation_loop is patched
    assert patched.__wrapped__ is original

    manager = FakeContinuousBatchingManager()
    assert manager._run_generation_loop() == "ok"
    assert entered_devices == [manager.model.device]
    assert calls == ["run"]


def test_transformer_monkey_patch_skips_cuda_context_for_cpu_manager(monkeypatch) -> None:
    """Verify transformer monkey patch skips cuda context for cpu manager."""
    import evalution.engines.transformers as transformer_module

    class FakeContinuousBatchingManager:
        """Provide the fake continuous batching manager helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.model = SimpleNamespace(device=SimpleNamespace(type="cpu"))

        def _run_generation_loop(self) -> str:
            """Run generation loop."""
            return "ok"

    def fail_if_called(device: object) -> object:
        """Support the surrounding tests with fail if called."""
        raise AssertionError(f"torch.cuda.device should not be used for CPU manager: {device!r}")

    monkeypatch.setattr(torch.cuda, "device", fail_if_called)
    # CPU sessions should keep the original control flow and never touch CUDA state.
    transformer_module._patch_continuous_batching_manager_cuda_context_once(FakeContinuousBatchingManager)

    manager = FakeContinuousBatchingManager()
    assert manager._run_generation_loop() == "ok"


def test_transformer_monkey_patch_skips_manager_without_generation_loop(monkeypatch) -> None:
    """Verify transformer monkey patch skips manager without generation loop."""
    import evalution.engines.transformers as transformer_module

    class FakeContinuousBatchingManager:
        """Provide the fake continuous batching manager helper used by the surrounding tests."""
        pass

    def fail_if_called(device: object) -> object:
        """Support the surrounding tests with fail if called."""
        raise AssertionError(f"torch.cuda.device should not be used without _run_generation_loop: {device!r}")

    monkeypatch.setattr(torch.cuda, "device", fail_if_called)
    transformer_module._patch_continuous_batching_manager_cuda_context_once(FakeContinuousBatchingManager)

    assert "_run_generation_loop" not in FakeContinuousBatchingManager.__dict__


@pytest.mark.parametrize(
    ("attn_implementation", "max_batch_tokens", "expected_blocks"),
    [
        ("flash_attention_2", 2048, 1),
        ("flash_attention_2", 8192, 16),
        ("flash_attention_3", 2048, 1),
    ],
)
def test_transformer_monkey_patch_seeds_flash_attention_cb_defaults(
    monkeypatch,
    attn_implementation: str,
    max_batch_tokens: int,
    expected_blocks: int,
) -> None:
    """Verify transformer monkey patch seeds flash attention cb defaults."""
    import evalution.engines.transformers as transformer_module
    from transformers import ContinuousBatchingConfig
    from transformers.generation.continuous_batching import continuous_api

    manager_cls = continuous_api.ContinuousBatchingManager
    processor_cls = continuous_api.ContinuousBatchProcessor

    monkeypatch.setattr(transformer_module, "_transformers_supports_flash_attention_auto_max_blocks", lambda: False)
    monkeypatch.setattr(transformer_module, "_transformers_supports_fa2_decode_fast_path", lambda: False)
    monkeypatch.setattr(manager_cls, "_create_batch_processor", lambda self: self.continuous_batching_config)
    monkeypatch.setattr(processor_cls, "_ensure_decode_fast_path_is_available", lambda self: None)

    transformer_module._patch_continuous_batching_flash_attention_decode_once()

    manager = SimpleNamespace(
        continuous_batching_config=ContinuousBatchingConfig(
            max_batch_tokens=max_batch_tokens,
            max_blocks_per_request=None,
            use_cuda_graph=None,
        ),
        model=SimpleNamespace(
            config=SimpleNamespace(_attn_implementation=attn_implementation),
        ),
    )

    cb_config = manager_cls._create_batch_processor(manager)

    assert cb_config.use_cuda_graph is False
    assert cb_config.max_blocks_per_request == expected_blocks


def test_transformer_monkey_patch_preserves_explicit_flash_attention_cb_settings(monkeypatch) -> None:
    """Verify transformer monkey patch preserves explicit flash attention cb settings."""
    import evalution.engines.transformers as transformer_module
    from transformers import ContinuousBatchingConfig
    from transformers.generation.continuous_batching import continuous_api

    manager_cls = continuous_api.ContinuousBatchingManager
    processor_cls = continuous_api.ContinuousBatchProcessor

    monkeypatch.setattr(transformer_module, "_transformers_supports_flash_attention_auto_max_blocks", lambda: False)
    monkeypatch.setattr(transformer_module, "_transformers_supports_fa2_decode_fast_path", lambda: False)
    monkeypatch.setattr(manager_cls, "_create_batch_processor", lambda self: self.continuous_batching_config)
    monkeypatch.setattr(processor_cls, "_ensure_decode_fast_path_is_available", lambda self: None)

    transformer_module._patch_continuous_batching_flash_attention_decode_once()

    manager = SimpleNamespace(
        continuous_batching_config=ContinuousBatchingConfig(
            max_batch_tokens=4096,
            max_blocks_per_request=0,
            use_cuda_graph=True,
        ),
        model=SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="flash_attention_2"),
        ),
    )

    cb_config = manager_cls._create_batch_processor(manager)

    assert cb_config.use_cuda_graph is True
    assert cb_config.max_blocks_per_request == 0


def test_transformer_monkey_patch_accepts_fa2_decode_fast_path(monkeypatch) -> None:
    """Verify transformer monkey patch accepts fa2 decode fast path."""
    import evalution.engines.transformers as transformer_module
    from transformers.generation.continuous_batching import continuous_api

    processor_cls = continuous_api.ContinuousBatchProcessor
    monkeypatch.setattr(transformer_module, "_transformers_supports_flash_attention_auto_max_blocks", lambda: False)
    monkeypatch.setattr(transformer_module, "_transformers_supports_fa2_decode_fast_path", lambda: False)
    monkeypatch.setattr(processor_cls, "_ensure_decode_fast_path_is_available", lambda self: setattr(self.cache, "max_blocks_per_request", 0))
    monkeypatch.setattr(
        "transformers.utils.generic.is_flash_attention_requested",
        lambda config, version=None: version == 2 and config._attn_implementation == "flash_attention_2",
    )
    monkeypatch.setattr(
        "transformers.modeling_flash_attention_utils.lazy_import_paged_flash_attention",
        lambda *_args, **_kwargs: (None, object()),
    )

    transformer_module._patch_continuous_batching_flash_attention_decode_once()

    cache = SimpleNamespace(max_blocks_per_request=4, num_sliding_attention_groups=0)
    processor = SimpleNamespace(
        cache=cache,
        config=SimpleNamespace(_attn_implementation="flash_attention_2"),
    )

    processor_cls._ensure_decode_fast_path_is_available(processor)

    assert cache.max_blocks_per_request == 4


def test_transformer_monkey_patch_skips_when_transformers_has_native_flash_attention_support(monkeypatch) -> None:
    """Verify transformer monkey patch skips when transformers has native flash attention support."""
    import evalution.engines.transformers as transformer_module
    from transformers.generation.continuous_batching import continuous_api

    manager_cls = continuous_api.ContinuousBatchingManager
    processor_cls = continuous_api.ContinuousBatchProcessor

    original_create_batch_processor = manager_cls._create_batch_processor
    original_ensure_fast_path = processor_cls._ensure_decode_fast_path_is_available
    monkeypatch.setattr(transformer_module, "_transformers_supports_flash_attention_auto_max_blocks", lambda: True)
    monkeypatch.setattr(transformer_module, "_transformers_supports_fa2_decode_fast_path", lambda: True)

    transformer_module._patch_continuous_batching_flash_attention_decode_once()

    assert manager_cls._create_batch_processor is original_create_batch_processor
    assert processor_cls._ensure_decode_fast_path_is_available is original_ensure_fast_path


def test_transformer_monkey_patch_keeps_defaults_patch_when_only_native_decode_support_exists(monkeypatch) -> None:
    """Verify transformer monkey patch keeps defaults patch when only native decode support exists."""
    import evalution.engines.transformers as transformer_module
    from transformers import ContinuousBatchingConfig
    from transformers.generation.continuous_batching import continuous_api

    manager_cls = continuous_api.ContinuousBatchingManager
    processor_cls = continuous_api.ContinuousBatchProcessor
    original_ensure_fast_path = processor_cls._ensure_decode_fast_path_is_available

    monkeypatch.setattr(transformer_module, "_transformers_supports_flash_attention_auto_max_blocks", lambda: False)
    monkeypatch.setattr(transformer_module, "_transformers_supports_fa2_decode_fast_path", lambda: True)
    monkeypatch.setattr(manager_cls, "_create_batch_processor", lambda self: self.continuous_batching_config)

    transformer_module._patch_continuous_batching_flash_attention_decode_once()

    manager = SimpleNamespace(
        continuous_batching_config=ContinuousBatchingConfig(
            max_batch_tokens=2048,
            max_blocks_per_request=None,
            use_cuda_graph=None,
        ),
        model=SimpleNamespace(
            config=SimpleNamespace(_attn_implementation="flash_attention_2"),
        ),
    )

    cb_config = manager_cls._create_batch_processor(manager)

    assert cb_config.use_cuda_graph is False
    assert cb_config.max_blocks_per_request == 1
    assert processor_cls._ensure_decode_fast_path_is_available is original_ensure_fast_path


def test_transformer_session_prepare_requests_batches_tokenization() -> None:
    """Verify transformer session prepare requests batches tokenization."""
    class FakePrepareTokenizer:
        """Provide the fake prepare tokenizer helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.encode_calls: list[list[str]] = []

        def __call__(self, prompts, *, add_special_tokens=False, padding=False):
            """Implement call for fake prepare tokenizer."""
            assert add_special_tokens is False
            assert padding is False
            assert isinstance(prompts, list)
            self.encode_calls.append(list(prompts))
            return {
                "input_ids": [
                    [index + 1, index + 2, index + 3]
                    for index, _prompt in enumerate(prompts)
                ]
            }

        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
            """Implement apply chat template for fake prepare tokenizer."""
            assert tokenize is False
            assert add_generation_prompt is True
            return f"chat::{messages[0]['content']}"

    prepare_tokenizer = FakePrepareTokenizer()
    session = TransformersSession(
        config=Transformers(),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(dtype="bfloat16"),
        tokenizer=SimpleNamespace(),
        prepare_tokenizer=prepare_tokenizer,
        input_device=SimpleNamespace(type="cpu"),
    )

    prepared = session.prepare_requests(
        [
            GenerationRequest(prompt="Q: 1 + 1\nA:"),
            GenerationRequest(messages=[{"role": "user", "content": "Q: 2 + 2\nA:"}]),
            GenerationRequest(
                prompt="Q: 3 + 3\nA:",
                rendered_prompt="Q: 3 + 3\nA:",
                input_ids=[9, 9, 9],
            ),
        ]
    )

    assert prepare_tokenizer.encode_calls == [["Q: 1 + 1\nA:", "chat::Q: 2 + 2\nA:"]]
    assert prepared[0].rendered_prompt == "Q: 1 + 1\nA:"
    assert prepared[0].input_ids == [1, 2, 3]
    assert prepared[1].rendered_prompt == "chat::Q: 2 + 2\nA:"
    assert prepared[1].input_ids == [2, 3, 4]
    assert prepared[2].rendered_prompt == "Q: 3 + 3\nA:"
    assert prepared[2].input_ids == [9, 9, 9]


def test_transformer_session_generate_reuses_pretokenized_requests() -> None:
    """Verify transformer session generate reuses pretokenized requests."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1

        def __init__(self) -> None:
            """Initialize this object."""
            self.pad_calls = 0
            self.encode_calls = 0

        def __call__(self, prompts, **kwargs):
            """Implement call for fake tokenizer."""
            del prompts, kwargs
            self.encode_calls += 1
            raise AssertionError("pretokenized requests should not be re-encoded")

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            """Implement pad for fake tokenizer."""
            assert return_tensors == "pt"
            assert padding is True
            self.pad_calls += 1
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

        def decode(self, token_ids, *, skip_special_tokens=False):
            """Implement decode for fake tokenizer."""
            del token_ids, skip_special_tokens
            return "The answer is 42."

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def generate(self, **kwargs):
            """Generate generate."""
            del kwargs
            return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    tokenizer = FakeTokenizer()
    session = TransformersSession(
        config=Transformers(),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=tokenizer,
        input_device=torch.device("cpu"),
    )

    outputs = session.generate(
        [
            GenerationRequest(
                prompt="Q: 40 + 2\nA:",
                rendered_prompt="Q: 40 + 2\nA:",
                input_ids=[1, 2, 3],
            )
        ],
        batch_size=1,
    )

    assert len(outputs) == 1
    assert outputs[0].text == "The answer is 42."
    assert tokenizer.pad_calls == 1
    assert tokenizer.encode_calls == 0


def test_transformer_session_generate_uses_continuous_batching_manager_when_paged_attention_is_enabled(
    monkeypatch,
) -> None:
    """Verify transformer session generate uses continuous batching manager when paged attention is enabled. Preserve the fallback order expected by the surrounding caller."""
    import transformers

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
                return {"input_ids": [11, 12, 13]}
            return {"input_ids": [[11, 12, 13] for _ in prompts]}

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

        def set_attn_implementation(self, value: str) -> None:
            """Implement set attn implementation for fake model."""
            self.config._attn_implementation = value

    class FakeContinuousBatchingManager:
        """Provide the fake continuous batching manager helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        last_instance: FakeContinuousBatchingManager | None = None

        def __init__(
            self,
            model,
            generation_config,
            *,
            manual_eviction=False,
            q_padding_interval_size=0,
            kv_padding_interval_size=0,
            max_cached_graphs=0,
            allow_block_sharing=True,
            use_async_batching=None,
        ):
            """Initialize this object."""
            self.model = model
            self.generation_config = generation_config
            self.manual_eviction = manual_eviction
            self.q_padding_interval_size = q_padding_interval_size
            self.kv_padding_interval_size = kv_padding_interval_size
            self.max_cached_graphs = max_cached_graphs
            self.allow_block_sharing = allow_block_sharing
            self.use_async_batching = use_async_batching
            self.event_log: list[str] = []
            self.result_queue: list[FakeContinuousOutput] = []
            self.active_requests = 0
            self.max_active_requests = 0
            self.started = False
            self.stopped = False
            self.stop_calls = 0
            self.added_request_ids: list[str] = []
            FakeContinuousBatchingManager.last_instance = self

        def start(self) -> None:
            """Implement start for fake continuous batching manager."""
            self.started = True

        def add_request(self, input_ids, *, request_id=None, max_new_tokens=None, streaming=False):
            """Implement add request for fake continuous batching manager."""
            del max_new_tokens, streaming
            assert input_ids == [11, 12, 13]
            assert request_id is not None
            self.added_request_ids.append(request_id)
            self.event_log.append(f"add:{request_id}")
            self.active_requests += 1
            self.max_active_requests = max(self.max_active_requests, self.active_requests)
            if len(self.added_request_ids) == 2:
                self.result_queue.extend(
                    [
                        FakeContinuousOutput(self.added_request_ids[1], [201, 202]),
                        FakeContinuousOutput(self.added_request_ids[0], [101, 102, 103]),
                    ]
                )
            elif len(self.added_request_ids) == 3:
                self.result_queue.append(FakeContinuousOutput(request_id, [301, 302, 303]))

        def get_result(self, timeout=None):
            """Get result."""
            del timeout
            if not self.result_queue:
                return None
            result = self.result_queue.pop(0)
            self.event_log.append(f"result:{result.request_id}")
            self.active_requests -= 1
            return result

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

    model = FakeModel()
    session = TransformersSession(
        config=Transformers(
            attn_implementation="paged|flash_attention_2",
            manual_eviction=True,
            allow_block_sharing=False,
            use_async_batching=False,
            q_padding_interval_size=128,
            kv_padding_interval_size=4096,
            max_cached_graphs=8,
        ),
        model_config=Model(path="/tmp/model"),
        model=model,
        tokenizer=FakeTokenizer(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    outputs = session.generate(
        [
            GenerationRequest(prompt="Q: 40 + 2\nA:", stop=["Q:"]),
            GenerationRequest(prompt="Q: 41 + 2\nA:", stop=["Q:"]),
            GenerationRequest(prompt="Q: 42 + 2\nA:", stop=["Q:"]),
        ],
        batch_size=2,
    )

    assert len(outputs) == 3
    assert [output.text for output in outputs] == ["The answer is 42."] * 3
    manager = FakeContinuousBatchingManager.last_instance
    assert manager is not None
    assert manager.started is True
    assert manager.stopped is False
    assert manager.stop_calls == 0
    assert manager.manual_eviction is True
    assert manager.allow_block_sharing is False
    assert manager.use_async_batching is False
    assert manager.q_padding_interval_size == 128
    assert manager.kv_padding_interval_size == 4096
    assert manager.max_cached_graphs == 8
    assert manager.generation_config.stop_strings == ["Q:"]


def test_transformer_session_generate_continuous_refills_paged_manager_while_caller_is_paused(
    monkeypatch,
) -> None:
    """Verify transformer session generate continuous refills paged manager while caller is paused. Preserve the fallback order expected by the surrounding caller."""
    import transformers

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
                return {"input_ids": [11, 12, 13]}
            return {"input_ids": [[11, 12, 13] for _ in prompts]}

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

    session = TransformersSession(
        config=Transformers(attn_implementation="paged|flash_attention_2"),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
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


def test_transformer_session_generate_continuous_bounds_paged_request_queue_and_skips_large_preview_when_batch_size_is_fixed(
    monkeypatch,
) -> None:
    """Verify transformer session generate continuous bounds paged request queue and skips large preview when batch size is fixed."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1

        def decode(self, token_ids, *, skip_special_tokens=False):
            """Implement decode for fake tokenizer."""
            del token_ids, skip_special_tokens
            return "The answer is 42."

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = PretrainedConfig()
            self.config._attn_implementation = "flash_attention_2"

        def set_attn_implementation(self, value: str) -> None:
            """Implement set attn implementation for fake model."""
            self.config._attn_implementation = value

    consumed_count = 0
    captured_queue_max_size: int | None = None

    def request_source():
        """Support the surrounding tests with request source."""
        nonlocal consumed_count
        for index in range(100):
            consumed_count += 1
            yield (
                index,
                GenerationRequest(
                    prompt=f"Q: {index}\nA:",
                    stop=["Q:"],
                ),
            )

    def fake_stream_request_results(
        requests,
        *,
        producer_name,
        consumer_name,
        process_requests,
        require_non_main_thread,
        request_queue_max_size=None,
    ):
        """Support the surrounding tests with fake stream request results."""
        nonlocal captured_queue_max_size
        del requests, producer_name, consumer_name, process_requests, require_non_main_thread
        captured_queue_max_size = request_queue_max_size
        return iter(())

    monkeypatch.setattr(
        "evalution.engines.transformers.stream_request_results",
        fake_stream_request_results,
    )

    session = TransformersSession(
        config=Transformers(attn_implementation="paged|flash_attention_2"),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    list(session.generate_continuous(request_source(), batch_size=32))

    assert captured_queue_max_size == 64
    assert consumed_count == 1


def test_transformer_session_generate_supports_config_object_continuous_batching_manager_api(monkeypatch) -> None:
    """Verify transformer session generate supports config object continuous batching manager API."""
    import transformers
    import torch as torch_module

    compile_calls: list[object] = []

    def fake_compile(obj, *args, **kwargs):
        """Support the surrounding tests with fake compile."""
        del args, kwargs
        compile_calls.append(obj)
        return obj

    monkeypatch.setattr(torch_module, "compile", fake_compile)

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
                return {"input_ids": [11, 12, 13]}
            return {"input_ids": [[11, 12, 13] for _ in prompts]}

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

        def set_attn_implementation(self, value: str) -> None:
            """Implement set attn implementation for fake model."""
            self.config._attn_implementation = value

    class FakeContinuousBatchingConfig:
        """Provide the fake continuous batching config helper used by the surrounding tests."""
        def __init__(
            self,
            *,
            allow_block_sharing=True,
            max_blocks_per_request=None,
            use_async_batching=None,
            use_cuda_graph=None,
            q_padding_interval_size=0,
            kv_padding_interval_size=0,
            max_cached_graphs=0,
            torch_compile=False,
        ):
            """Initialize this object."""
            if torch_compile:
                torch_module.compile(SimpleNamespace())
            self.allow_block_sharing = allow_block_sharing
            self.max_blocks_per_request = max_blocks_per_request
            self.use_async_batching = use_async_batching
            self.use_cuda_graph = use_cuda_graph
            self.q_padding_interval_size = q_padding_interval_size
            self.kv_padding_interval_size = kv_padding_interval_size
            self.max_cached_graphs = max_cached_graphs
            self.torch_compile = torch_compile

    class FakeContinuousBatchingManager:
        """Provide the fake continuous batching manager helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        last_instance: FakeContinuousBatchingManager | None = None

        def __init__(self, model, generation_config, continuous_batching_config):
            """Initialize this object."""
            self.model = model
            self.generation_config = generation_config
            self.continuous_batching_config = continuous_batching_config
            self.result_queue = [FakeContinuousOutput("req_0", [101, 102, 103])]
            self.started = False
            self.stopped = False
            self.added_request_ids: list[str] = []
            FakeContinuousBatchingManager.last_instance = self

        def start(self) -> None:
            """Implement start for fake continuous batching manager."""
            self.started = True

        def add_request(self, input_ids, *, request_id=None, max_new_tokens=None, streaming=False):
            """Implement add request for fake continuous batching manager."""
            del input_ids, max_new_tokens, streaming
            assert request_id is not None
            self.added_request_ids.append(request_id)

        def get_result(self, timeout=None):
            """Get result."""
            del timeout
            if not self.result_queue:
                return None
            return self.result_queue.pop(0)

        def is_running(self) -> bool:
            """Implement is running for fake continuous batching manager."""
            return self.started and not self.stopped

        def stop(self, block=True):
            """Implement stop for fake continuous batching manager."""
            assert block is True
            self.stopped = True

    monkeypatch.setattr(
        transformers,
        "ContinuousBatchingConfig",
        FakeContinuousBatchingConfig,
        raising=False,
    )
    monkeypatch.setattr(transformers, "ContinuousBatchingManager", FakeContinuousBatchingManager)

    session = TransformersSession(
        config=Transformers(
            attn_implementation="paged|flash_attention_2",
            manual_eviction=True,
            allow_block_sharing=False,
            max_blocks_per_request=32,
            use_async_batching=False,
            use_cuda_graph=True,
            q_padding_interval_size=128,
            kv_padding_interval_size=4096,
            max_cached_graphs=8,
        ),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    outputs = session.generate([GenerationRequest(prompt="Q: 40 + 2\nA:", stop=["Q:"])], batch_size=1)

    assert [output.text for output in outputs] == ["The answer is 42."]
    manager = FakeContinuousBatchingManager.last_instance
    assert manager is not None
    assert manager.started is True
    assert manager.generation_config.stop_strings == ["Q:"]
    assert manager.continuous_batching_config.allow_block_sharing is False
    assert manager.continuous_batching_config.max_blocks_per_request == 32
    assert manager.continuous_batching_config.use_async_batching is False
    assert manager.continuous_batching_config.use_cuda_graph is True
    assert manager.continuous_batching_config.q_padding_interval_size == 128
    assert manager.continuous_batching_config.kv_padding_interval_size == 4096
    assert manager.continuous_batching_config.max_cached_graphs == 8
    assert manager.continuous_batching_config.torch_compile is True
    assert len(compile_calls) == 1


def test_transformer_session_disables_continuous_batching_when_configured_off(monkeypatch) -> None:
    """Verify transformer session disables continuous batching when configured off."""
    import transformers

    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "left"

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = PretrainedConfig()

    compile_calls: list[object] = []

    def fake_compile(obj, *args, **kwargs):
        """Support the surrounding tests with fake compile."""
        del args, kwargs
        compile_calls.append(obj)
        return obj

    monkeypatch.setattr("torch.compile", fake_compile)
    monkeypatch.setattr(
        transformers,
        "ContinuousBatchingConfig",
        type(
            "FakeContinuousBatchingConfig",
            (),
            {
                "__init__": lambda self, **kwargs: setattr(self, "_kwargs", kwargs)
                or None,
            },
        ),
        raising=False,
    )

    monkeypatch.setattr(
        "evalution.engines.transformers.load_transformer_runtime",
        lambda config, model_config: SimpleNamespace(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            prepare_tokenizer=None,
            input_device=SimpleNamespace(type="cuda"),
            requested_attn_implementation="flash_attention_2",
        ),
    )

    session = TransformersSession.from_config(
        Transformers(continuous_batching=False, attn_implementation="paged|flash_attention_2"),
        Model(path="/tmp/model"),
    )

    assert session.paged_attention_enabled is False
    assert session.generation_backend == "generate"
    assert compile_calls == []


def test_transformer_session_loglikelihood_scores_pretokenized_requests() -> None:
    """Verify transformer session loglikelihood scores pretokenized requests. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "left"

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            """Implement pad for fake tokenizer."""
            assert return_tensors == "pt"
            assert padding is True
            sequences = [list(ids) for ids in encoded_inputs["input_ids"]]
            max_length = max(len(ids) for ids in sequences)
            padded_ids = []
            attention_masks = []
            for ids in sequences:
                pad_count = max_length - len(ids)
                padded_ids.append(([0] * pad_count) + ids)
                attention_masks.append(([0] * pad_count) + ([1] * len(ids)))
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = SimpleNamespace(max_position_embeddings=8)

        def __call__(self, *, input_ids, attention_mask=None):
            """Implement call for fake model. Keep the nested traversal explicit so ordering and metadata stay aligned."""
            del attention_mask
            batch_size, sequence_length = input_ids.shape
            vocab_size = 16
            logits = torch.full((batch_size, sequence_length, vocab_size), -2.0)
            for row in range(batch_size):
                for position in range(sequence_length - 1):
                    next_token = int(input_ids[row, position + 1].item())
                    logits[row, position, next_token] = 3.0
            return SimpleNamespace(logits=logits)

    session = TransformersSession(
        config=Transformers(batch_size=4),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
    )

    outputs = session.loglikelihood(
        [
            LoglikelihoodRequest(
                context_input_ids=[5, 6],
                continuation_input_ids=[7, 8],
            ),
            LoglikelihoodRequest(
                continuation_input_ids=[4],
            ),
        ],
        batch_size=2,
    )

    expected_token_logprob = float(
        torch.log_softmax(
            torch.tensor([3.0] + ([-2.0] * 15), dtype=torch.float32),
            dim=0,
        )[0].item()
    )

    assert len(outputs) == 2
    assert outputs[0].token_count == 2
    assert outputs[0].is_greedy is True
    assert outputs[0].logprob == pytest.approx(expected_token_logprob * 2, abs=1e-6)
    assert outputs[1].token_count == 1
    assert outputs[1].is_greedy is True
    assert outputs[1].logprob == pytest.approx(expected_token_logprob, abs=1e-6)


def test_transformer_session_loglikelihood_context_uses_tokenizer_default_special_tokens() -> None:
    """Verify transformer session loglikelihood context uses tokenizer default special tokens."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        bos_token_id = 99
        eos_token_id = 1
        pad_token_id = 0
        padding_side = "left"

        def __call__(self, prompt, *, add_special_tokens=True, **kwargs):
            """Implement call for fake tokenizer."""
            del kwargs
            base_ids = {
                "ctx": [11, 12],
                "<bos>ctx": [99, 11, 12],
                " target": [13],
            }[prompt]
            input_ids = list(base_ids)
            if add_special_tokens and (not input_ids or input_ids[0] != self.bos_token_id):
                input_ids = [self.bos_token_id] + input_ids
            return {"input_ids": input_ids}

        def decode(self, token_ids, *, skip_special_tokens=False):
            """Implement decode for fake tokenizer."""
            del skip_special_tokens
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            if list(token_ids) == [self.bos_token_id]:
                return "<bos>"
            raise AssertionError(f"unexpected token ids: {token_ids}")

    session = TransformersSession(
        config=Transformers(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
    )

    prefix_ids, target_ids, metadata = session._prepare_loglikelihood_request(
        LoglikelihoodRequest(
            context="ctx",
            continuation=" target",
            metadata={"row": 1},
        )
    )
    explicit_prefix_ids, explicit_target_ids, _explicit_metadata = session._prepare_loglikelihood_request(
        LoglikelihoodRequest(
            context="<bos>ctx",
            continuation=" target",
        )
    )

    assert prefix_ids == [99, 11, 12]
    assert target_ids == [13]
    assert metadata == {"row": 1}
    assert explicit_prefix_ids == [99, 11, 12]
    assert explicit_target_ids == [13]


def test_transformer_session_loglikelihood_right_pads_batches_without_attention_mask() -> None:
    """Verify transformer session loglikelihood right pads batches without attention mask."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "left"

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = SimpleNamespace(max_position_embeddings=8)

        def __call__(self, *, input_ids, attention_mask=None):
            """Implement call for fake model. Keep the nested traversal explicit so ordering and metadata stay aligned."""
            assert attention_mask is None
            assert input_ids.tolist() == [
                [5, 6, 7, 8],
                [1, 4, 0, 0],
            ]
            batch_size, sequence_length = input_ids.shape
            vocab_size = 16
            logits = torch.full((batch_size, sequence_length, vocab_size), -2.0)
            for row in range(batch_size):
                for position in range(sequence_length - 1):
                    next_token = int(input_ids[row, position + 1].item())
                    logits[row, position, next_token] = 3.0
            return SimpleNamespace(logits=logits)

    session = TransformersSession(
        config=Transformers(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
    )

    outputs = session.loglikelihood(
        [
            LoglikelihoodRequest(
                context_input_ids=[5, 6],
                continuation_input_ids=[7, 8],
            ),
            LoglikelihoodRequest(
                context_input_ids=[1],
                continuation_input_ids=[4],
            ),
        ],
        batch_size=2,
    )

    expected_token_logprob = float(
        torch.log_softmax(
            torch.tensor([3.0] + ([-2.0] * 15), dtype=torch.float32),
            dim=0,
        )[0].item()
    )

    assert len(outputs) == 2
    assert outputs[0].logprob == pytest.approx(expected_token_logprob * 2, abs=1e-6)
    assert outputs[1].logprob == pytest.approx(expected_token_logprob, abs=1e-6)


def test_transformer_session_loglikelihood_sorts_requests_by_total_length_before_scoring(
    monkeypatch,
) -> None:
    """Verify transformer session loglikelihood sorts requests by total length before scoring."""
    session = TransformersSession(
        config=Transformers(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(eos_token_id=1),
        input_device=torch.device("cpu"),
    )
    observed_chunk_order: list[int] = []

    def fake_prepare(self, request):
        """Support the surrounding tests with fake prepare."""
        return [], list(request.metadata["token_ids"]), dict(request.metadata)

    def fake_score(self, chunks, *, batch_size):
        """Support the surrounding tests with fake score. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        assert batch_size == 2
        observed_chunk_order[:] = [chunk.request_index for chunk in chunks]
        return [
            LoglikelihoodOutput(
                logprob=float(chunk.request_index),
                is_greedy=True,
                token_count=chunk.score_count,
                metadata=dict(chunk.metadata),
            )
            for chunk in chunks
        ]

    monkeypatch.setattr(
        TransformersSession,
        "_prepare_loglikelihood_request",
        fake_prepare,
    )
    monkeypatch.setattr(
        TransformersSession,
        "_score_chunks",
        fake_score,
    )

    outputs = session.loglikelihood(
        [
            LoglikelihoodRequest(metadata={"token_ids": [3, 4, 5]}),
            LoglikelihoodRequest(metadata={"token_ids": [7, 8, 9, 10, 11]}),
            LoglikelihoodRequest(metadata={"token_ids": [2]}),
        ],
        batch_size=2,
    )

    assert observed_chunk_order == [1, 0, 2]
    assert [output.logprob for output in outputs] == [0.0, 1.0, 2.0]


def test_transformer_session_loglikelihood_reports_non_greedy_predictions() -> None:
    """Verify transformer session loglikelihood reports non greedy predictions."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            """Implement pad for fake tokenizer."""
            assert return_tensors == "pt"
            assert padding is True
            sequences = [list(ids) for ids in encoded_inputs["input_ids"]]
            max_length = max(len(ids) for ids in sequences)
            padded_ids = []
            attention_masks = []
            for ids in sequences:
                pad_count = max_length - len(ids)
                padded_ids.append(ids + ([0] * pad_count))
                attention_masks.append(([1] * len(ids)) + ([0] * pad_count))
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = SimpleNamespace(max_position_embeddings=8)

        def __call__(self, *, input_ids, attention_mask=None):
            """Implement call for fake model. Keep the nested traversal explicit so ordering and metadata stay aligned."""
            del attention_mask
            batch_size, sequence_length = input_ids.shape
            vocab_size = 16
            logits = torch.full((batch_size, sequence_length, vocab_size), -2.0)
            for row in range(batch_size):
                for position in range(sequence_length - 1):
                    next_token = int(input_ids[row, position + 1].item())
                    logits[row, position, next_token] = 3.0
            logits[0, 0, 9] = 5.0
            return SimpleNamespace(logits=logits)

    session = TransformersSession(
        config=Transformers(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
    )

    outputs = session.loglikelihood(
        [
            LoglikelihoodRequest(
                continuation_input_ids=[4],
            )
        ],
        batch_size=1,
    )

    assert len(outputs) == 1
    assert outputs[0].token_count == 1
    assert outputs[0].is_greedy is False


def test_transformer_session_loglikelihood_uses_request_progress_title(monkeypatch) -> None:
    """Verify transformer session loglikelihood uses request progress title."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            """Implement pad for fake tokenizer."""
            assert return_tensors == "pt"
            assert padding is True
            sequences = [list(ids) for ids in encoded_inputs["input_ids"]]
            max_length = max(len(ids) for ids in sequences)
            padded_ids = []
            attention_masks = []
            for ids in sequences:
                pad_count = max_length - len(ids)
                padded_ids.append(ids + ([0] * pad_count))
                attention_masks.append(([1] * len(ids)) + ([0] * pad_count))
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = SimpleNamespace(max_position_embeddings=8)

        def __call__(self, *, input_ids, attention_mask=None):
            """Implement call for fake model. Keep the nested traversal explicit so ordering and metadata stay aligned."""
            del attention_mask
            batch_size, sequence_length = input_ids.shape
            vocab_size = 16
            logits = torch.full((batch_size, sequence_length, vocab_size), -2.0)
            for row in range(batch_size):
                for position in range(sequence_length - 1):
                    next_token = int(input_ids[row, position + 1].item())
                    logits[row, position, next_token] = 3.0
            return SimpleNamespace(logits=logits)

    class FakeProgressBar:
        """Provide the fake progress bar helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.titles: list[str] = []
            self.subtitles: list[str] = []
            self.next_calls = 0
            self.draw_calls = 0

        def title(self, value: str):
            """Implement title for fake progress bar."""
            self.titles.append(value)
            return self

        def subtitle(self, value: str):
            """Implement subtitle for fake progress bar."""
            self.subtitles.append(value)
            return self

        def next(self):
            """Implement next for fake progress bar."""
            self.next_calls += 1
            return self

        def draw(self):
            """Implement draw for fake progress bar."""
            self.draw_calls += 1
            return self

    progress_calls: list[tuple[int, str, str | None]] = []
    progress_bar = FakeProgressBar()

    def fake_manual_progress(total: int, *, title: str, subtitle: str | None = None):
        """Support the surrounding tests with fake manual progress."""
        progress_calls.append((total, title, subtitle))
        progress_bar.title(title)
        if subtitle is not None:
            progress_bar.subtitle(subtitle)
        return progress_bar

    monkeypatch.setattr(
        "evalution.engines.transformers_common.manual_progress",
        fake_manual_progress,
    )

    session = TransformersSession(
        config=Transformers(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
    )

    outputs = session.loglikelihood(
        [
            LoglikelihoodRequest(
                context_input_ids=[5],
                continuation_input_ids=[6],
                metadata={"_evalution_loglikelihood_progress_title": "mmlu_stem: scoring answer choices"},
            ),
            LoglikelihoodRequest(
                context_input_ids=[7],
                continuation_input_ids=[8],
                metadata={"_evalution_loglikelihood_progress_title": "mmlu_stem: scoring answer choices"},
            ),
        ],
        batch_size=1,
    )

    assert len(outputs) == 2
    assert progress_calls == [(2, "mmlu_stem: scoring answer choices", "batch_size=1")]
    assert progress_bar.titles == ["mmlu_stem: scoring answer choices"]
    assert progress_bar.subtitles == [
        "batch_size=1",
        "batch=1/2 batch_size=1",
        "batch=2/2 batch_size=1",
    ]
    assert progress_bar.next_calls == 2
    assert progress_bar.draw_calls == 2


def test_transformer_session_loglikelihood_continuous_scores_streamed_requests() -> None:
    """Verify transformer session loglikelihood continuous scores streamed requests. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            """Implement pad for fake tokenizer."""
            assert return_tensors == "pt"
            assert padding is True
            sequences = [list(ids) for ids in encoded_inputs["input_ids"]]
            max_length = max(len(ids) for ids in sequences)
            padded_ids = []
            attention_masks = []
            for ids in sequences:
                pad_count = max_length - len(ids)
                padded_ids.append(ids + ([0] * pad_count))
                attention_masks.append(([1] * len(ids)) + ([0] * pad_count))
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = SimpleNamespace(max_position_embeddings=8)

        def __call__(self, *, input_ids, attention_mask=None):
            """Implement call for fake model. Keep the nested traversal explicit so ordering and metadata stay aligned."""
            del attention_mask
            batch_size, sequence_length = input_ids.shape
            vocab_size = 16
            logits = torch.full((batch_size, sequence_length, vocab_size), -2.0)
            for row in range(batch_size):
                for position in range(sequence_length - 1):
                    next_token = int(input_ids[row, position + 1].item())
                    logits[row, position, next_token] = 3.0
            return SimpleNamespace(logits=logits)

    session = TransformersSession(
        config=Transformers(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
    )

    outputs = list(
        session.loglikelihood_continuous(
            (
                (
                    request_index,
                    LoglikelihoodRequest(
                        context_input_ids=[5 + request_index],
                        continuation_input_ids=[7 + request_index],
                    ),
                )
                for request_index in range(2)
            ),
            batch_size=1,
        )
    )

    expected_token_logprob = float(
        torch.log_softmax(
            torch.tensor([3.0] + ([-2.0] * 15), dtype=torch.float32),
            dim=0,
        )[0].item()
    )

    assert [request_key for request_key, _output in outputs] == [0, 1]
    assert all(output.is_greedy is True for _request_key, output in outputs)
    assert [output.token_count for _request_key, output in outputs] == [1, 1]
    assert all(output.metadata == {} for _request_key, output in outputs)
    assert [output.logprob for _request_key, output in outputs] == [
        pytest.approx(expected_token_logprob, abs=1e-6),
        pytest.approx(expected_token_logprob, abs=1e-6),
    ]


def test_transformer_session_loglikelihood_continuous_with_explicit_batch_size_limits_preview() -> None:
    """Verify transformer session loglikelihood continuous with explicit batch size limits preview."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            """Implement pad for fake tokenizer."""
            assert return_tensors == "pt"
            assert padding is True
            sequences = [list(ids) for ids in encoded_inputs["input_ids"]]
            max_length = max(len(ids) for ids in sequences)
            padded_ids = []
            attention_masks = []
            for ids in sequences:
                pad_count = max_length - len(ids)
                padded_ids.append(ids + ([0] * pad_count))
                attention_masks.append(([1] * len(ids)) + ([0] * pad_count))
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = SimpleNamespace(max_position_embeddings=8)

        def __call__(self, *, input_ids, attention_mask=None):
            """Implement call for fake model. Keep the nested traversal explicit so ordering and metadata stay aligned."""
            del attention_mask
            batch_size, sequence_length = input_ids.shape
            vocab_size = 16
            logits = torch.full((batch_size, sequence_length, vocab_size), -2.0)
            for row in range(batch_size):
                for position in range(sequence_length - 1):
                    next_token = int(input_ids[row, position + 1].item())
                    logits[row, position, next_token] = 3.0
            return SimpleNamespace(logits=logits)

    session = TransformersSession(
        config=Transformers(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
    )

    produced_count = {"value": 0}

    def iter_requests():
        """Iterate over requests."""
        for request_index in range(100):
            produced_count["value"] += 1
            yield (
                request_index,
                LoglikelihoodRequest(
                    context_input_ids=[5 + request_index],
                    continuation_input_ids=[7 + request_index],
                ),
            )

    iterator = session.loglikelihood_continuous(
        iter_requests(),
        batch_size=1,
    )

    assert next(iterator)[0] == 0
    # The producer may stay a few requests ahead of the first yielded result because the
    # background consumer can drain the bounded queue concurrently. The important regression check
    # is that explicit batch sizing still keeps prefetching tightly bounded instead of materializing
    # the full upstream iterator.
    assert produced_count["value"] <= 8
    iterator.close()


def test_transformer_session_loglikelihood_uses_logits_to_keep_for_single_chunk_scoring() -> None:
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"
        model_max_length = 32

        def __call__(self, text, add_special_tokens=False, **kwargs):
            del add_special_tokens, kwargs
            if isinstance(text, str):
                if text == "Prompt":
                    return {"input_ids": [10, 11, 12, 13]}
                return {"input_ids": [17]}
            return {"input_ids": [[10, 11, 12, 13] for _ in text]}

        def decode(self, token_ids, *, skip_special_tokens=False):
            del token_ids, skip_special_tokens
            return "<bos>"

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(max_position_embeddings=32)
            self.logits_to_keep_calls: list[int] = []

        def forward(self, *, input_ids, logits_to_keep=0):
            self.logits_to_keep_calls.append(int(logits_to_keep))
            batch_size, sequence_length = input_ids.shape
            kept_length = logits_to_keep if logits_to_keep else sequence_length
            vocab_size = 32
            logits = torch.full((batch_size, kept_length, vocab_size), -5.0)
            logits[0, 0, int(input_ids[0, -1].item())] = 5.0
            return SimpleNamespace(logits=logits)

        __call__ = forward

    session = TransformersSession(
        config=Transformers(batch_size=1),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
    )

    outputs = session.loglikelihood(
        [
            LoglikelihoodRequest(
                context="Prompt",
                continuation=" answer",
            )
        ],
        batch_size=1,
    )

    assert len(outputs) == 1
    assert outputs[0].token_count == 1
    assert session.model.logits_to_keep_calls == [2]


def test_transformer_session_loglikelihood_rolling_scores_chunked_text() -> None:
    """Verify transformer session loglikelihood rolling scores chunked text. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "left"

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            """Implement pad for fake tokenizer."""
            assert return_tensors == "pt"
            assert padding is True
            sequences = [list(ids) for ids in encoded_inputs["input_ids"]]
            max_length = max(len(ids) for ids in sequences)
            padded_ids = []
            attention_masks = []
            for ids in sequences:
                pad_count = max_length - len(ids)
                padded_ids.append(([0] * pad_count) + ids)
                attention_masks.append(([0] * pad_count) + ([1] * len(ids)))
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = SimpleNamespace(max_position_embeddings=4)
            self.calls = 0

        def __call__(self, *, input_ids, attention_mask=None):
            """Implement call for fake model. Keep the nested traversal explicit so ordering and metadata stay aligned."""
            del attention_mask
            self.calls += 1
            batch_size, sequence_length = input_ids.shape
            vocab_size = 16
            logits = torch.full((batch_size, sequence_length, vocab_size), -2.0)
            for row in range(batch_size):
                for position in range(sequence_length - 1):
                    next_token = int(input_ids[row, position + 1].item())
                    logits[row, position, next_token] = 3.0
            return SimpleNamespace(logits=logits)

    model = FakeModel()
    session = TransformersSession(
        config=Transformers(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=model,
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
    )

    outputs = session.loglikelihood_rolling(
        [
            RollingLoglikelihoodRequest(
                text="unused",
                input_ids=[2, 3, 4, 5],
            )
        ],
        batch_size=2,
    )

    expected_token_logprob = float(
        torch.log_softmax(
            torch.tensor([3.0] + ([-2.0] * 15), dtype=torch.float32),
            dim=0,
        )[0].item()
    )

    assert len(outputs) == 1
    assert outputs[0].token_count == 4
    assert outputs[0].logprob == pytest.approx(expected_token_logprob * 4, abs=1e-6)
    assert model.calls == 1


def test_transformer_session_loglikelihood_temporarily_restores_base_attention() -> None:
    """Verify transformer session loglikelihood temporarily restores base attention."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            """Implement pad for fake tokenizer."""
            assert return_tensors == "pt"
            assert padding is True
            return {
                "input_ids": torch.tensor([[1, 4]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = SimpleNamespace(max_position_embeddings=8)
            self.attn_history: list[str] = []

        def set_attn_implementation(self, value: str) -> None:
            """Implement set attn implementation for fake model."""
            self.attn_history.append(value)

        def __call__(self, *, input_ids, attention_mask=None):
            """Implement call for fake model."""
            del attention_mask
            logits = torch.full((1, input_ids.shape[1], 16), -2.0)
            logits[0, 0, 4] = 3.0
            return SimpleNamespace(logits=logits)

    model = FakeModel()
    session = TransformersSession(
        config=Transformers(batch_size=2, attn_implementation="paged|flash_attention_2"),
        model_config=Model(path="/tmp/model"),
        model=model,
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    outputs = session.loglikelihood(
        [LoglikelihoodRequest(continuation_input_ids=[4])],
        batch_size=1,
    )

    assert len(outputs) == 1
    assert outputs[0].is_greedy is True
    assert model.attn_history == ["flash_attention_2", "paged|flash_attention_2"]


def test_transformer_session_reuses_continuous_batching_manager_for_matching_signature(
    monkeypatch,
) -> None:
    """Verify transformer session reuses continuous batching manager for matching signature."""
    import transformers

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
                return {"input_ids": [11, 12, 13]}
            return {"input_ids": [[11, 12, 13] for _ in prompts]}

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

        def set_attn_implementation(self, value: str) -> None:
            """Implement set attn implementation for fake model."""
            self.config._attn_implementation = value

    class FakeContinuousBatchingManager:
        """Provide the fake continuous batching manager helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        instances: list[FakeContinuousBatchingManager] = []

        def __init__(self, model, generation_config, **kwargs):
            """Initialize this object."""
            del model, generation_config, kwargs
            self.started = False
            self.stopped = False
            self.result_queue: list[FakeContinuousOutput] = []
            self.added_request_ids: list[str] = []
            FakeContinuousBatchingManager.instances.append(self)

        def start(self) -> None:
            """Implement start for fake continuous batching manager."""
            self.started = True

        def add_request(self, input_ids, *, request_id=None, max_new_tokens=None, streaming=False):
            """Implement add request for fake continuous batching manager."""
            del input_ids, max_new_tokens, streaming
            assert request_id is not None
            self.added_request_ids.append(request_id)
            self.result_queue.append(FakeContinuousOutput(request_id, [101, 102, 103]))

        def get_result(self, timeout=None):
            """Get result."""
            del timeout
            if not self.result_queue:
                return None
            return self.result_queue.pop(0)

        def is_running(self) -> bool:
            """Implement is running for fake continuous batching manager."""
            return self.started and not self.stopped

        def stop(self, block=True):
            """Implement stop for fake continuous batching manager."""
            assert block is True
            self.stopped = True

    monkeypatch.setattr(
        transformers,
        "ContinuousBatchingManager",
        FakeContinuousBatchingManager,
    )

    session = TransformersSession(
        config=Transformers(attn_implementation="paged|flash_attention_2"),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    first_outputs = session.generate(
        [GenerationRequest(prompt="Q: 40 + 2\nA:", stop=["Q:"])],
        batch_size=1,
    )
    second_outputs = session.generate(
        [GenerationRequest(prompt="Q: 41 + 2\nA:", stop=["Q:"])],
        batch_size=1,
    )

    assert [output.text for output in first_outputs] == ["The answer is 42."]
    assert [output.text for output in second_outputs] == ["The answer is 42."]
    assert len(FakeContinuousBatchingManager.instances) == 1
    assert FakeContinuousBatchingManager.instances[0].added_request_ids == ["req_0", "req_1"]


def test_transformer_session_rebuilds_continuous_batching_manager_when_signature_changes(
    monkeypatch,
) -> None:
    """Verify transformer session rebuilds continuous batching manager when signature changes."""
    import transformers

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
                return {"input_ids": [11, 12, 13]}
            return {"input_ids": [[11, 12, 13] for _ in prompts]}

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

        def set_attn_implementation(self, value: str) -> None:
            """Implement set attn implementation for fake model."""
            self.config._attn_implementation = value

    class FakeContinuousBatchingManager:
        """Provide the fake continuous batching manager helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        instances: list[FakeContinuousBatchingManager] = []

        def __init__(self, model, generation_config, **kwargs):
            """Initialize this object."""
            del model, kwargs
            self.generation_config = generation_config
            self.started = False
            self.stopped = False
            self.result_queue: list[FakeContinuousOutput] = []
            FakeContinuousBatchingManager.instances.append(self)

        def start(self) -> None:
            """Implement start for fake continuous batching manager."""
            self.started = True

        def add_request(self, input_ids, *, request_id=None, max_new_tokens=None, streaming=False):
            """Implement add request for fake continuous batching manager."""
            del input_ids, max_new_tokens, streaming
            assert request_id is not None
            self.result_queue.append(FakeContinuousOutput(request_id, [101, 102, 103]))

        def get_result(self, timeout=None):
            """Get result."""
            del timeout
            if not self.result_queue:
                return None
            return self.result_queue.pop(0)

        def is_running(self) -> bool:
            """Implement is running for fake continuous batching manager."""
            return self.started and not self.stopped

        def stop(self, block=True):
            """Implement stop for fake continuous batching manager."""
            assert block is True
            self.stopped = True

    monkeypatch.setattr(
        transformers,
        "ContinuousBatchingManager",
        FakeContinuousBatchingManager,
    )

    session = TransformersSession(
        config=Transformers(attn_implementation="paged|flash_attention_2"),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    session.generate([GenerationRequest(prompt="Q: 40 + 2\nA:", stop=["Q:"])], batch_size=1)
    session.generate([GenerationRequest(prompt="Q: 41 + 2\nA:", stop=["A:"])], batch_size=1)

    assert len(FakeContinuousBatchingManager.instances) == 2
    assert FakeContinuousBatchingManager.instances[0].stopped is True
    assert FakeContinuousBatchingManager.instances[1].stopped is False
    assert FakeContinuousBatchingManager.instances[0].generation_config.stop_strings == ["Q:"]
    assert FakeContinuousBatchingManager.instances[1].generation_config.stop_strings == ["A:"]


def test_transformer_sessions_do_not_share_continuous_batching_managers(monkeypatch) -> None:
    """Verify transformer sessions do not share continuous batching managers."""
    import transformers

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
                return {"input_ids": [11, 12, 13]}
            return {"input_ids": [[11, 12, 13] for _ in prompts]}

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
        def __init__(self, *, tag: str) -> None:
            """Initialize this object."""
            self.tag = tag
            self.config = PretrainedConfig()
            self.config._attn_implementation = "flash_attention_2"

        def set_attn_implementation(self, value: str) -> None:
            """Implement set attn implementation for fake model."""
            self.config._attn_implementation = value

    class FakeContinuousBatchingManager:
        """Provide the fake continuous batching manager helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        instances: list[FakeContinuousBatchingManager] = []

        def __init__(self, model, generation_config, **kwargs):
            """Initialize this object."""
            del generation_config, kwargs
            self.model = model
            self.started = False
            self.stopped = False
            self.result_queue: list[FakeContinuousOutput] = []
            self.added_request_ids: list[str] = []
            FakeContinuousBatchingManager.instances.append(self)

        def start(self) -> None:
            """Implement start for fake continuous batching manager."""
            self.started = True

        def add_request(self, input_ids, *, request_id=None, max_new_tokens=None, streaming=False):
            """Implement add request for fake continuous batching manager."""
            del input_ids, max_new_tokens, streaming
            assert request_id is not None
            self.added_request_ids.append(request_id)
            self.result_queue.append(FakeContinuousOutput(request_id, [101, 102, 103]))

        def get_result(self, timeout=None):
            """Get result."""
            del timeout
            if not self.result_queue:
                return None
            return self.result_queue.pop(0)

        def is_running(self) -> bool:
            """Implement is running for fake continuous batching manager."""
            return self.started and not self.stopped

        def stop(self, block=True):
            """Implement stop for fake continuous batching manager."""
            assert block is True
            self.stopped = True

    monkeypatch.setattr(
        transformers,
        "ContinuousBatchingManager",
        FakeContinuousBatchingManager,
    )

    left_session = TransformersSession(
        config=Transformers(attn_implementation="paged|flash_attention_2"),
        model_config=Model(path="/tmp/model-left"),
        model=FakeModel(tag="left"),
        tokenizer=FakeTokenizer(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )
    right_session = TransformersSession(
        config=Transformers(attn_implementation="paged|flash_attention_2"),
        model_config=Model(path="/tmp/model-right"),
        model=FakeModel(tag="right"),
        tokenizer=FakeTokenizer(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    left_outputs = left_session.generate(
        [GenerationRequest(prompt="Q: left\nA:", stop=["Q:"])],
        batch_size=1,
    )
    right_outputs = right_session.generate(
        [GenerationRequest(prompt="Q: right\nA:", stop=["Q:"])],
        batch_size=1,
    )

    assert [output.text for output in left_outputs] == ["The answer is 42."]
    assert [output.text for output in right_outputs] == ["The answer is 42."]
    assert len(FakeContinuousBatchingManager.instances) == 2
    left_manager, right_manager = FakeContinuousBatchingManager.instances
    assert left_manager is left_session.continuous_batching_manager
    assert right_manager is right_session.continuous_batching_manager
    assert left_manager is not right_manager
    assert left_manager.model is left_session.model
    assert right_manager.model is right_session.model
    assert left_manager.added_request_ids == ["req_0"]
    assert right_manager.added_request_ids == ["req_0"]


def test_transformer_session_serializes_shared_tokenizer_access_between_prepare_and_generate() -> None:
    """Verify transformer session serializes shared tokenizer access between prepare and generate."""
    class BlockingTokenizer:
        """Define the blocking tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1

        def __init__(self) -> None:
            """Initialize this object."""
            self._state_lock = threading.Lock()
            self._active_calls = 0
            self.overlap_detected = False
            self.prepare_started = threading.Event()
            self.pad_started = threading.Event()
            self.allow_prepare_finish = threading.Event()

        def _enter(self) -> None:
            """Implement enter for blocking tokenizer."""
            with self._state_lock:
                self._active_calls += 1
                if self._active_calls > 1:
                    self.overlap_detected = True

        def _exit(self) -> None:
            """Implement exit for blocking tokenizer."""
            with self._state_lock:
                self._active_calls -= 1

        def __call__(self, prompts, *, add_special_tokens=False, padding=False, **kwargs):
            """Implement call for blocking tokenizer."""
            del kwargs
            assert add_special_tokens is False
            assert padding is False
            assert isinstance(prompts, list)
            self._enter()
            try:
                self.prepare_started.set()
                assert self.allow_prepare_finish.wait(timeout=1.0)
                return {"input_ids": [[1, 2, 3] for _ in prompts]}
            finally:
                self._exit()

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            """Implement pad for blocking tokenizer."""
            del encoded_inputs
            assert return_tensors == "pt"
            assert padding is True
            self._enter()
            try:
                self.pad_started.set()
                return {
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                }
            finally:
                self._exit()

        def decode(self, token_ids, *, skip_special_tokens=False):
            """Implement decode for blocking tokenizer."""
            del token_ids, skip_special_tokens
            return "The answer is 42."

    class FakeModel:
        """Provide the fake model helper used by the surrounding tests."""
        def generate(self, **kwargs):
            """Generate generate."""
            del kwargs
            return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    tokenizer = BlockingTokenizer()
    session = TransformersSession(
        config=Transformers(),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=tokenizer,
        prepare_tokenizer=None,
        input_device=torch.device("cpu"),
    )

    prepare_errors: list[BaseException] = []
    generate_errors: list[BaseException] = []

    def run_prepare() -> None:
        """Run prepare."""
        try:
            session.prepare_requests([GenerationRequest(prompt="Q: 1 + 1\nA:")])
        except BaseException as exc:  # pragma: no cover - asserted below
            prepare_errors.append(exc)

    def run_generate() -> None:
        """Run generate."""
        try:
            session.generate(
                [
                    GenerationRequest(
                        prompt="Q: 40 + 2\nA:",
                        rendered_prompt="Q: 40 + 2\nA:",
                        input_ids=[1, 2, 3],
                    )
                ],
                batch_size=1,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            generate_errors.append(exc)

    prepare_thread = threading.Thread(target=run_prepare)
    generate_thread = threading.Thread(target=run_generate)

    prepare_thread.start()
    assert tokenizer.prepare_started.wait(timeout=1.0)
    generate_thread.start()

    assert not tokenizer.pad_started.wait(timeout=0.1)

    tokenizer.allow_prepare_finish.set()
    prepare_thread.join(timeout=1.0)
    generate_thread.join(timeout=1.0)

    assert not prepare_errors
    assert not generate_errors
    assert tokenizer.overlap_detected is False


def test_transformer_session_serializes_generate_calls() -> None:
    """Verify transformer session serializes generate calls."""
    class FakeTokenizer:
        """Provide the fake tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1

        def pad(self, encoded_inputs, *, return_tensors="pt", padding=True):
            """Implement pad for fake tokenizer."""
            del encoded_inputs
            assert return_tensors == "pt"
            assert padding is True
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

        def decode(self, token_ids, *, skip_special_tokens=False):
            """Implement decode for fake tokenizer."""
            del token_ids, skip_special_tokens
            return "The answer is 42."

    class BlockingModel:
        """Define the blocking model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self._state_lock = threading.Lock()
            self._active_calls = 0
            self._call_count = 0
            self.overlap_detected = False
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.allow_first_finish = threading.Event()

        def generate(self, **kwargs):
            """Generate generate."""
            del kwargs
            with self._state_lock:
                self._active_calls += 1
                self._call_count += 1
                call_number = self._call_count
                if self._active_calls > 1:
                    self.overlap_detected = True
            try:
                if call_number == 1:
                    self.first_started.set()
                    assert self.allow_first_finish.wait(timeout=1.0)
                else:
                    self.second_started.set()
                return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
            finally:
                with self._state_lock:
                    self._active_calls -= 1

    model = BlockingModel()
    session = TransformersSession(
        config=Transformers(),
        model_config=Model(path="/tmp/model"),
        model=model,
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
    )
    requests = [
        GenerationRequest(
            prompt="Q: 40 + 2\nA:",
            rendered_prompt="Q: 40 + 2\nA:",
            input_ids=[1, 2, 3],
        )
    ]
    errors: list[BaseException] = []

    def run_generate() -> None:
        """Run generate."""
        try:
            session.generate(requests, batch_size=1)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first_thread = threading.Thread(target=run_generate)
    second_thread = threading.Thread(target=run_generate)

    first_thread.start()
    assert model.first_started.wait(timeout=1.0)

    second_thread.start()
    assert not model.second_started.wait(timeout=0.1)

    model.allow_first_finish.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert not errors
    assert model.overlap_detected is False


@pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="requires Python free-threading with GIL disabled",
)
def test_transformer_session_allows_nogil_prepare_overlap_with_continuous_batching(
    monkeypatch,
) -> None:
    """Verify transformer session allows no-GIL prepare overlap with continuous batching."""
    import transformers

    class PrepareTokenizer:
        """Define the prepare tokenizer helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.prepare_started = threading.Event()
            self.allow_prepare_finish = threading.Event()

        def __call__(self, prompts, *, add_special_tokens=False, padding=False, **kwargs):
            """Implement call for prepare tokenizer."""
            del kwargs
            assert add_special_tokens is False
            assert padding is False
            assert isinstance(prompts, list)
            self.prepare_started.set()
            assert self.allow_prepare_finish.wait(timeout=1.0)
            return {"input_ids": [[1, 2, 3] for _ in prompts]}

        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
            """Implement apply chat template for prepare tokenizer."""
            assert tokenize is False
            assert add_generation_prompt is True
            return f"chat::{messages[0]['content']}"

    class GenerateTokenizer:
        """Define the generate tokenizer helper used by the surrounding tests."""
        # Keep the class-level test state explicit for the surrounding assertions.
        pad_token_id = 0
        eos_token_id = 1

        def decode(self, token_ids, *, skip_special_tokens=False):
            """Implement decode for generate tokenizer."""
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

    class BlockingContinuousBatchingManager:
        """Define the blocking continuous batching manager helper used by the surrounding tests."""
        def __init__(self, model, generation_config):
            """Initialize this object."""
            del model, generation_config
            self.generate_started = threading.Event()
            self.allow_generate_finish = threading.Event()
            self.request_id: str | None = None
            self.result_returned = False

        def start(self) -> None:
            """Implement start for blocking continuous batching manager."""
            return None

        def add_request(self, input_ids, *, request_id=None, max_new_tokens=None, streaming=False):
            """Implement add request for blocking continuous batching manager."""
            del max_new_tokens, streaming
            assert input_ids == [1, 2, 3]
            assert request_id is not None
            self.request_id = request_id
            self.generate_started.set()

        def get_result(self, timeout=None):
            """Get result."""
            del timeout
            assert self.allow_generate_finish.wait(timeout=1.0)
            if self.result_returned:
                return None
            self.result_returned = True
            assert self.request_id is not None
            return FakeContinuousOutput(self.request_id, [101, 102, 103])

        def is_running(self) -> bool:
            """Implement is running for blocking continuous batching manager."""
            return True

        def stop(self, block=True):
            """Implement stop for blocking continuous batching manager."""
            del block
            return None

    class BlockingPagedModel:
        """Define the blocking paged model helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.config = PretrainedConfig()
            self.config._attn_implementation = "flash_attention_2"

        def set_attn_implementation(self, value: str) -> None:
            """Implement set attn implementation for blocking paged model."""
            self.config._attn_implementation = value

    manager = BlockingContinuousBatchingManager(None, None)
    monkeypatch.setattr(
        transformers,
        "ContinuousBatchingManager",
        lambda model, generation_config, **kwargs: manager,
    )

    prepare_tokenizer = PrepareTokenizer()
    model = BlockingPagedModel()
    session = TransformersSession(
        config=Transformers(attn_implementation="paged|flash_attention_2"),
        model_config=Model(path="/tmp/model"),
        model=model,
        tokenizer=GenerateTokenizer(),
        prepare_tokenizer=prepare_tokenizer,
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|flash_attention_2",
        effective_attn_implementation="paged|flash_attention_2",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    prepare_errors: list[BaseException] = []
    generate_errors: list[BaseException] = []

    def run_prepare() -> None:
        """Run prepare."""
        try:
            session.prepare_requests([GenerationRequest(prompt="Q: 1 + 1\nA:")])
        except BaseException as exc:  # pragma: no cover - asserted below
            prepare_errors.append(exc)

    def run_generate() -> None:
        """Run generate."""
        try:
            session.generate(
                [
                    GenerationRequest(
                        prompt="Q: 40 + 2\nA:",
                        rendered_prompt="Q: 40 + 2\nA:",
                        input_ids=[1, 2, 3],
                    )
                ],
                batch_size=1,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            generate_errors.append(exc)

    prepare_thread = threading.Thread(target=run_prepare)
    generate_thread = threading.Thread(target=run_generate)

    prepare_thread.start()
    assert prepare_tokenizer.prepare_started.wait(timeout=1.0)

    generate_thread.start()
    assert manager.generate_started.wait(timeout=1.0)

    manager.allow_generate_finish.set()
    prepare_tokenizer.allow_prepare_finish.set()
    prepare_thread.join(timeout=1.0)
    generate_thread.join(timeout=1.0)

    assert not prepare_errors
    assert not generate_errors
    assert not prepare_thread.is_alive()
    assert not generate_thread.is_alive()


def test_transformer_session_falls_back_to_standard_generate_when_paged_generation_fails(monkeypatch) -> None:
    """Verify transformer session falls back to standard generate when paged generation fails."""
    model = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa"),
        set_attn_implementation=lambda value: None,
    )
    session = TransformersSession(
        config=Transformers(attn_implementation="paged|sdpa"),
        model_config=Model(path="/tmp/model"),
        model=model,
        tokenizer=SimpleNamespace(),
        input_device=SimpleNamespace(type="cuda"),
        requested_attn_implementation="paged|sdpa",
        effective_attn_implementation="paged|sdpa",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )
    requests = [GenerationRequest(prompt=f"Q: {index}\nA:") for index in range(4)]
    calls: dict[str, object] = {}

    def fake_generate_paged(self, batch, *, batch_size):
        """Support the surrounding tests with fake generate paged."""
        del self, batch, batch_size
        raise RuntimeError("paged failure")

    def fake_generate_standard(self, batch, *, batch_size):
        """Support the surrounding tests with fake generate standard."""
        calls["batch_size"] = batch_size
        return [
            GenerationOutput(
                prompt=request.prompt or "",
                text="The answer is 42.",
            )
            for request in batch
        ]

    monkeypatch.setattr(TransformersSession, "_generate_paged", fake_generate_paged)
    monkeypatch.setattr(TransformersSession, "_generate_standard", fake_generate_standard)

    outputs = session.generate(requests, batch_size=4)

    assert len(outputs) == 4
    assert calls["batch_size"] == 2
    assert session.paged_attention_enabled is False
    assert session.generation_backend == "generate"
    assert session.effective_attn_implementation == "sdpa"
    assert session.standard_batch_size_cap == 2


def test_transformer_session_standard_generate_uses_effective_attention_backend() -> None:
    import torch

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "left"

        def __call__(self, prompts, **kwargs):
            del kwargs
            import torch

            if isinstance(prompts, str):
                return {"input_ids": torch.tensor([11, 12, 13], dtype=torch.long)}
            return {"input_ids": torch.tensor([[11, 12, 13] for _ in prompts], dtype=torch.long)}

        def pad(self, batch, return_tensors="pt", padding=True):
            del return_tensors, padding
            import torch

            return {"input_ids": torch.tensor(batch["input_ids"], dtype=torch.long)}

        def decode(self, token_ids, *, skip_special_tokens=False):
            del token_ids, skip_special_tokens
            return "The answer is 42."

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(_attn_implementation="flash_attention_2")
            self.attn_history: list[str] = []

        def set_attn_implementation(self, value: str) -> None:
            self.attn_history.append(value)
            self.config._attn_implementation = value

        def generate(self, **kwargs):
            del kwargs
            assert self.config._attn_implementation == "sdpa"
            import torch

            return torch.tensor([[11, 12, 13, 42]], dtype=torch.long)

    session = TransformersSession(
        config=Transformers(attn_implementation="paged|flash_attention_2"),
        model_config=Model(path="/tmp/model"),
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        input_device=torch.device("cpu"),
        requested_attn_implementation="paged|sdpa",
        effective_attn_implementation="paged|sdpa",
        paged_attention_enabled=True,
        generation_backend="continuous_batching",
    )

    outputs = session._generate_standard([GenerationRequest(prompt="Q: 40 + 2\nA:")], batch_size=1)

    assert [output.text for output in outputs] == ["The answer is 42."]
    assert session.model.attn_history == ["sdpa", "flash_attention_2"]
