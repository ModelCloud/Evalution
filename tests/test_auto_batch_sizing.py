# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from evalution.engines.memory import (
    auto_batch_bytes_per_token,
    build_memory_profile,
    prefill_logits_bytes_per_token,
    supports_logits_to_keep,
)
from evalution.engines.transformers_common import BaseTransformerSession

# Qwen2.5-0.5B-Instruct: the checkpoint that used to resolve to 384-768 requests.
_QWEN2_5_0_5B_CONFIG = SimpleNamespace(
    vocab_size=151936,
    hidden_size=896,
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
    head_dim=64,
)
_QWEN2_5_0_5B_KV_BYTES_PER_TOKEN = 2 * 24 * 2 * 64 * 2
_QWEN2_5_0_5B_LOGITS_BYTES_PER_TOKEN = 151936 * 4


class _OptimizedModel:
    """Native checkpoint whose forward accepts `logits_to_keep`."""

    dtype = torch.bfloat16
    config = _QWEN2_5_0_5B_CONFIG

    def forward(self, input_ids, logits_to_keep=0, **kwargs):
        del input_ids, logits_to_keep, kwargs

    def num_parameters(self) -> int:
        return 500_000_000


class _QuantizedWrapper:
    """Quantized wrapper that swallows kwargs, so prefill keeps every logit."""

    dtype = torch.bfloat16
    config = _QWEN2_5_0_5B_CONFIG

    def forward(self, **kwargs):
        del kwargs

    def num_parameters(self) -> int:
        return 500_000_000


def test_supports_logits_to_keep_matches_forward_signature() -> None:
    """The check mirrors what transformers does before passing `logits_to_keep`."""
    assert supports_logits_to_keep(_OptimizedModel())
    assert not supports_logits_to_keep(_QuantizedWrapper())
    assert not supports_logits_to_keep(SimpleNamespace())


def test_prefill_logits_are_counted_when_forward_cannot_slice() -> None:
    """Quantized wrappers materialize `[batch, prompt, vocab]` logits during prefill."""
    assert prefill_logits_bytes_per_token(_QuantizedWrapper()) == _QWEN2_5_0_5B_LOGITS_BYTES_PER_TOKEN


def test_prefill_logits_are_zero_when_generate_slices_to_last_token() -> None:
    """Native models keep only the last-token logits, so there is nothing to add."""
    assert prefill_logits_bytes_per_token(_OptimizedModel()) == 0


def test_auto_batch_cost_adds_kv_cache_and_prefill_logits() -> None:
    """The per-token estimate must cover both resident terms for a batch."""
    assert auto_batch_bytes_per_token(_QuantizedWrapper(), dtype_bytes=2) == (
        _QWEN2_5_0_5B_KV_BYTES_PER_TOKEN + _QWEN2_5_0_5B_LOGITS_BYTES_PER_TOKEN
    )
    assert auto_batch_bytes_per_token(_OptimizedModel(), dtype_bytes=2) == _QWEN2_5_0_5B_KV_BYTES_PER_TOKEN


def test_auto_batch_cost_falls_back_to_logits_without_kv_config() -> None:
    """A partial config still contributes the logits term instead of returning None."""
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=32000))
    assert auto_batch_bytes_per_token(model, dtype_bytes=2) == 32000 * 4


def test_auto_batch_cost_is_unknown_without_config() -> None:
    """Without a config there is nothing to estimate; keep the previous behavior."""
    assert auto_batch_bytes_per_token(SimpleNamespace(), dtype_bytes=2) is None


def test_memory_profile_reports_the_combined_auto_batch_cost(monkeypatch) -> None:
    """`build_memory_profile` feeds the combined value into auto batch sizing."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=32 * 1024**3),
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (32 * 1024**3, 32 * 1024**3))

    profile = build_memory_profile(
        _QuantizedWrapper(),
        input_device=torch.device("cuda:0"),
        configured_dtype="bfloat16",
    )

    assert profile.total_vram_gib == 32.0
    assert profile.kv_cache_bytes_per_token == (
        _QWEN2_5_0_5B_KV_BYTES_PER_TOKEN + _QWEN2_5_0_5B_LOGITS_BYTES_PER_TOKEN
    )


def _qwen_stats(*, bytes_per_token: int) -> dict[str, object]:
    return {
        "row_count": 512,
        "min_prompt_tokens": 400,
        "avg_prompt_tokens": 500.0,
        "max_prompt_tokens": 512,
        "max_new_tokens": 256,
        "max_num_beams": 1,
        "dtype_name": "bfloat16",
        "dtype_bytes": 2,
        "total_vram_gib": 32.0,
        "free_vram_gib": 32.0,
        "parameter_count_billions": 0.5,
        "kv_cache_bytes_per_token": bytes_per_token,
    }


def test_quantized_qwen_batch_is_bounded_on_32gib() -> None:
    """Counting prefill logits drops the estimate from ~80 to the known-safe range."""
    session = SimpleNamespace(input_device=SimpleNamespace(type="cuda"))
    stats = _qwen_stats(
        bytes_per_token=auto_batch_bytes_per_token(_QuantizedWrapper(), dtype_bytes=2),
    )

    assert BaseTransformerSession._estimate_auto_batch_size(session, stats) == 12


def test_native_qwen_keeps_the_token_budget_batch() -> None:
    """Native models still slice logits, so the existing throughput estimate stays."""
    session = SimpleNamespace(input_device=SimpleNamespace(type="cuda"))
    stats = _qwen_stats(
        bytes_per_token=auto_batch_bytes_per_token(_OptimizedModel(), dtype_bytes=2),
    )

    assert BaseTransformerSession._estimate_auto_batch_size(session, stats) == 80


@pytest.mark.parametrize("bytes_per_token", [None])
def test_missing_kv_config_keeps_the_token_budget_path(bytes_per_token) -> None:
    """The old estimate still applies when neither term can be resolved."""
    session = SimpleNamespace(input_device=SimpleNamespace(type="cuda"))
    stats = _qwen_stats(bytes_per_token=bytes_per_token)

    assert BaseTransformerSession._estimate_auto_batch_size(session, stats) == 80
