# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import evalution

from tests.models_support import (
    LLAMA3_2_TRANSFORMERS_COMPARE_TEST_MARKS,
    run_llama3_2_compare_suite,
)

# Keep shared test fixtures and expectations explicit at module scope.
pytestmark = LLAMA3_2_TRANSFORMERS_COMPARE_TEST_MARKS


def test_llama3_2_transformers_gsm8k_platinum_full_model_compare_eval(capsys):
    """Verify llama3 2 transformers GSM8K platinum full model compare eval."""
    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=True,
        batch_size=24,
        max_new_tokens=96,
        stream=False,
        max_rows=128,
    )
    result, compare_test_result = run_llama3_2_compare_suite(capsys, suite)

    assert compare_test_result.name == "gsm8k_platinum_cot"
    assert compare_test_result.left.metadata["stream"] is False
    assert compare_test_result.right.metadata["stream"] is False
    assert compare_test_result.left.metrics["acc,num"] == compare_test_result.metrics["acc,num"].left_value
    assert compare_test_result.right.metrics["acc,num"] == compare_test_result.metrics["acc,num"].right_value
    assert len(compare_test_result.left.samples) == 128
    assert len(compare_test_result.right.samples) == 128
    assert result.left.engine["execution"]["generation_backend"] == "continuous_batching"
    assert result.right.engine["execution"]["generation_backend"] == "continuous_batching"


def test_llama3_2_transformers_gsm8k_platinum_compare_loop_uses_fresh_lane_threads(capsys, monkeypatch):
    """Verify llama3 2 transformers GSM8K platinum compare loop uses fresh lane threads."""
    compare_module = importlib.import_module("evalution.compare")
    original_thread = compare_module.threading.Thread
    created_thread_ids: list[int] = []

    def _recording_thread(*args, **kwargs):
        """Support the surrounding tests with recording thread."""
        thread = original_thread(*args, **kwargs)
        created_thread_ids.append(id(thread))
        return thread

    monkeypatch.setattr(compare_module.threading, "Thread", _recording_thread)

    for _ in range(2):
        suite = evalution.benchmarks.gsm8k_platinum(
            variant="cot",
            apply_chat_template=True,
            batch_size=24,
            max_new_tokens=96,
            stream=False,
            max_rows=128,
        )
        result, compare_test_result = run_llama3_2_compare_suite(capsys, suite)
        assert compare_test_result.name == "gsm8k_platinum_cot"
        assert result.left.engine["execution"]["generation_backend"] == "continuous_batching"
        assert result.right.engine["execution"]["generation_backend"] == "continuous_batching"

    assert len(created_thread_ids) == 4
    assert len(set(created_thread_ids)) == 4
