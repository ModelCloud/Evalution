# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import evalution

from tests.models_support import (
    LLAMA3_2_LLAMACPP_TEST_MARKS,
    assert_metrics_match_baseline,
    run_llama3_2_llamacpp_suite,
)

# Keep the shared benchmark settings explicit so the llama.cpp baseline stays reproducible.
pytestmark = LLAMA3_2_LLAMACPP_TEST_MARKS


def test_llama3_2_llamacpp_gsm8k_platinum_full_model_eval(capsys):
    """Verify llama3 2 llama.cpp GSM8K Platinum full-model evaluation and baseline."""

    # Keep the native llama.cpp continuous-batching path on a practical slow-test budget while still
    # pinning a real benchmark score against the shared Llama 3.2 GGUF fixture.
    suite = evalution.benchmarks.gsm8k_platinum(
        variant="cot",
        apply_chat_template=True,
        batch_size=24,
        max_new_tokens=96,
        stream=True,
        max_rows=32,
    )
    result, test_result = run_llama3_2_llamacpp_suite(capsys, suite)

    assert test_result.name == "gsm8k_platinum_cot"
    assert test_result.metadata["variant"] == "cot"
    assert test_result.metadata["apply_chat_template"] is True
    assert test_result.metadata["stream"] is True
    assert test_result.metadata["generation_submission_mode"] == "continuous_refill"
    assert len(test_result.samples) == 32
    assert set(test_result.metrics) == {"acc,num"}
    assert_metrics_match_baseline(
        test_result.metrics,
        {"acc,num": 0.34375},
        abs_tolerance=2 / 32,
    )
    assert result.engine["execution"]["continuous_batching"] is True
