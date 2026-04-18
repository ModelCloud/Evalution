# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pytest

import evalution

from tests.models_support import (
    LLAMA3_2_1B_INSTRUCT,
    LLAMA3_2_1B_INSTRUCT_GGUF,
    LLAMA3_2_TINYGRAD_TEST_MARKS,
    _select_llama3_2_gpu_baseline,
    assert_metrics_match_baseline,
    run_llama3_2_tinygrad_suite,
)

# Keep the tinygrad model-regression suite on a one-sample GSM8K base-variant budget because CUDA
# runs use the profiled graph-free JIT path and remain much slower than the other backends.
pytestmark = LLAMA3_2_TINYGRAD_TEST_MARKS


def test_llama3_2_tinygrad_native_gsm8k_platinum_full_model_eval(capsys):
    """Verify native tinygrad Llama 3.2 GSM8K Platinum evaluation and pinned baseline."""

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="base",
        apply_chat_template=True,
        batch_size=1,
        max_new_tokens=16,
        max_rows=1,
    )
    result, test_result = run_llama3_2_tinygrad_suite(
        capsys,
        suite,
        path=str(LLAMA3_2_1B_INSTRUCT),
    )

    assert test_result.name == "gsm8k_platinum"
    assert test_result.metadata["variant"] == "base"
    assert test_result.metadata["apply_chat_template"] is True
    assert test_result.metadata["generation_submission_mode"] == "continuous_refill"
    assert len(test_result.samples) == 1
    assert set(test_result.metrics) == {"acc,num"}
    assert_metrics_match_baseline(
        test_result.metrics,
        _select_llama3_2_gpu_baseline(
            default={"acc,num": 0.0},
            rtx4090={"acc,num": 0.0},
            a100={"acc,num": 0.0},
        ),
        abs_tolerance=0.0,
    )
    assert result.engine["execution"]["load_format"] == "native"
    assert result.engine["execution"]["model_type"] == "llama"
    assert result.engine["execution"]["jit"] == 2
    assert result.engine["execution"]["jitbeam"] == 0


@pytest.mark.skipif(
    not LLAMA3_2_1B_INSTRUCT_GGUF.exists(),
    reason="local Llama 3.2 1B Instruct GGUF weights are not available",
)
def test_llama3_2_tinygrad_gguf_gsm8k_platinum_full_model_eval(capsys):
    """Verify GGUF tinygrad Llama 3.2 GSM8K Platinum evaluation and pinned baseline."""

    suite = evalution.benchmarks.gsm8k_platinum(
        variant="base",
        apply_chat_template=True,
        batch_size=1,
        max_new_tokens=16,
        max_rows=1,
    )
    result, test_result = run_llama3_2_tinygrad_suite(
        capsys,
        suite,
        path=str(LLAMA3_2_1B_INSTRUCT_GGUF),
        tokenizer_path=str(LLAMA3_2_1B_INSTRUCT),
    )

    assert test_result.name == "gsm8k_platinum"
    assert test_result.metadata["variant"] == "base"
    assert test_result.metadata["apply_chat_template"] is True
    assert test_result.metadata["generation_submission_mode"] == "continuous_refill"
    assert len(test_result.samples) == 1
    assert set(test_result.metrics) == {"acc,num"}
    assert_metrics_match_baseline(
        test_result.metrics,
        _select_llama3_2_gpu_baseline(
            default={"acc,num": 0.0},
            rtx4090={"acc,num": 0.0},
            a100={"acc,num": 0.0},
        ),
        abs_tolerance=0.0,
    )
    assert result.engine["execution"]["load_format"] == "gguf"
    assert result.engine["execution"]["jit"] == 2
    assert result.engine["execution"]["jitbeam"] == 0
