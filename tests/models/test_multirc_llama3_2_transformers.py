# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from tests.models_support import LLAMA3_2_TRANSFORMERS_TEST_MARKS, run_suite_spec

def test_llama3_2_transformers_multirc_full_model_eval():
    """Verify llama3 2 transformers multirc full model eval."""
    suite = evalution.benchmarks.multirc(batch_size=8, max_rows=16, max_new_tokens=64)
    result = (
        evalution.Transformers(
            dtype="bfloat16",
            attn_implementation="paged|flash_attention_2",
            device="cuda",
            batch_size=8,
        )
        .model(path="/monster/data/model/Llama-3.2-1B-Instruct")
        .run(suite)
        .result()
        .tests[0]
    )
    metrics = result.metrics
    assert set(metrics) == {"em", "f1a"}
    # Baseline measured on RTX 4090 with 4 executed questions (engine returned fewer rows).
    assert len(result.samples) > 0
    assert metrics["em"] == pytest.approx(0.0, abs=0.001)
    assert metrics["f1a"] == pytest.approx(0.1667, rel=0.25)
