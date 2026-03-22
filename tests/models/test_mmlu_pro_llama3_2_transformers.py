# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import evalution


LLAMA3_2_1B_INSTRUCT = Path("/monster/data/model/Llama-3.2-1B-Instruct")
MMLU_PRO_BASELINE = {
    "exact_match,choice-label": 0.125,
}
SCORE_BASELINE_ABS_TOLERANCE = 2 / 32

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def assert_metrics_match_baseline(
    actual: dict[str, float],
    expected: dict[str, float],
) -> None:
    assert set(actual) == set(expected)
    for key, expected_value in expected.items():
        assert actual[key] == pytest.approx(expected_value, abs=SCORE_BASELINE_ABS_TOLERANCE)


@pytest.mark.skipif(
    not LLAMA3_2_1B_INSTRUCT.exists(),
    reason="local Llama 3.2 1B Instruct weights are not available",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the llama 3.2 integration test",
)
@pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="the full-model continuous batching integration test requires Python free-threading with GIL disabled",
)
def test_llama3_2_transformers_mmlu_pro_full_model_eval(capsys: pytest.CaptureFixture[str]) -> None:
    with capsys.disabled():
        result = (
            evalution.engine(
                evalution.Transformers(
                    dtype="bfloat16",
                    attn_implementation="flash_attention_2",
                    paged_attention=True,
                    device="cuda:0",
                    batch_size="auto",
                )
            )
            .model(evalution.Model(path=str(LLAMA3_2_1B_INSTRUCT)))
            .run(
                evalution.mmlu_pro(
                    category="all",
                    num_fewshot=5,
                    batch_size=4,
                    max_new_tokens=512,
                    streaming=True,
                    max_rows=32,
                )
            )
            .result()
        )

    assert result.model["path"] == str(LLAMA3_2_1B_INSTRUCT)
    assert result.engine["dtype"] == "bfloat16"
    assert result.engine["attn_implementation"] == "flash_attention_2"
    assert result.engine["batch_size"] == "auto"
    assert result.engine["paged_attention"] is True
    assert result.engine["execution"]["effective_attn_implementation"] == "paged|flash_attention_2"
    assert result.engine["execution"]["generation_backend"] == "continuous_batching"
    assert result.engine["execution"]["paged_attention"] is True
    assert len(result.tests) == 1

    test_result = result.tests[0]
    assert test_result.name == "mmlu_pro"
    assert test_result.metadata["streaming"] is True
    assert test_result.metadata["dataset_path"] == "TIGER-Lab/MMLU-Pro"
    assert test_result.metadata["dataset_name"] is None
    assert test_result.metadata["split"] == "test"
    assert test_result.metadata["fewshot_split"] == "validation"
    assert test_result.metadata["num_fewshot"] == 5
    assert test_result.metadata["category"] == "all"
    assert test_result.metadata["apply_chat_template"] is False
    assert test_result.metadata["generation_submission_mode"] == "continuous_refill"
    assert test_result.metadata["scoring_mode"] == "generated_choice_label_exact_match"
    assert len(test_result.samples) == 32
    assert set(test_result.metrics) == {
        "exact_match,choice-label",
    }
    assert_metrics_match_baseline(test_result.metrics, MMLU_PRO_BASELINE)

    invalid_predictions = 0
    exact_matches = 0
    for index, sample in enumerate(test_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target
        assert sample.prediction
        assert sample.prompt.startswith("The following are multiple choice questions")
        assert "Question:" in sample.prompt
        assert "Options:" in sample.prompt
        assert "Answer: Let's think step by step." in sample.prompt
        assert set(sample.extracted) == {"choice-label", "choice-text"}
        assert set(sample.scores) == {"exact_match,choice-label"}
        assert sample.metadata["category"]
        assert sample.metadata["question_id"] is not None
        assert sample.metadata["src"]
        assert 0 <= int(sample.metadata["fewshot_count"]) <= 5
        assert 3 <= len(sample.metadata["choice_texts"]) <= 10
        if sample.extracted["choice-label"] == "[invalid]":
            invalid_predictions += 1
        if sample.scores["exact_match,choice-label"] == 1.0:
            exact_matches += 1

    assert exact_matches > 0
    assert invalid_predictions / len(test_result.samples) < 0.25
