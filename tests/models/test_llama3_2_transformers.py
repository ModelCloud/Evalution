from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import evalution

LLAMA3_2_1B_INSTRUCT = Path("/monster/data/model/Llama-3.2-1B-Instruct")
GSM8K_BASELINE = {
    "exact_match,strict-match": 0.3828125,
    "exact_match,flexible-extract": 0.4296875,
}
GSM8K_PLATINUM_BASELINE = {
    "exact_match,strict-match": 0.3828125,
    "exact_match,flexible-extract": 0.4375,
}
ARC_CHALLENGE_BASELINE = {
    "exact_match,choice-label": 0.46875,
    "exact_match,choice-text": 0.46875,
}
HELLASWAG_BASELINE = {
    "accuracy,loglikelihood": 0.4375,
    "accuracy,loglikelihood_norm": 0.5390625,
}

pytestmark = [pytest.mark.integration, pytest.mark.slow]


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
def test_llama3_2_transformers_full_model_eval_run(capsys: pytest.CaptureFixture[str]) -> None:
    # Disable pytest capture for the actual eval so LogBar output stays visible.
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
                evalution.gsm8k(
                    variant="cot",
                    apply_chat_template=True,
                    batch_size=24,
                    max_new_tokens=96,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.gsm8k_platinum(
                    variant="cot",
                    apply_chat_template=True,
                    batch_size=24,
                    max_new_tokens=96,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.arc_challenge(
                    apply_chat_template=True,
                    batch_size=24,
                    max_new_tokens=8,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.hellaswag(
                    batch_size=24,
                    streaming=True,
                    max_rows=128,
                )
            )
        )

    assert result.model["path"] == str(LLAMA3_2_1B_INSTRUCT)
    assert result.engine["dtype"] == "bfloat16"
    assert result.engine["attn_implementation"] == "flash_attention_2"
    assert result.engine["batch_size"] == "auto"
    assert result.engine["paged_attention"] is True
    assert result.engine["execution"]["effective_attn_implementation"] == "paged|flash_attention_2"
    assert result.engine["execution"]["generation_backend"] == "continuous_batching"
    assert result.engine["execution"]["paged_attention"] is True
    assert len(result.tests) == 4

    tests_by_name = {test_result.name: test_result for test_result in result.tests}
    assert set(tests_by_name) == {
        "gsm8k_cot",
        "gsm8k_platinum_cot",
        "arc_challenge",
        "hellaswag",
    }

    gsm8k_result = tests_by_name["gsm8k_cot"]
    assert gsm8k_result.metadata["variant"] == "cot"
    assert gsm8k_result.metadata["apply_chat_template"] is True
    assert gsm8k_result.metadata["fewshot_as_multiturn"] is True
    assert gsm8k_result.metadata["streaming"] is True
    assert gsm8k_result.metadata["generation_submission_mode"] == "continuous_refill"
    assert gsm8k_result.metadata["num_fewshot"] == 8
    assert gsm8k_result.metadata["dataset_path"] == "openai/gsm8k"
    assert len(gsm8k_result.samples) == 128
    assert set(gsm8k_result.metrics) == {
        "exact_match,strict-match",
        "exact_match,flexible-extract",
    }
    assert gsm8k_result.metrics == GSM8K_BASELINE

    test_result = tests_by_name["gsm8k_platinum_cot"]
    assert test_result.metadata["variant"] == "cot"
    assert test_result.metadata["apply_chat_template"] is True
    assert test_result.metadata["fewshot_as_multiturn"] is True
    assert test_result.metadata["streaming"] is True
    assert test_result.metadata["generation_submission_mode"] == "continuous_refill"
    assert test_result.metadata["num_fewshot"] == 8
    assert len(test_result.samples) == 128
    assert set(test_result.metrics) == {
        "exact_match,strict-match",
        "exact_match,flexible-extract",
    }
    flexible_score = test_result.metrics["exact_match,flexible-extract"]
    strict_score = test_result.metrics["exact_match,strict-match"]
    assert test_result.metrics == GSM8K_PLATINUM_BASELINE
    assert isinstance(flexible_score, float)
    assert isinstance(strict_score, float)

    invalid_predictions = 0
    exact_matches = 0
    for index, sample in enumerate(test_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target
        assert sample.prediction
        assert "<|start_header_id|>user<|end_header_id|>" in sample.prompt
        assert "Q:" in sample.prompt
        assert "A:" in sample.prompt
        assert set(sample.extracted) == {"strict-match", "flexible-extract"}
        assert set(sample.scores) == {
            "exact_match,strict-match",
            "exact_match,flexible-extract",
        }
        if sample.extracted["flexible-extract"] == "[invalid]":
            invalid_predictions += 1
        if sample.scores["exact_match,flexible-extract"] == 1.0:
            exact_matches += 1

    assert exact_matches > 0
    assert invalid_predictions / len(test_result.samples) < 0.40

    arc_result = tests_by_name["arc_challenge"]
    assert arc_result.metadata["apply_chat_template"] is True
    assert arc_result.metadata["streaming"] is True
    assert arc_result.metadata["generation_submission_mode"] == "continuous_refill"
    assert arc_result.metadata["dataset_path"] == "allenai/ai2_arc"
    assert arc_result.metadata["dataset_name"] == "ARC-Challenge"
    assert len(arc_result.samples) == 128
    assert set(arc_result.metrics) == {
        "exact_match,choice-label",
        "exact_match,choice-text",
    }
    choice_label_score = arc_result.metrics["exact_match,choice-label"]
    choice_text_score = arc_result.metrics["exact_match,choice-text"]
    assert arc_result.metrics == ARC_CHALLENGE_BASELINE
    assert isinstance(choice_label_score, float)
    assert isinstance(choice_text_score, float)

    arc_invalid_predictions = 0
    for index, sample in enumerate(arc_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target
        assert sample.prediction
        assert "<|start_header_id|>user<|end_header_id|>" in sample.prompt
        assert "Question:" in sample.prompt
        assert "Choices:" in sample.prompt
        assert set(sample.extracted) == {"choice-label", "choice-text"}
        assert set(sample.scores) == {
            "exact_match,choice-label",
            "exact_match,choice-text",
        }
        if sample.extracted["choice-label"] == "[invalid]":
            arc_invalid_predictions += 1

    assert arc_invalid_predictions / len(arc_result.samples) < 0.75

    hellaswag_result = tests_by_name["hellaswag"]
    assert hellaswag_result.metadata["streaming"] is True
    assert hellaswag_result.metadata["dataset_path"] == "Rowan/hellaswag"
    assert hellaswag_result.metadata["dataset_name"] is None
    assert hellaswag_result.metadata["split"] == "validation"
    assert hellaswag_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(hellaswag_result.samples) == 128
    assert set(hellaswag_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    hellaswag_raw_score = hellaswag_result.metrics["accuracy,loglikelihood"]
    hellaswag_norm_score = hellaswag_result.metrics["accuracy,loglikelihood_norm"]
    assert hellaswag_result.metrics == HELLASWAG_BASELINE
    assert isinstance(hellaswag_raw_score, float)
    assert isinstance(hellaswag_norm_score, float)

    for index, sample in enumerate(hellaswag_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target
        assert sample.prediction
        assert ":" in sample.prompt
        assert set(sample.extracted) == {
            "gold_index",
            "predicted_index",
            "predicted_index_norm",
        }
        assert set(sample.scores) == {
            "accuracy,loglikelihood",
            "accuracy,loglikelihood_norm",
        }
        assert "choice_logprobs" in sample.metadata
        assert "choice_logprobs_norm" in sample.metadata

    serialized = result.to_dict()
    serialized_tests = {test["name"]: test for test in serialized["tests"]}
    assert set(serialized_tests) == set(tests_by_name)
    assert len(serialized_tests["gsm8k_platinum_cot"]["samples"]) == len(test_result.samples)
    assert serialized_tests["gsm8k_platinum_cot"]["samples"][0]["prediction"]
    assert len(serialized_tests["arc_challenge"]["samples"]) == len(arc_result.samples)
    assert serialized_tests["arc_challenge"]["samples"][0]["prediction"]
    assert len(serialized_tests["hellaswag"]["samples"]) == len(hellaswag_result.samples)
    assert serialized_tests["hellaswag"]["samples"][0]["prediction"]
