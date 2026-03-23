# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

import evalution

LLAMA3_2_1B_INSTRUCT = Path("/monster/data/model/Llama-3.2-1B-Instruct")
LLAMA3_2_TRANSFORMERS_DEVICE = os.environ.get("EVALUTION_TEST_DEVICE", "cuda:0")
LLAMA3_2_TRANSFORMERS_COMPARE_LEFT_DEVICE = os.environ.get(
    "EVALUTION_TEST_COMPARE_LEFT_DEVICE",
    "cuda:0",
)
LLAMA3_2_TRANSFORMERS_COMPARE_RIGHT_DEVICE = os.environ.get(
    "EVALUTION_TEST_COMPARE_RIGHT_DEVICE",
    "cuda:1",
)
SCORE_BASELINE_ABS_TOLERANCE = 2 / 128
SCORE_BASELINE_ABS_TOLERANCE_32 = 2 / 32
MMLU_STEM_SUBSETS = {
    "stem.abstract_algebra",
    "stem.anatomy",
    "stem.astronomy",
    "stem.college_biology",
    "stem.college_chemistry",
    "stem.college_computer_science",
    "stem.college_mathematics",
    "stem.college_physics",
    "stem.computer_security",
    "stem.conceptual_physics",
    "stem.electrical_engineering",
    "stem.elementary_mathematics",
    "stem.high_school_biology",
    "stem.high_school_chemistry",
    "stem.high_school_computer_science",
    "stem.high_school_mathematics",
    "stem.high_school_physics",
    "stem.high_school_statistics",
    "stem.machine_learning",
}
MMLU_PRO_STEM_SUBSETS = {
    "stem.biology",
    "stem.chemistry",
    "stem.computer_science",
    "stem.engineering",
    "stem.math",
    "stem.physics",
}

LLAMA3_2_TRANSFORMERS_TEST_MARKS = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not LLAMA3_2_1B_INSTRUCT.exists(),
        reason="local Llama 3.2 1B Instruct weights are not available",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA is required for the llama 3.2 integration test",
    ),
    pytest.mark.skipif(
        not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
        reason="the full-model continuous batching integration test requires Python free-threading with GIL disabled",
    ),
]
LLAMA3_2_TRANSFORMERS_COMPARE_TEST_MARKS = [
    *LLAMA3_2_TRANSFORMERS_TEST_MARKS,
    pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="the full-model compare integration test requires at least two CUDA devices",
    ),
]


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    suite_factory: Callable[[], Any]
    expected_name: str
    baseline: dict[str, float]
    expected_metrics: frozenset[str]
    expected_metadata: dict[str, Any]
    expected_sample_count: int
    sample_validator: Callable[[Any, int], None]
    result_validator: Callable[[Any], None] | None = None
    abs_tolerance: float = SCORE_BASELINE_ABS_TOLERANCE


def assert_metrics_match_baseline(
    actual: dict[str, float],
    expected: dict[str, float],
    *,
    abs_tolerance: float,
) -> None:
    assert set(actual) == set(expected)
    for key, expected_value in expected.items():
        assert actual[key] == pytest.approx(expected_value, abs=abs_tolerance)


def run_llama3_2_suite(
    capsys: pytest.CaptureFixture[str],
    suite: Any,
) -> tuple[Any, Any]:
    with capsys.disabled():
        result = (
            evalution.Transformers(
                dtype="bfloat16",
                attn_implementation="paged|flash_attention_2",
                device=LLAMA3_2_TRANSFORMERS_DEVICE,
                batch_size="auto",
            )
            .model(evalution.Model(path=str(LLAMA3_2_1B_INSTRUCT)))
            .run(suite)
            .result()
        )

    assert result.model["path"] == str(LLAMA3_2_1B_INSTRUCT)
    assert result.engine["dtype"] == "bfloat16"
    assert result.engine["attn_implementation"] == "paged|flash_attention_2"
    assert result.engine["batch_size"] == "auto"
    assert result.engine["execution"]["effective_attn_implementation"] == "paged|flash_attention_2"
    assert result.engine["execution"]["generation_backend"] == "continuous_batching"
    assert result.engine["execution"]["paged_attention"] is True
    assert len(result.tests) == 1
    return result, result.tests[0]


def run_llama3_2_compare_suite(
    capsys: pytest.CaptureFixture[str],
    suite: Any,
) -> tuple[Any, Any]:
    with capsys.disabled():
        result = (
            evalution.compare(
                evalution.Transformers(
                    dtype="bfloat16",
                    attn_implementation="paged|flash_attention_2",
                    device=LLAMA3_2_TRANSFORMERS_COMPARE_LEFT_DEVICE,
                    batch_size="auto",
                ).model(
                    evalution.Model(path=str(LLAMA3_2_1B_INSTRUCT)),
                    label=LLAMA3_2_TRANSFORMERS_COMPARE_LEFT_DEVICE,
                ),
                evalution.Transformers(
                    dtype="bfloat16",
                    attn_implementation="paged|flash_attention_2",
                    device=LLAMA3_2_TRANSFORMERS_COMPARE_RIGHT_DEVICE,
                    batch_size="auto",
                ).model(
                    evalution.Model(path=str(LLAMA3_2_1B_INSTRUCT)),
                    label=LLAMA3_2_TRANSFORMERS_COMPARE_RIGHT_DEVICE,
                ),
            )
            .run(suite)
            .result()
        )

    assert result.left_name == LLAMA3_2_TRANSFORMERS_COMPARE_LEFT_DEVICE
    assert result.right_name == LLAMA3_2_TRANSFORMERS_COMPARE_RIGHT_DEVICE
    assert result.left.model["path"] == str(LLAMA3_2_1B_INSTRUCT)
    assert result.right.model["path"] == str(LLAMA3_2_1B_INSTRUCT)
    assert result.left.model["label"] == LLAMA3_2_TRANSFORMERS_COMPARE_LEFT_DEVICE
    assert result.right.model["label"] == LLAMA3_2_TRANSFORMERS_COMPARE_RIGHT_DEVICE
    assert result.left.engine["dtype"] == "bfloat16"
    assert result.right.engine["dtype"] == "bfloat16"
    assert result.left.engine["attn_implementation"] == "paged|flash_attention_2"
    assert result.right.engine["attn_implementation"] == "paged|flash_attention_2"
    assert result.left.engine["batch_size"] == "auto"
    assert result.right.engine["batch_size"] == "auto"
    assert result.left.engine["device"] == LLAMA3_2_TRANSFORMERS_COMPARE_LEFT_DEVICE
    assert result.right.engine["device"] == LLAMA3_2_TRANSFORMERS_COMPARE_RIGHT_DEVICE
    assert (
        result.left.engine["execution"]["effective_attn_implementation"]
        == "paged|flash_attention_2"
    )
    assert (
        result.right.engine["execution"]["effective_attn_implementation"]
        == "paged|flash_attention_2"
    )
    assert result.left.engine["execution"]["generation_backend"] == "continuous_batching"
    assert result.right.engine["execution"]["generation_backend"] == "continuous_batching"
    assert result.left.engine["execution"]["paged_attention"] is True
    assert result.right.engine["execution"]["paged_attention"] is True
    assert len(result.left.tests) == 1
    assert len(result.right.tests) == 1
    assert len(result.tests) == 1
    return result, result.tests[0]


def assert_single_test_serialization(result: Any, test_result: Any) -> None:
    serialized = result.to_dict()
    assert len(serialized["tests"]) == 1
    serialized_test = serialized["tests"][0]
    assert serialized_test["name"] == test_result.name
    assert len(serialized_test["samples"]) == len(test_result.samples)
    if test_result.samples:
        assert serialized_test["samples"][0]["prediction"]


def _assert_multiple_choice_loglikelihood_sample(
    sample: Any,
    index: int,
    *,
    target_values: set[str] | None = None,
    prediction_values: set[str] | None = None,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    prompt_substrings: tuple[str, ...] = (),
    metadata_validator: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    assert sample.index == index
    assert sample.prompt
    if prompt_prefix is not None:
        assert sample.prompt.startswith(prompt_prefix)
    if prompt_suffix is not None:
        assert sample.prompt.endswith(prompt_suffix)
    for expected in prompt_substrings:
        assert expected in sample.prompt
    if target_values is None:
        assert sample.target
    else:
        assert sample.target in target_values
    if prediction_values is None:
        assert sample.prediction
    else:
        assert sample.prediction in prediction_values
    assert set(sample.extracted) == {
        "gold_index",
        "predicted_index",
        "predicted_index_norm",
    }
    assert set(sample.scores) == {
        "acc,ll",
        "acc,ll_avg",
    }
    assert "choice_logprobs" in sample.metadata
    assert "choice_logprobs_norm" in sample.metadata
    if metadata_validator is not None:
        metadata_validator(sample.metadata)


def _assert_multiple_choice_loglikelihood_label_perm_sample(
    sample: Any,
    index: int,
    *,
    label_permutations: float,
    target_values: set[str] | None = None,
    prediction_values: set[str] | None = None,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    prompt_substrings: tuple[str, ...] = (),
    metadata_validator: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    assert sample.index == index
    assert sample.prompt
    if prompt_prefix is not None:
        assert sample.prompt.startswith(prompt_prefix)
    if prompt_suffix is not None:
        assert sample.prompt.endswith(prompt_suffix)
    for expected in prompt_substrings:
        assert expected in sample.prompt
    if target_values is None:
        assert sample.target
    else:
        assert sample.target in target_values
    if prediction_values is None:
        assert sample.prediction
    else:
        assert sample.prediction in prediction_values
    label_metric_suffix = f"label_perm:{label_permutations}"
    assert set(sample.extracted) == {
        "gold_index",
        "predicted_index",
        "predicted_index_norm",
        f"predicted_index_{label_metric_suffix}",
    }
    assert set(sample.scores) == {
        "acc,ll",
        "acc,ll_avg",
        f"acc,{label_metric_suffix}",
    }
    assert "choice_logprobs" in sample.metadata
    assert "choice_logprobs_norm" in sample.metadata
    assert f"choice_logprobs_{label_metric_suffix}" in sample.metadata
    assert "label_permutation_count" in sample.metadata
    assert sample.metadata["label_permutation_count"] > 0
    if metadata_validator is not None:
        metadata_validator(sample.metadata)


def _assert_gsm8k_sample(sample: Any, index: int) -> None:
    assert sample.index == index
    assert sample.prompt
    assert sample.target
    assert sample.prediction
    assert "<|start_header_id|>user<|end_header_id|>" in sample.prompt
    assert "Q:" in sample.prompt
    assert "A:" in sample.prompt
    assert set(sample.extracted) == {"numeric-extract"}
    assert set(sample.scores) == {"acc,num"}


def _assert_arc_exam_sample(sample: Any, index: int) -> None:
    assert sample.index == index
    assert sample.prompt
    assert sample.target
    assert sample.prediction
    assert sample.prompt.startswith("Question: ")
    assert sample.prompt.endswith("\nAnswer:")
    assert set(sample.extracted) == {
        "gold_index",
        "selected_indices",
        "selected_labels",
    }
    assert set(sample.scores) == {"acc,exam"}
    assert "choice_logprobs" in sample.metadata
    assert "selected_count" in sample.metadata


def _assert_arc_exam_label_perm_sample(
    sample: Any,
    index: int,
    *,
    label_permutations: float,
) -> None:
    assert sample.index == index
    assert sample.prompt
    assert sample.target
    assert sample.prediction
    assert sample.prompt.startswith("Question: ")
    assert sample.prompt.endswith("\nAnswer:")
    label_metric_suffix = f"label_perm:{label_permutations}"
    assert set(sample.extracted) == {
        "gold_index",
        "selected_indices",
        "selected_labels",
        f"predicted_index_{label_metric_suffix}",
    }
    assert set(sample.scores) == {
        "acc,exam",
        f"acc,{label_metric_suffix}",
    }
    assert "choice_logprobs" in sample.metadata
    assert f"choice_logprobs_{label_metric_suffix}" in sample.metadata
    assert "selected_count" in sample.metadata
    assert "label_permutation_count" in sample.metadata
    assert sample.metadata["label_permutation_count"] > 0


def _assert_mmlu_pro_sample(
    sample: Any,
    index: int,
    *,
    allowed_subsets: set[str] | None = None,
) -> None:
    assert sample.index == index
    assert sample.prompt
    assert sample.target
    assert sample.prediction
    assert sample.prompt.startswith("The following are multiple choice questions")
    assert "Question:" in sample.prompt
    assert "Options:" in sample.prompt
    assert "Answer: Let's think step by step." in sample.prompt
    assert set(sample.extracted) == {"choice-label", "choice-text"}
    assert set(sample.scores) == {"em,choice_label"}
    subset = sample.metadata["subset"]
    assert subset
    assert sample.metadata["subset_kind"] == "leaf"
    if allowed_subsets is not None:
        assert subset in allowed_subsets
    assert sample.metadata["subset_path"] == subset.split(".")
    assert sample.metadata["subset_value"]
    assert sample.metadata["question_id"] is not None
    assert sample.metadata["src"]
    assert 0 <= int(sample.metadata["fewshot_count"]) <= 5
    assert 3 <= len(sample.metadata["choice_texts"]) <= 10


def _validate_gsm8k_like_result(test_result: Any) -> None:
    invalid_predictions = 0
    numeric_matches = 0
    for sample in test_result.samples:
        if sample.extracted["numeric-extract"] == "[invalid]":
            invalid_predictions += 1
        if sample.scores["acc,num"] == 1.0:
            numeric_matches += 1
    assert numeric_matches > 0
    assert invalid_predictions / len(test_result.samples) < 0.40


def _validate_arc_exam_result(test_result: Any) -> None:
    exact_matches = sum(
        1
        for sample in test_result.samples
        if sample.scores["acc,exam"] == 1.0
    )
    assert exact_matches > 0


def _validate_mmlu_pro_result(test_result: Any) -> None:
    invalid_predictions = 0
    exact_matches = 0
    for sample in test_result.samples:
        if sample.extracted["choice-label"] == "[invalid]":
            invalid_predictions += 1
        if sample.scores["em,choice_label"] == 1.0:
            exact_matches += 1
    assert exact_matches > 0
    assert invalid_predictions / len(test_result.samples) < 0.25


def _metadata_has_choice_labels(min_count: int | None = None, exact_count: int | None = None) -> Callable[[dict[str, Any]], None]:
    def validate(metadata: dict[str, Any]) -> None:
        labels = metadata["choice_labels"]
        if min_count is not None:
            assert len(labels) >= min_count
        if exact_count is not None:
            assert len(labels) == exact_count

    return validate


def _metadata_field_in(field: str, allowed_values: set[str]) -> Callable[[dict[str, Any]], None]:
    def validate(metadata: dict[str, Any]) -> None:
        assert metadata[field] in allowed_values

    return validate


def _metadata_field_truthy(field: str) -> Callable[[dict[str, Any]], None]:
    def validate(metadata: dict[str, Any]) -> None:
        assert metadata[field]

    return validate


def _metadata_sentence_has_blank(metadata: dict[str, Any]) -> None:
    assert " _ " in metadata["sentence"]


def _metadata_subset_in(allowed_subsets: set[str] | None = None) -> Callable[[dict[str, Any]], None]:
    def validate(metadata: dict[str, Any]) -> None:
        subset = metadata["subset"]
        assert subset
        assert metadata["subset_kind"] == "leaf"
        assert metadata["subset_path"] == subset.split(".")
        assert metadata["subset_value"]
        if allowed_subsets is not None:
            assert subset in allowed_subsets
        assert len(metadata["choice_texts"]) == 4

    return validate


SUITE_SPECS = {
    "gsm8k": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.gsm8k(
            variant="cot",
            apply_chat_template=True,
            batch_size=24,
            max_new_tokens=96,
            streaming=True,
            max_rows=128,
        ),
        expected_name="gsm8k_cot",
        baseline={
            "acc,num": 0.38671875,
        },
        expected_metrics=frozenset({"acc,num"}),
        expected_metadata={
            "variant": "cot",
            "apply_chat_template": True,
            "fewshot_as_multiturn": True,
            "streaming": True,
            "generation_submission_mode": "continuous_refill",
            "num_fewshot": 8,
            "dataset_path": "openai/gsm8k",
            "scoring_mode": "numeric_format_insensitive",
            "primary_metric": "acc,num",
        },
        expected_sample_count=128,
        sample_validator=_assert_gsm8k_sample,
        result_validator=_validate_gsm8k_like_result,
    ),
    "gsm8k_platinum": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.gsm8k_platinum(
            variant="cot",
            apply_chat_template=True,
            batch_size=24,
            max_new_tokens=96,
            streaming=True,
            max_rows=128,
        ),
        expected_name="gsm8k_platinum_cot",
        baseline={
            "acc,num": 0.390625,
        },
        expected_metrics=frozenset({"acc,num"}),
        expected_metadata={
            "variant": "cot",
            "apply_chat_template": True,
            "fewshot_as_multiturn": True,
            "streaming": True,
            "generation_submission_mode": "continuous_refill",
            "num_fewshot": 8,
            "scoring_mode": "numeric_format_insensitive",
            "primary_metric": "acc,num",
        },
        expected_sample_count=128,
        sample_validator=_assert_gsm8k_sample,
        result_validator=_validate_gsm8k_like_result,
    ),
    "boolq": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.boolq(batch_size=24, streaming=True, max_rows=128),
        expected_name="boolq",
        baseline={
            "acc,ll": 0.6796875,
            "acc,ll_avg": 0.6796875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "super_glue",
            "dataset_name": "boolq",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"yes", "no"},
            prediction_values={"yes", "no"},
            prompt_substrings=("\nQuestion: ", "\nAnswer:"),
        ),
    ),
    "cb": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.cb(batch_size=24, streaming=True, max_rows=56),
        expected_name="cb",
        baseline={
            "acc,ll": 0.5714285714285714,
            "acc,ll_avg": 0.5714285714285714,
            "f1,ll_macro": 0.39345839345839345,
            "f1,ll_avg_macro": 0.39345839345839345,
        },
        expected_metrics=frozenset(
            {
                "acc,ll",
                "acc,ll_avg",
                "f1,ll_macro",
                "f1,ll_avg_macro",
            }
        ),
        expected_metadata={
            "streaming": True,
            "dataset_path": "super_glue",
            "dataset_name": "cb",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=56,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"True", "False", "Neither"},
            prediction_values={"True", "False", "Neither"},
            prompt_substrings=("\nQuestion: ", "True, False, or Neither?\nAnswer:"),
        ),
    ),
    "cola": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.cola(batch_size=24, streaming=True, max_rows=128),
        expected_name="cola",
        baseline={
            "acc,ll": 0.6484375,
            "acc,ll_avg": 0.6484375,
            "mcc,ll": 0.0,
            "mcc,ll_avg": 0.0,
        },
        expected_metrics=frozenset(
            {
                "acc,ll",
                "acc,ll_avg",
                "mcc,ll",
                "mcc,ll_avg",
            }
        ),
        expected_metadata={
            "streaming": True,
            "dataset_path": "nyu-mll/glue",
            "dataset_name": "cola",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"yes", "no"},
            prediction_values={"yes", "no"},
            prompt_substrings=("\nQuestion: Does this sentence make sense?\nAnswer:",),
        ),
    ),
    "commonsense_qa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.commonsense_qa(
            batch_size=24,
            streaming=True,
            max_rows=128,
        ),
        expected_name="commonsense_qa",
        baseline={
            "acc,ll": 0.546875,
            "acc,ll_avg": 0.546875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "tau/commonsense_qa",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_prefix="Question: ",
            prompt_suffix="\nAnswer:",
            prompt_substrings=("\nA. ", "\nE. "),
            metadata_validator=_metadata_has_choice_labels(exact_count=5),
        ),
    ),
    "copa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.copa(batch_size=24, streaming=True, max_rows=100),
        expected_name="copa",
        baseline={
            "acc,ll": 0.74,
            "acc,ll_avg": 0.68,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "super_glue",
            "dataset_name": "copa",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=100,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_field_in("question", {"cause", "effect"}),
        ),
    ),
    "arc_easy": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.arc_easy(batch_size=24, streaming=True, max_rows=128),
        expected_name="arc_easy",
        baseline={
            "acc,exam": 0.6640625,
        },
        expected_metrics=frozenset({"acc,exam"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "allenai/ai2_arc",
            "dataset_name": "ARC-Easy",
            "split": "test",
            "scoring_mode": "multiple_choice_exam_score",
            "scoring_reference": "clark2018arc arc-solvers calculate_scores.py",
        },
        expected_sample_count=128,
        sample_validator=_assert_arc_exam_sample,
        result_validator=_validate_arc_exam_result,
    ),
    "arc_challenge": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.arc_challenge(
            batch_size=24,
            streaming=True,
            max_rows=128,
        ),
        expected_name="arc_challenge",
        baseline={
            "acc,exam": 0.40625,
        },
        expected_metrics=frozenset({"acc,exam"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "allenai/ai2_arc",
            "dataset_name": "ARC-Challenge",
            "split": "test",
            "scoring_mode": "multiple_choice_exam_score",
            "scoring_reference": "clark2018arc arc-solvers calculate_scores.py",
        },
        expected_sample_count=128,
        sample_validator=_assert_arc_exam_sample,
        result_validator=_validate_arc_exam_result,
    ),
    "arc_challenge_label_perm_0_25": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.arc_challenge(
            batch_size=24,
            streaming=True,
            max_rows=128,
            label_permutations=0.25,
        ),
        expected_name="arc_challenge",
        baseline={
            "acc,exam": 0.40625,
            "acc,label_perm:0.25": 0.515625,
        },
        expected_metrics=frozenset(
            {
                "acc,exam",
                "acc,label_perm:0.25",
            }
        ),
        expected_metadata={
            "streaming": True,
            "dataset_path": "allenai/ai2_arc",
            "dataset_name": "ARC-Challenge",
            "split": "test",
            "scoring_mode": "multiple_choice_exam_score",
            "scoring_reference": "clark2018arc arc-solvers calculate_scores.py",
            "extra_scoring_mode": "multiple_choice_label_permutation_average",
            "label_permutations": 0.25,
            "label_permutation_metric": "acc,label_perm:0.25",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_arc_exam_label_perm_sample(
            sample,
            index,
            label_permutations=0.25,
        ),
        result_validator=_validate_arc_exam_result,
    ),
    "hellaswag": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.hellaswag(batch_size=24, streaming=True, max_rows=128),
        expected_name="hellaswag",
        baseline={
            "acc,ll": 0.4375,
            "acc,ll_avg": 0.5390625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "Rowan/hellaswag",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_substrings=(":",),
        ),
    ),
    "hellaswag_label_perm_0_25": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.hellaswag(
            batch_size=24,
            streaming=True,
            max_rows=128,
            label_permutations=0.25,
        ),
        expected_name="hellaswag",
        baseline={
            "acc,ll": 0.4375,
            "acc,ll_avg": 0.5390625,
            "acc,label_perm:0.25": 0.46875,
        },
        expected_metrics=frozenset(
            {
                "acc,ll",
                "acc,ll_avg",
                "acc,label_perm:0.25",
            }
        ),
        expected_metadata={
            "streaming": True,
            "dataset_path": "Rowan/hellaswag",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
            "extra_scoring_mode": "multiple_choice_label_permutation_average",
            "label_permutations": 0.25,
            "label_permutation_metric": "acc,label_perm:0.25",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_label_perm_sample(
            sample,
            index,
            label_permutations=0.25,
            prompt_substrings=(":",),
        ),
    ),
    "mmlu_all": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mmlu(
            subsets="all",
            num_fewshot=5,
            batch_size=24,
            streaming=True,
            max_rows=128,
        ),
        expected_name="mmlu",
        baseline={
            "acc,ll": 0.3671875,
            "acc,ll_avg": 0.3671875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "cais/mmlu",
            "dataset_name": "all",
            "subsets": ["all"],
            "subset_paths": [["all"]],
            "subset_kinds": ["all"],
            "selection_mode": "single",
            "split": "validation",
            "fewshot_split": "dev",
            "num_fewshot": 5,
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="The following are multiple choice questions",
            metadata_validator=_metadata_subset_in(),
        ),
    ),
    "mmlu_stem": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mmlu(
            subsets="stem",
            num_fewshot=5,
            batch_size=24,
            streaming=True,
            max_rows=32,
        ),
        expected_name="mmlu_stem",
        baseline={
            "acc,ll": 0.34375,
            "acc,ll_avg": 0.34375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "cais/mmlu",
            "dataset_name": "all",
            "subsets": ["stem"],
            "subset_paths": [["stem"]],
            "subset_kinds": ["node"],
            "selection_mode": "single",
            "split": "validation",
            "fewshot_split": "dev",
            "num_fewshot": 5,
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="The following are multiple choice questions",
            metadata_validator=_metadata_subset_in(MMLU_STEM_SUBSETS),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "mnli": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mnli(batch_size=24, streaming=True, max_rows=128),
        expected_name="mnli",
        baseline={
            "acc,ll": 0.5078125,
            "acc,ll_avg": 0.5078125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "nyu-mll/glue",
            "dataset_name": "mnli",
            "split": "validation_matched",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"True", "Neither", "False"},
            prediction_values={"True", "Neither", "False"},
            prompt_substrings=(" True, False or Neither?\nAnswer:",),
        ),
    ),
    "mrpc": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mrpc(batch_size=24, streaming=True, max_rows=128),
        expected_name="mrpc",
        baseline={
            "acc,ll": 0.6953125,
            "acc,ll_avg": 0.6953125,
            "f1,ll_yes": 0.8186046511627907,
            "f1,ll_avg_yes": 0.8186046511627907,
        },
        expected_metrics=frozenset(
            {
                "acc,ll",
                "acc,ll_avg",
                "f1,ll_yes",
                "f1,ll_avg_yes",
            }
        ),
        expected_metadata={
            "streaming": True,
            "dataset_path": "nyu-mll/glue",
            "dataset_name": "mrpc",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"no", "yes"},
            prediction_values={"no", "yes"},
            prompt_prefix="Sentence 1: ",
            prompt_substrings=(
                "\nSentence 2: ",
                "Do both sentences mean the same thing?",
            ),
        ),
    ),
    "openbookqa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.openbookqa(batch_size=24, streaming=True, max_rows=128),
        expected_name="openbookqa",
        baseline={
            "acc,ll": 0.25,
            "acc,ll_avg": 0.328125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "allenai/openbookqa",
            "dataset_name": "main",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Question: ",
            prompt_suffix="\nAnswer:",
            metadata_validator=_metadata_has_choice_labels(exact_count=4),
        ),
    ),
    "openbookqa_label_perm_0_25": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.openbookqa(
            batch_size=24,
            streaming=True,
            max_rows=128,
            label_permutations=0.25,
        ),
        expected_name="openbookqa",
        baseline={
            "acc,ll": 0.25,
            "acc,ll_avg": 0.328125,
            "acc,label_perm:0.25": 0.59375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg", "acc,label_perm:0.25"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "allenai/openbookqa",
            "dataset_name": "main",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
            "extra_scoring_mode": "multiple_choice_label_permutation_average",
            "label_permutations": 0.25,
            "label_permutation_metric": "acc,label_perm:0.25",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_label_perm_sample(
            sample,
            index,
            label_permutations=0.25,
            prompt_prefix="Question: ",
            prompt_suffix="\nAnswer:",
            metadata_validator=_metadata_has_choice_labels(exact_count=4),
        ),
    ),
    "piqa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.piqa(batch_size=24, streaming=True, max_rows=128),
        expected_name="piqa",
        baseline={
            "acc,ll": 0.71875,
            "acc,ll_avg": 0.7890625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "baber/piqa",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Question: ",
            prompt_substrings=("\nAnswer:",),
        ),
    ),
    "qnli": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.qnli(batch_size=24, streaming=True, max_rows=128),
        expected_name="qnli",
        baseline={
            "acc,ll": 0.4609375,
            "acc,ll_avg": 0.4609375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "nyu-mll/glue",
            "dataset_name": "qnli",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"yes", "no"},
            prediction_values={"yes", "no"},
            prompt_substrings=("Question: Does this response answer the question?\nAnswer:",),
        ),
    ),
    "qqp": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.qqp(batch_size=24, streaming=True, max_rows=128),
        expected_name="qqp",
        baseline={
            "acc,ll": 0.34375,
            "acc,ll_avg": 0.34375,
            "f1,ll_yes": 0.4615384615384615,
            "f1,ll_avg_yes": 0.4615384615384615,
        },
        expected_metrics=frozenset(
            {
                "acc,ll",
                "acc,ll_avg",
                "f1,ll_yes",
                "f1,ll_avg_yes",
            }
        ),
        expected_metadata={
            "streaming": True,
            "dataset_path": "nyu-mll/glue",
            "dataset_name": "qqp",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"yes", "no"},
            prediction_values={"yes", "no"},
            prompt_substrings=(
                "Question 1: ",
                "\nQuestion 2: ",
                "\nQuestion: Do both questions ask the same thing?\nAnswer:",
            ),
        ),
    ),
    "rte": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.rte(batch_size=24, streaming=True, max_rows=128),
        expected_name="rte",
        baseline={
            "acc,ll": 0.625,
            "acc,ll_avg": 0.625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "super_glue",
            "dataset_name": "rte",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"True", "False"},
            prediction_values={"True", "False"},
            prompt_substrings=("\nQuestion: ", " True or False?\nAnswer:"),
        ),
    ),
    "sciq": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.sciq(batch_size=24, streaming=True, max_rows=128),
        expected_name="sciq",
        baseline={
            "acc,ll": 0.9296875,
            "acc,ll_avg": 0.8984375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "allenai/sciq",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_substrings=("Question: ", "\nAnswer:"),
            metadata_validator=_metadata_has_choice_labels(exact_count=4),
        ),
    ),
    "sst2": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.sst2(batch_size=24, streaming=True, max_rows=128),
        expected_name="sst2",
        baseline={
            "acc,ll": 0.5390625,
            "acc,ll_avg": 0.5390625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "nyu-mll/glue",
            "dataset_name": "sst2",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"negative", "positive"},
            prediction_values={"negative", "positive"},
            prompt_substrings=("\nQuestion: Is this sentence positive or negative?\nAnswer:",),
        ),
    ),
    "wic": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.wic(batch_size=24, streaming=True, max_rows=128),
        expected_name="wic",
        baseline={
            "acc,ll": 0.5,
            "acc,ll_avg": 0.5,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "super_glue",
            "dataset_name": "wic",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"yes", "no"},
            prediction_values={"yes", "no"},
            prompt_prefix="Sentence 1: ",
            prompt_substrings=("\nSentence 2: ", "used in the same way"),
            metadata_validator=_metadata_field_truthy("word"),
        ),
    ),
    "wnli": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.wnli(batch_size=24, streaming=True, max_rows=71),
        expected_name="wnli",
        baseline={
            "acc,ll": 0.4225352112676056,
            "acc,ll_avg": 0.4225352112676056,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "nyu-mll/glue",
            "dataset_name": "wnli",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=71,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"False", "True"},
            prediction_values={"False", "True"},
            prompt_substrings=("\nQuestion: ", " True or False?\nAnswer:"),
        ),
    ),
    "winogrande": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.winogrande(batch_size=24, streaming=True, max_rows=128),
        expected_name="winogrande",
        baseline={
            "acc,ll": 0.5625,
            "acc,ll_avg": 0.5703125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "winogrande",
            "dataset_name": "winogrande_xl",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_sentence_has_blank,
        ),
    ),
    "mmlu_pro_all": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mmlu_pro(
            subsets="all",
            num_fewshot=5,
            batch_size=4,
            max_new_tokens=512,
            streaming=True,
            max_rows=32,
        ),
        expected_name="mmlu_pro",
        baseline={"em,choice_label": 0.125},
        expected_metrics=frozenset({"em,choice_label"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "TIGER-Lab/MMLU-Pro",
            "dataset_name": None,
            "split": "test",
            "fewshot_split": "validation",
            "num_fewshot": 5,
            "subsets": ["all"],
            "subset_paths": [["all"]],
            "subset_kinds": ["all"],
            "selection_mode": "single",
            "apply_chat_template": False,
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_choice_label_exact_match",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_mmlu_pro_sample(sample, index),
        result_validator=_validate_mmlu_pro_result,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "mmlu_pro_stem": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mmlu_pro(
            subsets="stem",
            num_fewshot=5,
            batch_size=4,
            max_new_tokens=512,
            streaming=True,
            max_rows=32,
        ),
        expected_name="mmlu_pro_stem",
        baseline={"em,choice_label": 0.34375},
        expected_metrics=frozenset({"em,choice_label"}),
        expected_metadata={
            "streaming": True,
            "dataset_path": "TIGER-Lab/MMLU-Pro",
            "dataset_name": None,
            "split": "test",
            "fewshot_split": "validation",
            "num_fewshot": 5,
            "subsets": ["stem"],
            "subset_paths": [["stem"]],
            "subset_kinds": ["node"],
            "selection_mode": "single",
            "apply_chat_template": False,
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_choice_label_exact_match",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_mmlu_pro_sample(
            sample,
            index,
            allowed_subsets=MMLU_PRO_STEM_SUBSETS,
        ),
        result_validator=_validate_mmlu_pro_result,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
}


def run_suite_spec(
    capsys: pytest.CaptureFixture[str],
    suite_key: str,
) -> tuple[Any, Any]:
    spec = SUITE_SPECS[suite_key]
    result, test_result = run_llama3_2_suite(capsys, spec.suite_factory())
    assert test_result.name == spec.expected_name
    for key, expected_value in spec.expected_metadata.items():
        assert test_result.metadata[key] == expected_value
    assert len(test_result.samples) == spec.expected_sample_count
    assert set(test_result.metrics) == spec.expected_metrics
    assert_metrics_match_baseline(
        test_result.metrics,
        spec.baseline,
        abs_tolerance=spec.abs_tolerance,
    )
    for index, sample in enumerate(test_result.samples):
        spec.sample_validator(sample, index)
    if spec.result_validator is not None:
        spec.result_validator(test_result)
    assert_single_test_serialization(result, test_result)
    return result, test_result


def run_compare_suite_spec(
    capsys: pytest.CaptureFixture[str],
    suite_key: str,
) -> tuple[Any, Any]:
    spec = SUITE_SPECS[suite_key]
    result, compare_test_result = run_llama3_2_compare_suite(capsys, spec.suite_factory())
    left_test = compare_test_result.left
    right_test = compare_test_result.right

    assert compare_test_result.name == spec.expected_name
    assert left_test.name == spec.expected_name
    assert right_test.name == spec.expected_name

    for test_result in (left_test, right_test):
        for key, expected_value in spec.expected_metadata.items():
            assert test_result.metadata[key] == expected_value
        assert len(test_result.samples) == spec.expected_sample_count
        assert set(test_result.metrics) == spec.expected_metrics
        assert_metrics_match_baseline(
            test_result.metrics,
            spec.baseline,
            abs_tolerance=spec.abs_tolerance,
        )
        for index, sample in enumerate(test_result.samples):
            spec.sample_validator(sample, index)
        if spec.result_validator is not None:
            spec.result_validator(test_result)

    assert set(compare_test_result.metrics) == spec.expected_metrics
    for metric_name, metric_result in compare_test_result.metrics.items():
        assert left_test.metrics[metric_name] == pytest.approx(
            right_test.metrics[metric_name],
            abs=spec.abs_tolerance,
        )
        assert metric_result.left_value == pytest.approx(
            left_test.metrics[metric_name],
            abs=spec.abs_tolerance,
        )
        assert metric_result.right_value == pytest.approx(
            right_test.metrics[metric_name],
            abs=spec.abs_tolerance,
        )
        assert metric_result.delta == pytest.approx(
            left_test.metrics[metric_name] - right_test.metrics[metric_name],
            abs=spec.abs_tolerance,
        )
        assert abs(metric_result.delta) <= spec.abs_tolerance

    serialized = result.to_dict()
    assert serialized["left_name"] == LLAMA3_2_TRANSFORMERS_COMPARE_LEFT_DEVICE
    assert serialized["right_name"] == LLAMA3_2_TRANSFORMERS_COMPARE_RIGHT_DEVICE
    assert len(serialized["tests"]) == 1
    serialized_test = serialized["tests"][0]
    assert serialized_test["name"] == compare_test_result.name
    assert serialized_test["left"]["name"] == left_test.name
    assert serialized_test["right"]["name"] == right_test.name
    return result, compare_test_result
