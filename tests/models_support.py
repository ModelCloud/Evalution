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

# Shared local model path and device selection for the Llama 3.2 integration matrix.
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
# Baseline tolerances track the maximum accepted score drift for each benchmark sample size.
SCORE_BASELINE_ABS_TOLERANCE = 2 / 128
SCORE_BASELINE_ABS_TOLERANCE_9 = 2 / 9
SCORE_BASELINE_ABS_TOLERANCE_32 = 2 / 32
SCORE_BASELINE_ABS_TOLERANCE_35 = 2 / 35
SCORE_BASELINE_ABS_TOLERANCE_42 = 2 / 42
SCORE_BASELINE_ABS_TOLERANCE_73 = 2 / 73
SCORE_BASELINE_ABS_TOLERANCE_89 = 2 / 89
SCORE_BASELINE_ABS_TOLERANCE_104 = 2 / 104
SCORE_BASELINE_ABS_TOLERANCE_115 = 2 / 115
SCORE_BASELINE_ABS_TOLERANCE_272 = 2 / 272
_MIN_A100_CLASS_VRAM_BYTES = 90 * 1024**3
_GPU_BASELINE_BUCKET_DEFAULT = "default"
_GPU_BASELINE_BUCKET_A100 = "a100"
_GPU_BASELINE_BUCKET_RTX4090 = "rtx4090"


def _visible_llama3_2_gpu_baseline_bucket() -> str:
    """Bucket the single visible CUDA device into the hardware baseline family used by these tests."""
    if not torch.cuda.is_available():
        return _GPU_BASELINE_BUCKET_DEFAULT

    device_props = torch.cuda.get_device_properties(0)
    device_name = device_props.name.lower()
    if "rtx 4090" in device_name:
        return _GPU_BASELINE_BUCKET_RTX4090
    # CI exposes the 96 GB-class Ampere cards as PG506 SKUs; treat them as the A100 bucket.
    if device_props.major == 8 and device_props.total_memory >= _MIN_A100_CLASS_VRAM_BYTES:
        return _GPU_BASELINE_BUCKET_A100
    return _GPU_BASELINE_BUCKET_DEFAULT


_LLAMA3_2_GPU_BASELINE_BUCKET = _visible_llama3_2_gpu_baseline_bucket()


def _select_llama3_2_gpu_baseline(
    *,
    default: dict[str, float],
    rtx4090: dict[str, float] | None = None,
    a100: dict[str, float] | None = None,
) -> dict[str, float]:
    """Select the per-GPU baseline for the single visible CUDA device."""
    if _LLAMA3_2_GPU_BASELINE_BUCKET == _GPU_BASELINE_BUCKET_RTX4090 and rtx4090 is not None:
        return rtx4090
    if _LLAMA3_2_GPU_BASELINE_BUCKET == _GPU_BASELINE_BUCKET_A100 and a100 is not None:
        return a100
    return default


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
AEXAMS_TASKS = (
    "aexams_biology",
    "aexams_islamic_studies",
    "aexams_physics",
    "aexams_science",
    "aexams_social",
)
AGIEVAL_TASKS = (
    "agieval_aqua_rat",
    "agieval_logiqa_en",
    "agieval_sat_math",
    "agieval_gaokao_english",
)
AFRIMGSM_TASKS = (
    "afrimgsm_eng",
    "afrimgsm_fra",
    "afrimgsm_swa",
    "afrimgsm_yor",
)
AFRIMMLU_TASKS = (
    "afrimmlu_eng",
    "afrimmlu_fra",
    "afrimmlu_hau",
    "afrimmlu_swa",
)
DARIJAMMLU_TASKS = (
    "darijammlu_arabic_language",
    "darijammlu_biology",
    "darijammlu_computer_science",
    "darijammlu_driving_test",
)
EGYMMLU_TASKS = (
    "egymmlu_arabic_language",
    "egymmlu_biology",
    "egymmlu_computer_science",
    "egymmlu_driving_test",
)
EUS_EXAMS_TASKS = (
    "eus_exams_eu_opeosakiadmineu",
    "eus_exams_eu_opeosakiauxenfeu",
    "eus_exams_es_ejadministrativo",
    "eus_exams_es_ejauxiliar",
)
CAREQA_TASKS = (
    "careqa_en",
    "careqa_es",
)
CABBQ_TASKS = (
    "cabbq_age",
    "cabbq_disability_status",
    "cabbq_gender",
    "cabbq_nationality",
)
ESBBQ_TASKS = (
    "esbbq_age",
    "esbbq_disability_status",
    "esbbq_gender",
    "esbbq_nationality",
)
ARABICMMLU_TASKS = (
    "arabicmmlu_all",
    "arabicmmlu_islamic_studies",
    "arabicmmlu_computer_science_high_school",
    "arabicmmlu_driving_test",
)
# AIME integration coverage tracks the legacy set plus the new 2026 release.
AIME_TASKS = ("aime", "aime24", "aime25", "aime26")
# CMMLU integration coverage uses one representative subject from the new family.
CMMLU_TASKS = ("cmmlu_agronomy",)
# KMMLU integration coverage uses one representative subject from the new family.
KMMLU_TASKS = ("kmmlu_accounting",)
# MGSM integration coverage uses one representative direct-answer language.
MGSM_TASKS = ("mgsm_direct_en",)
# MMLU-CF integration coverage uses one representative contamination-free subject.
MMLU_CF_TASKS = ("mmlu_cf_biology",)
HENDRYCKS_MATH_TASKS = (
    "hendrycks_math_algebra",
)
ALGHAFA_TASKS = ("copa_ar", "piqa_ar")
ARITHMETIC_TASKS = (
    "arithmetic_1dc",
    "arithmetic_2da",
    "arithmetic_2dm",
    "arithmetic_2ds",
    "arithmetic_3da",
    "arithmetic_3ds",
    "arithmetic_4da",
    "arithmetic_4ds",
    "arithmetic_5da",
    "arithmetic_5ds",
)
BEAR_TASKS = (
    "bear",
    "bear_big",
)
BABILONG_TASKS = tuple(evalution.benchmarks.BABILONG_TASKS)
BBH_TASKS = tuple(evalution.benchmarks.BBH_TASKS)
BANGLA_TASKS = (
    "bangla_boolqa",
    "bangla_commonsenseqa",
    "bangla_mmlu",
    "bangla_openbookqa",
    "bangla_piqa",
)
PAWS_X_TASKS = (
    "paws_x_de",
    "paws_x_en",
    "paws_x_es",
    "paws_x_fr",
    "paws_x_ja",
    "paws_x_ko",
    "paws_x_zh",
)
XCOPA_TASKS = (
    "xcopa_et",
    "xcopa_ht",
    "xcopa_id",
    "xcopa_it",
    "xcopa_qu",
    "xcopa_sw",
    "xcopa_ta",
    "xcopa_th",
    "xcopa_tr",
    "xcopa_vi",
    "xcopa_zh",
)
ARC_MT_TASKS = (
    "arc_mt_da",
    "arc_mt_fi",
    "arc_mt_is",
    "arc_mt_pt",
)
AFRIXNLI_TASKS = (
    "afrixnli_amh",
    "afrixnli_eng",
    "afrixnli_fra",
    "afrixnli_swa",
)
XNLI_TASKS = (
    "xnli_ar",
    "xnli_en",
    "xnli_fr",
    "xnli_sw",
)
XQUAD_TASKS = (
    "xquad_ar",
    "xquad_en",
    "xquad_es",
    "xquad_zh",
)
TRUTHFULQA_TASKS = (
    "truthfulqa_mc1",
    "truthfulqa_mc2",
)
XWINOGRAD_TASKS = (
    "xwinograd_en",
    "xwinograd_fr",
    "xwinograd_jp",
    "xwinograd_pt",
    "xwinograd_ru",
    "xwinograd_zh",
)
XSTORYCLOZE_TASKS = (
    "xstorycloze_ar",
    "xstorycloze_en",
    "xstorycloze_es",
    "xstorycloze_eu",
    "xstorycloze_hi",
    "xstorycloze_id",
    "xstorycloze_my",
    "xstorycloze_ru",
    "xstorycloze_sw",
    "xstorycloze_te",
    "xstorycloze_zh",
)
WINOGENDER_TASKS = (
    "winogender_all",
    "winogender_female",
    "winogender_gotcha",
    "winogender_gotcha_female",
    "winogender_gotcha_male",
    "winogender_male",
    "winogender_neutral",
)
BLIMP_INTEGRATION_SUBSETS = (
    "adjunct_island",
    "anaphor_gender_agreement",
    "animate_subject_passive",
    "animate_subject_trans",
    "complex_NP_island",
    "determiner_noun_agreement_1",
    "matrix_question_npi_licensor_present",
    "npi_present_1",
)
BLIMP_TASKS = tuple(
    f"blimp_{subset.lower()}"
    for subset in BLIMP_INTEGRATION_SUBSETS
)
BELEBELE_TASKS = (
    "belebele_amh_Ethi",
    "belebele_eng_Latn",
    "belebele_fra_Latn",
    "belebele_por_Latn",
    "belebele_spa_Latn",
    "belebele_swh_Latn",
)
COPAL_ID_TASKS = (
    "copal_id_standard",
    "copal_id_colloquial",
)
CEVAL_TASKS = (
    "ceval_accountant",
    "ceval_computer_network",
    "ceval_high_school_physics",
    "ceval_law",
)
GPQA_TASKS = (
    "gpqa_main",
    "gpqa_diamond",
    "gpqa_extended",
)
CODE_X_GLUE_TASKS = (
    "code2text_go",
    "code2text_java",
    "code2text_javascript",
    "code2text_php",
    "code2text_python",
    "code2text_ruby",
)
KOBEST_TASKS = (
    "kobest_boolq",
    "kobest_copa",
    "kobest_hellaswag",
    "kobest_sentineg",
    "kobest_wic",
)
CROWS_PAIRS_TASKS = tuple(evalution.benchmarks.CROWS_PAIRS_TASKS)

# Reuse one standard mark bundle across all Llama 3.2 full-model transformer integrations.
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
# Compare-mode suites inherit the standard marks and add the two-GPU requirement.
LLAMA3_2_TRANSFORMERS_COMPARE_TEST_MARKS = [
    *LLAMA3_2_TRANSFORMERS_TEST_MARKS,
    pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="the full-model compare integration test requires at least two CUDA devices",
    ),
]


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    # Capture the expected benchmark shape and score so each model test can share one validator.
    """Define the suite spec helper used by the surrounding tests."""
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
    """Compare metric dictionaries with one explicit absolute tolerance for each score."""

    assert set(actual) == set(expected)
    for key, expected_value in expected.items():
        assert actual[key] == pytest.approx(expected_value, abs=abs_tolerance), f'expected: `{pytest.approx(expected_value, abs=abs_tolerance)}`, actual: `{actual[key]}`'


def run_llama3_2_suite(
    capsys: pytest.CaptureFixture[str],
    suite: Any,
) -> tuple[Any, Any]:
    """Run llama3 2 suite."""
    with capsys.disabled():
        result = (
            evalution.Transformers(
                dtype="bfloat16",
                attn_implementation="paged|flash_attention_2",
                device=LLAMA3_2_TRANSFORMERS_DEVICE,
                batch_size="auto",
            )
            .model(path=str(LLAMA3_2_1B_INSTRUCT))
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


def run_llama3_2_suites(
    capsys: pytest.CaptureFixture[str],
    suites: list[Any],
) -> tuple[Any, list[Any]]:
    """Run llama3 2 suites."""
    with capsys.disabled():
        evaluation = evalution.Transformers(
            dtype="bfloat16",
            attn_implementation="paged|flash_attention_2",
            device=LLAMA3_2_TRANSFORMERS_DEVICE,
            batch_size="auto",
        ).model(path=str(LLAMA3_2_1B_INSTRUCT))
        for suite in suites:
            evaluation = evaluation.run(suite)
        result = evaluation.result()

    assert result.model["path"] == str(LLAMA3_2_1B_INSTRUCT)
    assert result.engine["dtype"] == "bfloat16"
    assert result.engine["attn_implementation"] == "paged|flash_attention_2"
    assert result.engine["batch_size"] == "auto"
    assert result.engine["execution"]["effective_attn_implementation"] == "paged|flash_attention_2"
    assert result.engine["execution"]["generation_backend"] == "continuous_batching"
    assert result.engine["execution"]["paged_attention"] is True
    assert len(result.tests) == len(suites)
    return result, result.tests


def run_llama3_2_compare_suite(
    capsys: pytest.CaptureFixture[str],
    suite: Any,
) -> tuple[Any, Any]:
    """Run llama3 2 compare suite."""
    with capsys.disabled():
        result = (
            evalution.compare(
                evalution.Transformers(
                    dtype="bfloat16",
                    attn_implementation="paged|flash_attention_2",
                    device=LLAMA3_2_TRANSFORMERS_COMPARE_LEFT_DEVICE,
                    batch_size="auto",
                ).model(
                    path=str(LLAMA3_2_1B_INSTRUCT),
                    label=LLAMA3_2_TRANSFORMERS_COMPARE_LEFT_DEVICE,
                ),
                evalution.Transformers(
                    dtype="bfloat16",
                    attn_implementation="paged|flash_attention_2",
                    device=LLAMA3_2_TRANSFORMERS_COMPARE_RIGHT_DEVICE,
                    batch_size="auto",
                ).model(
                    path=str(LLAMA3_2_1B_INSTRUCT),
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
    """Assert single test serialization."""
    serialized = result.to_dict()
    assert len(serialized["tests"]) == 1
    serialized_test = serialized["tests"][0]
    assert serialized_test["name"] == test_result.name
    assert len(serialized_test["samples"]) == len(test_result.samples)
    if test_result.samples:
        assert serialized_test["samples"][0]["prediction"] is not None


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
    """Assert multiple choice loglikelihood sample for the surrounding tests. Preserve the fallback order expected by the surrounding caller."""
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
    """Assert multiple choice loglikelihood label perm sample for the surrounding tests. Preserve the fallback order expected by the surrounding caller."""
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


def _assert_inverse_scaling_sample(sample: Any, index: int, *, subset: str) -> None:
    """Assert inverse scaling sample for the surrounding tests."""
    if subset == "resisting-correction":
        assert sample.index == index
        assert sample.prompt
        assert "\nOutput:" in sample.prompt
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
        _metadata_fields_truthy(
            "subset",
            "round",
            "part",
            "choice_labels",
            "choice_texts",
        )(sample.metadata)
        return

    _assert_multiple_choice_loglikelihood_sample(
        sample,
        index,
        prompt_suffix="Answer:",
        metadata_validator=_metadata_fields_truthy(
            "subset",
            "round",
            "part",
            "choice_labels",
            "choice_texts",
        ),
    )


def _assert_blimp_sample(sample: Any, index: int, *, subset: str) -> None:
    """Assert blimp sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt == ""
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "gold_index",
        "predicted_index",
        "predicted_index_norm",
    }
    assert set(sample.scores) == {
        "acc,ll",
        "acc,ll_avg",
    }
    assert sample.metadata["subset"] == subset
    assert sample.metadata["field"] in {
        "morphology",
        "semantics",
        "syntax",
        "syntax_semantics",
        "syntax-semantics",
    }
    assert sample.metadata["linguistics_term"]
    assert sample.metadata["uid"]
    assert isinstance(sample.metadata["simple_lm_method"], bool)
    assert isinstance(sample.metadata["one_prefix_method"], bool)
    assert isinstance(sample.metadata["two_prefix_method"], bool)
    assert isinstance(sample.metadata["lexically_identical"], bool)
    assert sample.metadata["pair_id"] >= 0
    assert "choice_logprobs" in sample.metadata
    assert "choice_logprobs_norm" in sample.metadata


def _assert_assin_sample(sample: Any, index: int, *, variant: str) -> None:
    """Assert assin sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt == ""
    assert sample.target
    assert sample.prediction
    assert sample.target.startswith(sample.metadata["premise"])
    assert sample.prediction.startswith(sample.metadata["premise"])
    assert ", certo? " in sample.target
    assert ", certo? " in sample.prediction
    assert sample.metadata["hypothesis"] in sample.target
    assert sample.metadata["hypothesis"] in sample.prediction
    assert set(sample.extracted) == {
        "gold_index",
        "predicted_index",
        "predicted_index_norm",
    }
    assert set(sample.scores) == {
        "acc,ll",
        "acc,ll_avg",
    }
    assert sample.metadata["variant"] == variant
    assert sample.metadata["sentence_pair_id"]
    assert sample.metadata["premise"]
    assert sample.metadata["hypothesis"]
    assert isinstance(sample.metadata["relatedness_score"], float)
    assert len(sample.metadata["choice_logprobs"]) == 2
    assert len(sample.metadata["choice_logprobs_norm"]) == 2


def _assert_bear_sample(
    sample: Any,
    index: int,
    *,
    variant: str,
    min_choice_count: int,
) -> None:
    """Assert bear sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt == ""
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "gold_index",
        "predicted_index",
        "predicted_index_norm",
    }
    assert set(sample.scores) == {
        "acc,ll",
        "acc,ll_avg",
    }
    assert sample.metadata["variant"] == variant
    assert sample.metadata["composite_id"]
    assert sample.metadata["relation"]
    assert sample.metadata["template"]
    assert sample.metadata["subject"]
    assert sample.metadata["item"] >= 0
    assert sample.metadata["template_index"] >= 0
    assert sample.metadata["choice_count"] >= min_choice_count
    assert len(sample.metadata["answer_options"]) == sample.metadata["choice_count"]
    assert "choice_logprobs" in sample.metadata
    assert "choice_logprobs_norm" in sample.metadata


def _assert_xwinograd_sample(sample: Any, index: int, *, language: str) -> None:
    """Assert xwinograd sample for the surrounding tests."""
    assert sample.index == index
    assert "_" in sample.prompt
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "gold_index",
        "predicted_index",
        "predicted_index_norm",
    }
    assert set(sample.scores) == {
        "acc,ll",
        "acc,ll_avg",
    }
    assert sample.metadata["language"] == language
    assert sample.metadata["sentence"] == sample.prompt
    assert sample.metadata["answer_label"] in {"1", "2"}
    assert sample.metadata["blank_index"] >= 0
    assert len(sample.metadata["choice_texts"]) == 2
    assert sample.metadata["choice_labels"] == ["A", "B"]
    assert "choice_logprobs" in sample.metadata
    assert "choice_logprobs_norm" in sample.metadata


def _assert_icelandic_winogrande_sample(sample: Any, index: int) -> None:
    """Assert icelandic winogrande sample for the surrounding tests."""
    assert sample.index == index
    assert "_" in sample.prompt
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "gold_index",
        "predicted_index",
        "predicted_index_norm",
    }
    assert set(sample.scores) == {
        "acc,ll",
        "acc,ll_avg",
    }
    assert sample.metadata["qID"]
    assert sample.metadata["sentence"] == sample.prompt
    assert sample.metadata["answer_label"] in {"1", "2"}
    assert sample.metadata["blank_index"] >= 0
    assert len(sample.metadata["choice_texts"]) == 2
    assert sample.metadata["choice_labels"] == ["A", "B"]
    assert "choice_logprobs" in sample.metadata
    assert "choice_logprobs_norm" in sample.metadata


def _assert_xstorycloze_sample(sample: Any, index: int, *, language: str) -> None:
    """Assert xstorycloze sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "gold_index",
        "predicted_index",
        "predicted_index_norm",
    }
    assert set(sample.scores) == {
        "acc,ll",
        "acc,ll_avg",
    }
    assert sample.metadata["language"] == language
    assert sample.metadata["story_id"]
    assert len(sample.metadata["input_sentences"]) == 4
    assert len(sample.metadata["choice_texts"]) == 2
    assert "choice_logprobs" in sample.metadata
    assert "choice_logprobs_norm" in sample.metadata


def _assert_crows_pairs_sample(
    sample: Any,
    index: int,
    *,
    language: str,
    bias_type: str | None,
) -> None:
    """Assert crows pairs sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt == ""
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "predicted_index",
        "predicted_label",
    }
    assert set(sample.scores) == {
        "pct_stereotype",
        "ll_diff",
    }
    assert sample.metadata["language"] == language
    assert sample.metadata["bias_category"] == (bias_type or "all")
    assert sample.metadata["bias_type"]
    assert sample.metadata["stereo_antistereo"] in {"stereo", "antistereo"}
    assert sample.metadata["choice_labels"] == ["sent_more", "sent_less"]
    assert len(sample.metadata["choice_texts"]) == 2
    assert len(sample.metadata["choice_logprobs"]) == 2


def _assert_winogender_sample(
    sample: Any,
    index: int,
    *,
    variant: str,
    gender: str | None,
) -> None:
    """Assert winogender sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.endswith("' refers to the")
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "gold_index",
        "predicted_index",
        "predicted_index_norm",
    }
    assert set(sample.scores) == {
        "acc,ll",
        "acc,ll_avg",
    }
    assert sample.metadata["variant"] == variant
    assert sample.metadata["sentid"]
    assert sample.metadata["sentence"]
    assert sample.metadata["pronoun"]
    assert len(sample.metadata["choice_texts"]) == 2
    assert sample.metadata["gender"] in {"male", "female", "neutral"}
    if gender is not None:
        assert sample.metadata["gender"] == gender
    assert isinstance(sample.metadata["gotcha"], bool)
    assert "choice_logprobs" in sample.metadata
    assert "choice_logprobs_norm" in sample.metadata


def _assert_gsm8k_sample(sample: Any, index: int) -> None:
    """Assert GSM8K sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt
    assert sample.target
    assert set(sample.extracted) == {"numeric-extract"}
    assert sample.prediction is not None
    if sample.extracted["numeric-extract"] != "[invalid]":
        assert sample.prediction
    assert "<|start_header_id|>user<|end_header_id|>" in sample.prompt
    assert "Q:" in sample.prompt
    assert "A:" in sample.prompt
    assert set(sample.scores) == {"acc,num"}


# Validate the translated direct-answer GSM8K prompts without assuming chat-template wrapping.
def _assert_gsm8k_translated_sample(sample: Any, index: int) -> None:
    """Assert GSM8K translated sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.count("Question: ") >= 1
    assert sample.prompt.endswith("\nAnswer:")
    assert "<|start_header_id|>" not in sample.prompt
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {"numeric-extract"}
    assert set(sample.scores) == {"acc,num"}


def _assert_afrimgsm_sample(sample: Any, index: int, *, language: str) -> None:
    """Assert afrimgsm sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Question: ")
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {"numeric-extract"}
    assert set(sample.scores) == {"acc,num"}
    assert sample.metadata["language"] == language
    assert sample.metadata["question"]
    assert sample.metadata["answer_number"]


def _assert_asdiv_cot_llama_sample(sample: Any, index: int) -> None:
    """Assert asdiv cot llama sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt
    assert sample.target
    assert sample.prediction
    assert "<|start_header_id|>user<|end_header_id|>" in sample.prompt
    assert "Given the following problem" in sample.prompt
    assert "Problem: " in sample.prompt
    assert 'The final answer is [answer]' in sample.prompt
    assert set(sample.extracted) == {"numeric-extract"}
    assert set(sample.scores) == {"acc,num"}
    assert sample.metadata["solution_type"]
    assert sample.metadata["formula"]


def _assert_aime_sample(sample: Any, index: int) -> None:
    """Assert aime sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Question: ")
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {
        "prediction-stripped",
        "answer-extract",
        "prediction-normalized",
        "target-normalized",
    }
    assert set(sample.scores) == {"em"}
    assert sample.metadata["problem_id"]


def _assert_hendrycks_math_sample(sample: Any, index: int, *, subset: str) -> None:
    """Assert hendrycks math sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Problem: ")
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {
        "prediction-stripped",
        "answer-extract",
        "prediction-normalized",
        "target-normalized",
    }
    assert set(sample.scores) == {"em"}
    assert sample.metadata["subset"] == subset
    assert sample.metadata["level"]
    assert sample.metadata["problem_type"]


def _assert_arithmetic_sample(sample: Any, index: int, *, task_name: str) -> None:
    """Assert arithmetic sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Question: ")
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target.startswith(" ")
    assert sample.prediction
    assert set(sample.extracted) == {
        "greedy_match",
        "token_count",
    }
    assert set(sample.scores) == {"acc,ll"}
    assert sample.metadata["variant"] == task_name
    assert sample.metadata["source_file"].startswith("data/")
    assert sample.metadata["raw_context"]
    assert sample.metadata["raw_completion"]


def _assert_arc_exam_sample(sample: Any, index: int) -> None:
    """Assert ARC exam sample for the surrounding tests."""
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


def _assert_arc_mt_sample(sample: Any, index: int, *, language: str) -> None:
    """Assert ARC mt sample for the surrounding tests."""
    _assert_arc_exam_sample(sample, index)
    assert sample.metadata["language"] == language
    assert len(sample.metadata["choice_labels"]) == 4


def _assert_arc_exam_label_perm_sample(
    sample: Any,
    index: int,
    *,
    label_permutations: float,
) -> None:
    """Assert ARC exam label perm sample for the surrounding tests."""
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


def _assert_generated_exact_match_sample(
    sample: Any,
    index: int,
    *,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    prompt_substrings: tuple[str, ...] = (),
    metadata_validator: Callable[[dict[str, Any]], None] | None = None,
    allow_empty_prediction: bool = False,
) -> None:
    """Assert generated exact match sample for the surrounding tests. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    assert sample.index == index
    assert sample.prompt
    if prompt_prefix is not None:
        assert sample.prompt.startswith(prompt_prefix)
    if prompt_suffix is not None:
        assert sample.prompt.endswith(prompt_suffix)
    for expected in prompt_substrings:
        assert expected in sample.prompt
    assert sample.target
    if allow_empty_prediction:
        assert sample.prediction is not None
    else:
        assert sample.prediction
    assert set(sample.extracted) == {
        "prediction-stripped",
        "target-stripped",
    }
    assert set(sample.scores) == {"em"}
    if metadata_validator is not None:
        metadata_validator(sample.metadata)


def _assert_generated_numeric_exact_match_sample(
    sample: Any,
    index: int,
    *,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    prompt_substrings: tuple[str, ...] = (),
    metadata_validator: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Assert generated numeric exact match sample for the surrounding tests. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    assert sample.index == index
    assert sample.prompt
    if prompt_prefix is not None:
        assert sample.prompt.startswith(prompt_prefix)
    if prompt_suffix is not None:
        assert sample.prompt.endswith(prompt_suffix)
    for expected in prompt_substrings:
        assert expected in sample.prompt
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {"numeric-extract"}
    assert set(sample.scores) == {"acc,num"}
    if metadata_validator is not None:
        metadata_validator(sample.metadata)


def _assert_generated_regex_extract_exact_match_sample(
    sample: Any,
    index: int,
    *,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    prompt_substrings: tuple[str, ...] = (),
    metadata_validator: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Assert generated regex extract exact match sample for the surrounding tests. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    assert sample.index == index
    assert sample.prompt
    if prompt_prefix is not None:
        assert sample.prompt.startswith(prompt_prefix)
    if prompt_suffix is not None:
        assert sample.prompt.endswith(prompt_suffix)
    for expected in prompt_substrings:
        assert expected in sample.prompt
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "strict-match",
        "flexible-extract",
        "target-stripped",
    }
    assert set(sample.scores) == {"em,strict", "em,flex"}
    if metadata_validator is not None:
        metadata_validator(sample.metadata)


def _assert_generated_contains_sample(sample: Any, index: int) -> None:
    """Assert generated contains sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {
        "contains-target",
        "target",
        "target-matched",
    }
    assert set(sample.scores) == {"contains"}
    assert sample.extracted["target"] == sample.target


def _assert_ruler_sample(sample: Any, index: int, *, variant: str, max_length: int) -> None:
    """Assert ruler sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {
        "prediction-normalized",
        "outputs",
        "matched_outputs",
    }
    assert set(sample.scores) == {"contains_fraction"}
    assert sample.metadata["variant"] == variant
    assert sample.metadata["gen_prefix"]
    assert sample.metadata["max_length"] == max_length
    assert sample.metadata["length"] > 0


def _metadata_has_moral_stories_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has moral stories fields."""
    assert metadata["guid"]
    assert metadata["norm"]
    assert metadata["situation"]
    assert metadata["intention"]
    assert metadata["moral_action"]
    assert metadata["immoral_action"]
    assert metadata["moral_consequence"]
    assert metadata["immoral_consequence"]


def _assert_mbpp_sample(sample: Any, index: int) -> None:
    """Assert MBPP sample for the surrounding tests."""
    assert sample.index == index
    assert "[BEGIN]" in sample.prompt
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {"passed", "code"}
    assert set(sample.scores) == {"pass@1"}
    assert sample.metadata["task_id"]
    assert sample.metadata["source_file"]
    assert "test_import_count" in sample.metadata


def _assert_ifeval_sample(sample: Any, index: int) -> None:
    """Assert IFEval sample for the surrounding tests."""
    assert sample.index == index
    assert sample.target
    assert sample.prompt
    assert sample.prediction
    assert sample.target == sample.prompt
    assert set(sample.extracted) == {
        "instruction_id_list",
        "prompt_level_strict",
        "prompt_level_loose",
        "inst_level_strict",
        "inst_level_loose",
    }
    assert set(sample.scores) == {
        "prompt_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_strict_acc",
        "inst_level_loose_acc",
    }
    assert isinstance(sample.scores["prompt_level_strict_acc"], float)
    assert isinstance(sample.scores["prompt_level_loose_acc"], float)
    assert isinstance(sample.scores["inst_level_strict_acc"], float)
    assert isinstance(sample.scores["inst_level_loose_acc"], float)
    assert sample.metadata["key"]
    assert int(sample.metadata["instruction_count"]) >= 1


def _assert_humaneval_sample(sample: Any, index: int) -> None:
    """Assert humaneval sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Complete the following Python function.")
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {"passed", "code"}
    assert set(sample.scores) == {"pass@1"}
    assert sample.metadata["task_id"]
    assert sample.metadata["entry_point"]


def _assert_babilong_sample(
    sample: Any,
    index: int,
    *,
    qa_split: str,
) -> None:
    """Assert babilong sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Context:\n")
    assert sample.prompt.endswith("\n\nAnswer:")
    assert "\n\nQuestion:\n" in sample.prompt
    assert sample.target
    assert set(sample.extracted) == {"prediction-stripped", "target-stripped"}
    assert set(sample.scores) == {"em"}
    _metadata_babilong_split(qa_split)(sample.metadata)


def _assert_generated_summary_sample(
    sample: Any,
    index: int,
    *,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    prompt_substrings: tuple[str, ...] = (),
    metadata_validator: Callable[[dict[str, Any]], None] | None = None,
    allow_empty_prediction: bool = False,
) -> None:
    """Assert generated summary sample for the surrounding tests. Preserve the fallback order expected by the surrounding caller."""
    assert sample.index == index
    assert sample.prompt
    if prompt_prefix is not None:
        assert sample.prompt.startswith(prompt_prefix)
    if prompt_suffix is not None:
        assert sample.prompt.endswith(prompt_suffix)
    for expected in prompt_substrings:
        assert expected in sample.prompt
    assert sample.target
    if allow_empty_prediction:
        assert sample.prediction is not None
    else:
        assert sample.prediction
    assert set(sample.extracted) == {
        "prediction-stripped",
        "reference-stripped",
    }
    assert set(sample.scores) == {"rouge1", "rouge2", "rougeLsum"}
    if metadata_validator is not None:
        metadata_validator(sample.metadata)


def _assert_longbench_sample(
    sample: Any,
    index: int,
    *,
    task_root: str,
    metric_name: str,
    language: str,
) -> None:
    """Assert longbench sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {
        "prediction-scored",
        "best_reference_index",
        "best_reference",
    }
    assert set(sample.scores) == {"score", metric_name}
    assert sample.metadata["dataset"] == task_root
    assert sample.metadata["task"] == task_root
    assert sample.metadata["language"] == language
    assert sample.metadata["length"]
    assert sample.metadata["answers"]
    assert sample.metadata["all_classes"]
    assert sample.extracted["best_reference"] in sample.metadata["answers"]
    assert sample.extracted["prediction-scored"]


def _assert_translation_corpus_sample(
    sample: Any,
    index: int,
    *,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    prompt_substrings: tuple[str, ...] = (),
    metadata_validator: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Assert translation corpus sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt
    if prompt_prefix is not None:
        assert sample.prompt.startswith(prompt_prefix)
    if prompt_suffix is not None:
        assert sample.prompt.endswith(prompt_suffix)
    for expected in prompt_substrings:
        assert expected in sample.prompt
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "prediction-stripped",
        "reference-stripped",
    }
    assert sample.scores == {}
    if metadata_validator is not None:
        metadata_validator(sample.metadata)


def _assert_generated_bleu_rouge_sample(
    sample: Any,
    index: int,
    *,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    prompt_substrings: tuple[str, ...] = (),
    metadata_validator: Callable[[dict[str, Any]], None] | None = None,
    allow_empty_prediction: bool = False,
) -> None:
    """Assert generated bleu ROUGE sample for the surrounding tests. Preserve the fallback order expected by the surrounding caller."""
    assert sample.index == index
    assert sample.prompt
    if prompt_prefix is not None:
        assert sample.prompt.startswith(prompt_prefix)
    if prompt_suffix is not None:
        assert sample.prompt.endswith(prompt_suffix)
    for expected in prompt_substrings:
        assert expected in sample.prompt
    assert sample.target
    if allow_empty_prediction:
        assert sample.prediction is not None
    else:
        assert sample.prediction
    assert set(sample.extracted) == {
        "prediction-stripped",
        "reference-stripped",
    }
    assert set(sample.scores) == {"bleu", "rouge1", "rouge2", "rougeL"}
    if metadata_validator is not None:
        metadata_validator(sample.metadata)


def _assert_noticia_sample(sample: Any, index: int) -> None:
    """Assert noticia sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith(
        "Ahora eres una Inteligencia Artificial experta en desmontar titulares "
        "sensacionalistas o clickbait."
    )
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {
        "prediction-clean",
        "reference-clean",
    }
    assert set(sample.scores) == {"rouge1", "average_len"}
    assert sample.metadata["web_url"].startswith("http")
    assert sample.metadata["web_headline"]
    assert sample.metadata["web_text_chars"] > 0


def _assert_cocoteros_sample(sample: Any, index: int) -> None:
    """Assert cocoteros sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Genera una frase corta con estas palabras: ")
    assert sample.prompt.endswith("\n\nRespuesta:")
    assert " El contexto es: " in sample.prompt
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {
        "prediction-stripped",
        "reference-stripped",
    }
    assert sample.scores == {}
    assert sample.metadata["keywords"]
    assert sample.metadata["context"]
    assert sample.metadata["keywords"] in sample.prompt
    assert sample.metadata["context"] in sample.prompt


def _assert_copa_es_sample(sample: Any, index: int) -> None:
    """Assert COPA es sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt
    assert sample.prompt.endswith(("porque", "y por lo tanto"))
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "gold_index",
        "predicted_index",
        "predicted_index_norm",
    }
    assert set(sample.scores) == {
        "acc,ll",
        "acc,ll_avg",
    }
    assert sample.metadata["id"]
    assert sample.metadata["question"] in {"cause", "effect"}
    assert sample.metadata["premise"]
    assert "choice_logprobs" in sample.metadata
    assert "choice_logprobs_norm" in sample.metadata


def _assert_simple_cooccurrence_bias_sample(sample: Any, index: int) -> None:
    """Assert simple cooccurrence bias sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt
    assert sample.target == "female/woman/male/man"
    assert sample.prediction in {"female", "woman", "male", "man"}
    assert set(sample.extracted) == {
        "predicted_index",
        "predicted_label",
        "preferred_group",
    }
    assert set(sample.scores) == {"likelihood_diff", "pct_male_preferred"}
    assert sample.extracted["preferred_group"] in {"female", "male"}
    assert sample.metadata["occupation"]
    assert sample.metadata["choice_texts"] == ["female", "woman", "male", "man"]
    assert len(sample.metadata["choice_logprobs"]) == 4
    assert isinstance(sample.metadata["female_logsumexp"], float)
    assert isinstance(sample.metadata["male_logsumexp"], float)


def _assert_cnn_dailymail_metadata(metadata: dict[str, Any]) -> None:
    """Assert CNN dailymail metadata for the surrounding tests."""
    assert metadata["id"]
    assert metadata["article_chars"] > 0
    assert metadata["reference_lines"] >= 1


def _assert_code2text_sample(sample: Any, index: int, *, language: str) -> None:
    """Assert code2text sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt == sample.prompt.strip()
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "prediction-stripped",
        "reference-stripped",
    }
    assert sample.scores == {}
    assert sample.metadata["language"] == language
    assert sample.metadata["repo"]
    assert sample.metadata["path"]
    assert sample.metadata["func_name"]
    assert sample.metadata["sha"]
    assert sample.metadata["url"]
    assert sample.metadata["code_token_count"] > 0
    assert sample.metadata["docstring_token_count"] > 0


def _assert_single_continuation_loglikelihood_sample(
    sample: Any,
    index: int,
    *,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    prompt_substrings: tuple[str, ...] = (),
    metadata_validator: Callable[[dict[str, Any]], None] | None = None,
    expected_scores: frozenset[str] = frozenset({"acc,ll", "ppl,ll"}),
    require_leading_space_target: bool = True,
) -> None:
    """Assert single continuation loglikelihood sample for the surrounding tests. Preserve the fallback order expected by the surrounding caller."""
    assert sample.index == index
    assert sample.prompt
    if prompt_prefix is not None:
        assert sample.prompt.startswith(prompt_prefix)
    if prompt_suffix is not None:
        assert sample.prompt.endswith(prompt_suffix)
    for expected in prompt_substrings:
        assert expected in sample.prompt
    if require_leading_space_target:
        assert sample.target.startswith(" ")
    else:
        assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "greedy_match",
        "token_count",
    }
    assert set(sample.scores) == expected_scores
    assert "logprob" in sample.metadata
    assert "token_count" in sample.metadata
    assert "is_greedy" in sample.metadata
    if metadata_validator is not None:
        metadata_validator(sample.metadata)


def _assert_webqs_sample(sample: Any, index: int) -> None:
    """Assert webqs sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Question: ")
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "greedy_alias_index",
        "highest_logprob_alias_index",
    }
    assert set(sample.scores) == {"em"}
    assert sample.metadata["question"]
    assert sample.metadata["url"]
    assert sample.metadata["accepted_answers"]
    assert sample.metadata["choice_texts"] == sample.metadata["accepted_answers"]
    assert len(sample.metadata["choice_logprobs"]) == len(sample.metadata["accepted_answers"])
    assert len(sample.metadata["choice_greedy"]) == len(sample.metadata["accepted_answers"])


def _assert_wikitext_sample(sample: Any, index: int) -> None:
    """Assert wikitext sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt == ""
    assert sample.target == "[document]"
    assert sample.prediction == "[rolling-loglikelihood]"
    assert set(sample.extracted) == {"token_count", "word_count", "byte_count"}
    assert set(sample.scores) == {"word_perplexity", "byte_perplexity", "bits_per_byte"}
    assert sample.metadata["page_preview"]
    assert sample.metadata["detokenized_preview"]
    assert sample.metadata["page_char_count"] > 0
    assert "logprob" in sample.metadata
    assert sample.metadata["token_count"] >= 1


def _assert_c4_sample(sample: Any, index: int) -> None:
    """Assert c4 sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt == ""
    assert sample.target == "[document]"
    assert sample.prediction == "[rolling-loglikelihood]"
    assert set(sample.extracted) == {"token_count", "word_count", "byte_count"}
    assert set(sample.scores) == {"word_perplexity", "byte_perplexity", "bits_per_byte"}
    assert sample.metadata["text_preview"]
    assert sample.metadata["text_char_count"] > 0
    assert sample.metadata["url"]
    assert sample.metadata["timestamp"]
    assert "logprob" in sample.metadata
    assert sample.metadata["token_count"] >= 1


def _assert_pile_10k_sample(sample: Any, index: int) -> None:
    """Assert pile 10k sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt == ""
    assert sample.target == "[document]"
    assert sample.prediction == "[rolling-loglikelihood]"
    assert set(sample.extracted) == {"token_count", "word_count", "byte_count"}
    assert set(sample.scores) == {"word_perplexity", "byte_perplexity", "bits_per_byte"}
    assert sample.metadata["text_preview"]
    assert sample.metadata["text_char_count"] > 0
    assert sample.metadata["pile_set_name"]
    assert "logprob" in sample.metadata
    assert sample.metadata["token_count"] >= 1


def _assert_squadv2_sample(sample: Any, index: int) -> None:
    """Assert squadv2 sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Title: ")
    assert "\nContext: " in sample.prompt
    assert "\nQuestion: " in sample.prompt
    assert "unanswerable" in sample.prompt
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {"prediction-normalized", "best_answer_index", "best_answer"}
    assert set(sample.scores) == {"em", "f1"}
    assert sample.metadata["id"]
    assert sample.metadata["title"] is not None
    assert sample.metadata["question"]
    assert sample.metadata["answer_texts"]
    assert "has_answer" in sample.metadata


def _assert_triviaqa_sample(sample: Any, index: int) -> None:
    """Assert triviaqa sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Question: ")
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {"prediction-normalized", "best_answer_index", "best_answer"}
    assert set(sample.scores) == {"em", "f1"}
    assert sample.metadata["question_id"]
    assert sample.metadata["question_source"]
    assert sample.metadata["question"]
    assert sample.metadata["answer_aliases"]
    assert sample.metadata["answer_type"]
    assert sample.metadata["answer_value"]


def _assert_xquad_sample(sample: Any, index: int, *, language: str) -> None:
    """Assert XQuAD sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Context: ")
    assert "\n\nQuestion: " in sample.prompt
    assert sample.prompt.endswith("\n\nAnswer:")
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {"prediction-normalized", "best_answer_index", "best_answer"}
    assert set(sample.scores) == {"em", "f1"}
    assert sample.metadata["id"]
    assert sample.metadata["language"] == language
    assert sample.metadata["question"]
    assert sample.metadata["context"]
    assert sample.metadata["answer_texts"]


def _assert_mlqa_sample(sample: Any, index: int, *, context_language: str, question_language: str) -> None:
    """Assert MLQA sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Context: ")
    assert "\n\nQuestion: " in sample.prompt
    assert sample.prompt.endswith("\n\nAnswer:")
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {"prediction-normalized", "best_answer_index", "best_answer"}
    assert set(sample.scores) == {"em", "f1"}
    assert sample.metadata["id"]
    assert sample.metadata["context_language"] == context_language
    assert sample.metadata["question_language"] == question_language
    assert sample.metadata["question"]
    assert sample.metadata["context"]
    assert sample.metadata["answer_texts"]


def _assert_truthfulqa_sample(sample: Any, index: int, *, variant: str) -> None:
    """Assert truthfulqa sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Q: What is human life expectancy in the United States?")
    assert "\n\nQ: " in sample.prompt
    assert sample.prompt.endswith("\nA:")
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {"predicted_index", "correct_indices"}
    assert set(sample.scores) == {"acc"}
    assert sample.metadata["question"]
    assert sample.metadata["variant"] == variant
    assert sample.metadata["choice_texts"]
    assert sample.metadata["choice_labels"]
    assert sample.metadata["choice_logprobs"]
    assert sample.metadata["choice_probs"]


def _assert_nq_open_sample(sample: Any, index: int) -> None:
    """Assert nq open sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Question: ")
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {"prediction-normalized", "best_answer_index", "best_answer"}
    assert set(sample.scores) == {"em", "f1"}
    assert sample.metadata["question"]
    assert sample.metadata["answer_aliases"]


def _assert_qasper_freeform_sample(sample: Any, index: int) -> None:
    """Assert QASPER freeform sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("TITLE: ")
    assert "\nABSTRACT: " in sample.prompt
    assert "\n\nQ: " in sample.prompt
    assert sample.prompt.endswith("\n\nA:")
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {"prediction-normalized", "answer-normalized"}
    assert set(sample.scores) == {"f1"}
    _metadata_has_qasper_fields(answer_type="free form answer")(sample.metadata)


def _assert_scrolls_qa_sample(sample: Any, index: int, *, variant: str) -> None:
    """Assert scrolls QA sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt
    assert "\n\nQuestion: " in sample.prompt
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {"prediction-normalized", "best_answer_index", "best_answer"}
    assert set(sample.scores) == {"em", "f1"}
    assert sample.metadata["id"]
    assert sample.metadata["pid"]
    assert sample.metadata["variant"] == variant
    assert sample.metadata["question"]
    assert sample.metadata["text"]
    assert sample.metadata["outputs"]


def _assert_coqa_sample(sample: Any, index: int) -> None:
    """Assert coqa sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Story: ")
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {"prediction-normalized", "best_answer_index", "best_answer"}
    assert set(sample.scores) == {"em", "f1"}
    assert sample.metadata["source"]
    assert sample.metadata["conversation_index"] >= 0
    assert sample.metadata["turn_index"] >= 1
    assert sample.metadata["turn_count"] >= sample.metadata["turn_index"]
    assert sample.metadata["history_turns"] == sample.metadata["turn_index"] - 1
    assert sample.metadata["question"]
    assert isinstance(sample.metadata["answer_start"], int)
    assert isinstance(sample.metadata["answer_end"], int)
    assert sample.prompt.count("\nQuestion: ") == sample.metadata["history_turns"] + 1
    assert sample.prompt.count("\nAnswer:") == sample.metadata["history_turns"] + 1


def _assert_drop_sample(sample: Any, index: int) -> None:
    """Assert drop sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt.startswith("Passage: ")
    assert "\nQuestion: " in sample.prompt
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target
    assert sample.prediction is not None
    assert set(sample.extracted) == {"prediction-normalized", "best_answer_index", "best_answer"}
    assert set(sample.scores) == {"em", "f1"}
    assert sample.metadata["section_id"]
    assert sample.metadata["query_id"]
    assert sample.metadata["question"]
    assert sample.metadata["answer_spans"]
    assert sample.metadata["answer_types"]


def _assert_mmlu_pro_sample(
    sample: Any,
    index: int,
    *,
    allowed_subsets: set[str] | None = None,
    max_choice_count: int = 10,
) -> None:
    """Assert MMLU pro sample for the surrounding tests."""
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
    assert 3 <= len(sample.metadata["choice_texts"]) <= max_choice_count


def _assert_gpqa_sample(sample: Any, index: int, *, subset: str) -> None:
    """Assert GPQA sample for the surrounding tests."""
    assert sample.index == index
    assert sample.prompt
    assert sample.target in {"A", "B", "C", "D"}
    assert sample.prediction
    assert sample.prompt.startswith("What is the correct answer to this question: ")
    assert "\nChoices:\n(A) " in sample.prompt
    assert sample.prompt.endswith('Format your response as follows: "The correct answer is (insert answer here)"')
    assert set(sample.extracted) == {"choice-label", "choice-text"}
    assert set(sample.scores) == {"em,choice_label"}
    assert sample.metadata["subset"] == subset
    assert sample.metadata["record_id"]
    assert sample.metadata["question"]
    assert sample.metadata["high_level_domain"]
    assert sample.metadata["subdomain"]
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert len(sample.metadata["choice_texts"]) == 4
    assert sample.metadata["gold_choice"] in sample.metadata["choice_texts"]
    assert sample.metadata["shuffle_seed"] == 0


def _validate_gsm8k_like_result(test_result: Any) -> None:
    """Validate GSM8K like result."""
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
    """Validate ARC exam result."""
    exact_matches = sum(
        1
        for sample in test_result.samples
        if sample.scores["acc,exam"] == 1.0
    )
    assert exact_matches > 0


def _validate_mmlu_pro_result(test_result: Any) -> None:
    """Validate MMLU pro result."""
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
    """Support the surrounding tests with metadata has choice labels."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        labels = metadata["choice_labels"]
        if min_count is not None:
            assert len(labels) >= min_count
        if exact_count is not None:
            assert len(labels) == exact_count

    return validate


def _metadata_field_in(field: str, allowed_values: set[str]) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata field in."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata[field] in allowed_values

    return validate


def _metadata_field_truthy(field: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata field truthy."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata[field]

    return validate


def _metadata_fields_truthy(*fields: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata fields truthy."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        for field in fields:
            assert metadata[field]

    return validate


def _metadata_fields_present(*fields: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata fields present."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        for field in fields:
            assert field in metadata

    return validate


def _metadata_question_and_variant(variant: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata question and variant."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["question"] in {"cause", "effect"}
        assert metadata["variant"] == variant
        assert len(metadata["raw_choices"]) == 2

    return validate


def _metadata_ceval_subset(subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata ceval subset."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subset"] == subset
        assert metadata["answer_label"] in {"A", "B", "C", "D"}
        assert len(metadata["raw_choices"]) == 4

    return validate


def _metadata_agieval_subset(subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata agieval subset."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subset"] == subset
        assert metadata["question"]
        assert metadata["answer_label"] in {"A", "B", "C", "D", "E"}
        assert metadata["choice_labels"]
        assert metadata["raw_choices"]
        assert len(metadata["choice_labels"]) == len(metadata["raw_choices"])

    return validate


def _metadata_afrimgsm_language(language: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata afrimgsm language."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["language"] == language
        assert metadata["question"]
        assert metadata["answer_number"]

    return validate


def _metadata_cmmlu_subset(subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata cmmlu subset."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subset"] == subset
        assert metadata["question"]

    return validate


def _metadata_kmmlu_subset(subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata kmmlu subset."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subset"] == subset
        assert metadata["category"]
        assert metadata["question"]
        assert metadata["human_accuracy"] >= 0.0

    return validate


def _metadata_mmlu_cf_subject(subject: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata MMLU cf subject."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subject"] == subject
        assert metadata["question"]

    return validate


def _metadata_afrimmlu_language(language: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata afrimmlu language."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["language"] == language
        assert metadata["subject"]
        assert metadata["question"]
        assert metadata["answer_label"] in {"A", "B", "C", "D"}
        assert metadata["raw_choices"]
        assert len(metadata["raw_choices"]) == 4

    return validate


def _metadata_arabicmmlu_subset(subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata arabicmmlu subset."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subset"] == subset
        assert metadata["group"]
        assert metadata["subject"]
        assert metadata["question"]
        assert metadata["answer_label"] in {"A", "B", "C", "D", "E"}
        assert metadata["choice_labels"]
        assert metadata["raw_choices"]
        assert len(metadata["choice_labels"]) == len(metadata["raw_choices"])

    return validate


def _metadata_afrixnli_language(language: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata afrixnli language."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["language"] == language
        assert metadata["premise"]
        assert metadata["hypothesis"]
        assert metadata["choice_texts"] == ["entailment", "neutral", "contradiction"]

    return validate


def _metadata_xnli_language(language: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata XNLI language."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["language"] == language
        assert metadata["premise"]
        assert metadata["hypothesis"]
        assert metadata["choice_texts"] == ["entailment", "neutral", "contradiction"]

    return validate


def _metadata_belebele_language(language: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata belebele language."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["language"] == language
        assert metadata["dialect"] == language
        assert metadata["question_number"] > 0
        assert metadata["link"]
        assert metadata["passage"]
        assert metadata["question"]
        assert len(metadata["raw_choices"]) == 4
        assert metadata["correct_answer_num"] in {"1", "2", "3", "4"}

    return validate


def _metadata_arc_mt_language(language: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata ARC mt language."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["language"] == language

    return validate


def _metadata_bbh_subset(subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata BBH subset."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subset"] == subset
        assert metadata["input"]
        assert metadata["target_text"]

    return validate


def _metadata_babilong_split(qa_split: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata babilong split."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["context_length"] == "0k"
        assert metadata["qa_split"] == qa_split
        assert metadata["question"]

    return validate


def _metadata_bangla_subset(subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata bangla subset. Preserve the fallback order expected by the surrounding caller."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate. Preserve the fallback order expected by the surrounding caller."""
        assert metadata["subset"] == subset
        if subset == "boolqa":
            assert metadata["passage"]
            assert metadata["question"]
            return
        assert metadata["question"]
        assert metadata["choice_labels"]
        assert metadata["raw_choices"]
        if subset == "commonsenseqa":
            assert len(metadata["raw_choices"]) == 5
            return
        if subset == "mmlu":
            assert len(metadata["raw_choices"]) == 4
            assert "subject" in metadata
            return
        if subset == "openbookqa":
            assert len(metadata["raw_choices"]) == 4
            return
        if subset == "piqa":
            assert len(metadata["raw_choices"]) == 2
            return
        raise AssertionError(f"unsupported bangla subset metadata: {subset!r}")

    return validate


def _metadata_kobest_subset(subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata kobest subset. Preserve the fallback order expected by the surrounding caller."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate. Preserve the fallback order expected by the surrounding caller."""
        assert metadata["subset"] == subset
        if subset == "boolq":
            assert metadata["paragraph"]
            assert metadata["question"]
            return
        if subset == "copa":
            assert metadata["question"] in {"원인", "결과"}
            assert len(metadata["raw_choices"]) == 2
            return
        if subset == "hellaswag":
            assert metadata["context"]
            assert len(metadata["raw_choices"]) == 4
            return
        if subset == "sentineg":
            assert metadata["sentence"]
            return
        if subset == "wic":
            assert metadata["word"]
            assert metadata["context_1"]
            assert metadata["context_2"]
            return
        raise AssertionError(f"unsupported kobest subset metadata: {subset!r}")

    return validate


def _metadata_sentence_has_blank(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata sentence has blank."""
    assert " _ " in metadata["sentence"]


def _metadata_has_wsc_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has WSC fields."""
    assert metadata["noun"]
    assert metadata["pronoun"]
    assert isinstance(metadata["span2_index"], int)


def _metadata_has_multirc_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has multirc fields."""
    assert metadata["paragraph"]
    assert metadata["question"]
    assert metadata["answer"]
    assert metadata["idx"]["paragraph"] >= 0
    assert metadata["idx"]["question"] >= 0
    assert metadata["idx"]["answer"] >= 0


def _metadata_has_record_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has record fields."""
    assert metadata["query"]
    assert metadata["answers"]
    assert metadata["entities"]
    assert metadata["idx"]["passage"] >= 0
    assert metadata["idx"]["query"] >= 0


def _metadata_has_eus_trivia_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has eus trivia fields."""
    assert isinstance(metadata["id"], int)
    assert metadata["category"]
    assert metadata["difficulty"]
    assert metadata["question"]
    assert metadata["raw_choices"]
    assert metadata["choice_labels"] in (
        ["A", "B"],
        ["A", "B", "C"],
        ["A", "B", "C", "D"],
    )


def _metadata_has_eus_proficiency_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has eus proficiency fields."""
    assert isinstance(metadata["id"], int)
    assert metadata["question"]
    assert metadata["raw_choices"]
    assert metadata["choice_labels"] == ["A", "B", "C", "D"]


def _metadata_has_eus_reading_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has eus reading fields."""
    assert isinstance(metadata["id"], int)
    assert metadata["context"]
    assert metadata["question"]
    assert metadata["raw_choices"]
    assert metadata["choice_labels"] in (
        ["A", "B"],
        ["A", "B", "C"],
        ["A", "B", "C", "D"],
    )


def _metadata_has_xnli_eu_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has XNLI eu fields."""
    assert metadata["language"] == "eu"
    assert metadata["premise"]
    assert metadata["hypothesis"]
    assert metadata["choice_texts"] == ["Bai", "Gainera", "Ez"]


def _metadata_has_toxigen_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has toxigen fields."""
    assert metadata["text"]
    assert metadata["target_group"]
    assert metadata["predicted_group"]
    assert metadata["factual"]
    assert metadata["framing"] is not None
    assert metadata["predicted_author"]


def _assert_generated_label_micro_f1_sample(
    sample: Any,
    index: int,
    *,
    prompt_prefix: str | None = None,
    prompt_suffix: str | None = None,
    metadata_validator: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Assert generated label micro F1 sample for the surrounding tests. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    assert sample.index == index
    assert sample.prompt
    if prompt_prefix is not None:
        assert sample.prompt.startswith(prompt_prefix)
    if prompt_suffix is not None:
        assert sample.prompt.endswith(prompt_suffix)
    assert sample.target
    assert sample.prediction
    assert set(sample.extracted) == {
        "prediction-stripped",
        "target-stripped",
    }
    assert set(sample.scores) == {"f1"}
    if metadata_validator is not None:
        metadata_validator(sample.metadata)


def _metadata_has_polemo2_fields(metadata: dict[str, Any], *, variant: str) -> None:
    """Support the surrounding tests with metadata has polemo2 fields."""
    assert metadata["variant"] == variant
    assert metadata["sentence"]
    assert metadata["target_label"].startswith("__label__meta_")


def _metadata_has_mastermind_fields(
    metadata: dict[str, Any],
    *,
    variant: str,
    code_shape: str,
    difficulty: str,
) -> None:
    """Support the surrounding tests with metadata has mastermind fields."""
    assert metadata["id"] >= 0
    assert metadata["variant"] == variant
    assert metadata["code_shape"] == code_shape
    assert metadata["difficulty"] == difficulty
    assert metadata["option_labels"] == ["A", "B", "C", "D"]
    assert len(metadata["choice_texts"]) == 4


def _metadata_has_click_fields(*, subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata has click fields."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subset"] == subset
        assert metadata["id"]
        assert metadata["question"]
        assert metadata["answer_text"] in metadata["raw_choices"]
        assert metadata["choice_labels"] in (
            ["A", "B", "C", "D"],
            ["A", "B", "C", "D", "E"],
        )
        assert len(metadata["choice_labels"]) == len(metadata["raw_choices"])

    return validate


def _metadata_has_phrases_es_fields(
    *,
    direction: str,
    source_language: str,
    target_language: str,
) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata has phrases es fields."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["direction"] == direction
        assert metadata["source_language"] == source_language
        assert metadata["target_language"] == target_language
        assert isinstance(metadata["id"], int)

    return validate


def _metadata_has_flores_fields(
    *,
    direction: str,
    source_language: str,
    target_language: str,
) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata has flores fields."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["direction"] == direction
        assert metadata["source_language"] == source_language
        assert metadata["target_language"] == target_language
        assert isinstance(metadata["id"], int)
        assert metadata["URL"].startswith("http")
        assert metadata["domain"]
        assert isinstance(metadata["has_image"], bool)
        assert isinstance(metadata["has_hyperlink"], bool)

    return validate


def _metadata_has_groundcocoa_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has groundcocoa fields."""
    assert metadata["id"]
    assert metadata["query_pos"]
    assert isinstance(metadata["is_typical"], bool)


def _metadata_has_escola_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has escola fields."""
    assert metadata["id"]
    assert metadata["source"]
    assert metadata["category"] >= 0
    assert metadata["split_name"]


def _metadata_has_meqsum_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has meqsum fields."""
    assert metadata["file"]
    assert metadata["question_chars"] > 0
    assert metadata["summary_words"] > 0


def _metadata_has_xlsum_es_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has XLSum es fields."""
    assert metadata["id"]
    assert metadata["url"].startswith("http")
    assert metadata["title"]
    assert metadata["article_chars"] > 0
    assert metadata["reference_lines"] >= 1


def _metadata_has_mediqa_qa2019_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has mediqa qa2019 fields."""
    assert metadata["qid"]
    assert metadata["answer_count"] >= 1
    assert metadata["first_answer_aid"]
    assert metadata["first_answer_reference_rank"] >= 0
    assert metadata["first_answer_reference_score"] >= 0


def _metadata_has_qasper_fields(*, answer_type: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata has QASPER fields."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["title"]
        assert metadata["abstract"]
        assert metadata["question"]
        assert metadata["answer_type"] == answer_type

    return validate


def _metadata_has_mmlu_redux_fields(*, subset: str, subject: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata has MMLU redux fields."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subject"] == subject
        assert metadata["subset"] == subset
        assert metadata["subset_path"] == subset.split(".")
        assert metadata["subset_kind"] == "leaf"
        assert metadata["question"]
        assert len(metadata["choice_texts"]) == 4

    return validate


def _metadata_has_haerae_fields(*, subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata has haerae fields."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subset"] == subset
        assert metadata["dataset_name"]
        assert metadata["query"].endswith("### 정답:")
        assert metadata["answer"] in {"(A)", "(B)", "(C)", "(D)", "(E)"}
        assert len(metadata["raw_choices"]) == 5

    return validate


def _metadata_has_fld_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has fld fields."""
    assert metadata["proof_label"] in {"PROVED", "DISPROVED", "UNKNOWN"}
    assert metadata["world_assump_label"] in {"PROVED", "DISPROVED", "UNKNOWN"}
    assert metadata["negative_world_assump_label"] in {"PROVED", "DISPROVED", "UNKNOWN", "None", None}
    assert metadata["num_formula_distractors"] >= 0
    assert metadata["num_translation_distractors"] >= 0
    assert metadata["num_all_distractors"] >= 0


def _metadata_has_french_bench_arc_challenge_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has french bench ARC challenge fields."""
    assert metadata["id"]
    assert metadata["choice_labels"] == ["A", "B", "C", "D"]


def _metadata_has_kormedmcqa_fields(*, subset: str | None = None, allowed_subsets: set[str] | None = None) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata has kormedmcqa fields."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        if subset is not None:
            assert metadata["subset"] == subset
            assert metadata["subject"] == subset
        if allowed_subsets is not None:
            assert metadata["subset"] in allowed_subsets
            assert metadata["subject"] in allowed_subsets
        assert metadata["year"] >= 0
        assert metadata["period"] >= 0
        assert metadata["q_number"] >= 0
        assert len(metadata["raw_choices"]) == 5

    return validate


def _metadata_has_gsm_plus_fields(metadata: dict[str, Any]) -> None:
    """Support the surrounding tests with metadata has gsm plus fields."""
    assert metadata["perturbation_type"] is not None
    assert metadata["seed_question"]
    assert "seed_solution" in metadata
    assert metadata["seed_answer"]


def _metadata_subset_in(allowed_subsets: set[str] | None = None) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata subset in."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        subset = metadata["subset"]
        assert subset
        assert metadata["subset_kind"] == "leaf"
        assert metadata["subset_path"] == subset.split(".")
        assert metadata["subset_value"]
        if allowed_subsets is not None:
            assert subset in allowed_subsets
        assert len(metadata["choice_texts"]) == 4

    return validate


def _arithmetic_suite_spec(task_name: str, baseline: float) -> SuiteSpec:
    """Support the surrounding tests with arithmetic suite spec."""
    return SuiteSpec(
        suite_factory=lambda task_name=task_name: getattr(evalution.benchmarks, task_name)(
            batch_size=24,
            stream=True,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline={
            "acc,ll": baseline,
        },
        expected_metrics=frozenset({"acc,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/arithmetic",
            "dataset_name": task_name,
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, task_name=task_name: _assert_arithmetic_sample(
            sample,
            index,
            task_name=task_name,
        ),
    )


def _bbh_suite_spec(task_name: str, subset: str, baseline: float) -> SuiteSpec:
    """Support the surrounding tests with BBH suite spec."""
    return SuiteSpec(
        suite_factory=lambda task_name=task_name: getattr(evalution.benchmarks, task_name)(
            batch_size=4,
            max_rows=32,
        ),
        expected_name=task_name,
        baseline={"em": baseline},
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "lukaemon/bbh",
            "dataset_name": subset,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, subset=subset: _assert_generated_exact_match_sample(
            sample,
            index,
            prompt_prefix="Q: ",
            prompt_suffix="\nA:",
            metadata_validator=_metadata_bbh_subset(subset),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _babilong_suite_spec(task_name: str, qa_split: str, baseline: float) -> SuiteSpec:
    """Support the surrounding tests with babilong suite spec."""
    return SuiteSpec(
        suite_factory=lambda task_name=task_name: getattr(evalution.benchmarks, task_name)(
            batch_size=4,
            max_rows=32,
        ),
        expected_name=task_name,
        baseline={"em": baseline},
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "RMT-team/babilong",
            "dataset_name": "0k",
            "split": qa_split,
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
            "context_length": "0k",
            "qa_split": qa_split,
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, qa_split=qa_split: _assert_babilong_sample(
            sample,
            index,
            qa_split=qa_split,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _metadata_language_and_id(language: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata language and id."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["language"] == language
        assert metadata["id"] is not None

    return validate


def _metadata_language_and_idx(language: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata language and idx."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["language"] == language
        assert metadata["idx"] is not None
        assert metadata["question"] in {"cause", "effect"}
        assert len(metadata["raw_choices"]) == 2

    return validate


def _paws_x_suite_spec(
    task_name: str,
    *,
    language: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with paws x suite spec."""
    return SuiteSpec(
        suite_factory=lambda language=language: evalution.benchmarks.paws_x(
            language=language,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset(
            {
                "acc,ll",
                "acc,ll_avg",
                "f1,ll_boolean",
                "f1,ll_avg_boolean",
            }
        ),
        expected_metadata={
            "stream": False,
            "dataset_path": "paws-x",
            "dataset_name": language,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, language=language: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"yes", "no"},
            prediction_values={"yes", "no"},
            prompt_substrings=(
                "Sentence 1: ",
                "\nSentence 2: ",
                "\nQuestion: Do both sentences mean the same thing?\nAnswer:",
            ),
            metadata_validator=_metadata_language_and_id(language),
        ),
    )


def _xcopa_suite_spec(
    task_name: str,
    *,
    language: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with XCOPA suite spec."""
    return SuiteSpec(
        suite_factory=lambda language=language: evalution.benchmarks.xcopa(
            language=language,
            batch_size=24,
            max_rows=100,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "xcopa",
            "dataset_name": language,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=100,
        sample_validator=lambda sample, index, language=language: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B"},
            prediction_values={"A", "B"},
            prompt_substrings=(
                "Premise: ",
                "\nQuestion: Which option is the more likely ",
                "\nA. ",
                "\nB. ",
                "\nAnswer:",
            ),
            metadata_validator=_metadata_language_and_idx(language),
        ),
    )


def _afrixnli_suite_spec(
    *,
    language: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with afrixnli suite spec."""
    return SuiteSpec(
        suite_factory=lambda language=language: evalution.benchmarks.afrixnli(
            language=language,
            batch_size=24,
            max_rows=32,
        ),
        expected_name=f"afrixnli_{language}",
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "masakhane/afrixnli",
            "dataset_name": language,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, language=language: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"entailment", "neutral", "contradiction"},
            prediction_values={"entailment", "neutral", "contradiction"},
            prompt_substrings=(
                "Premise: ",
                "\nHypothesis: ",
                "\nQuestion: What is the relationship between the premise and hypothesis: entailment, neutral, or contradiction?\nAnswer:",
            ),
            metadata_validator=_metadata_afrixnli_language(language),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _xnli_suite_spec(
    *,
    language: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with XNLI suite spec."""
    return SuiteSpec(
        suite_factory=lambda language=language: evalution.benchmarks.xnli(
            language=language,
            batch_size=24,
            max_rows=32,
        ),
        expected_name=f"xnli_{language}",
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "facebook/xnli",
            "dataset_name": language,
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, language=language: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"entailment", "neutral", "contradiction"},
            prediction_values={"entailment", "neutral", "contradiction"},
            prompt_substrings=(
                "Premise: ",
                "\nHypothesis: ",
                "\nQuestion: What is the relationship between the premise and hypothesis: entailment, neutral, or contradiction?\nAnswer:",
            ),
            metadata_validator=_metadata_xnli_language(language),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _xquad_suite_spec(
    *,
    language: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with XQuAD suite spec."""
    return SuiteSpec(
        suite_factory=lambda language=language: evalution.benchmarks.xquad(
            language=language,
            batch_size=16,
            max_rows=32,
        ),
        expected_name=f"xquad_{language}",
        baseline=baseline,
        expected_metrics=frozenset({"em", "f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "google/xquad",
            "dataset_name": f"xquad.{language}",
            "split": "validation",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, language=language: _assert_xquad_sample(
            sample,
            index,
            language=language,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _truthfulqa_suite_spec(
    *,
    variant: str,
    baseline: float,
) -> SuiteSpec:
    """Support the surrounding tests with truthfulqa suite spec."""
    return SuiteSpec(
        suite_factory=lambda variant=variant: evalution.benchmarks.truthfulqa(
            variant=variant,
            batch_size=16,
            max_rows=32,
        ),
        expected_name=f"truthfulqa_{variant}",
        baseline={"acc": baseline},
        expected_metrics=frozenset({"acc"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "truthfulqa/truthful_qa",
            "dataset_name": "multiple_choice",
            "split": "validation",
            "scoring_mode": f"truthfulqa_{variant}_multiple_choice",
            "primary_metric": "acc",
            "variant": variant,
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, variant=variant: _assert_truthfulqa_sample(
            sample,
            index,
            variant=variant,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _inverse_scaling_suite_spec(
    *,
    task_name: str,
    subset: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with inverse scaling suite spec."""
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.inverse_scaling(
            subset=subset,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "pminervini/inverse-scaling",
            "dataset_name": subset,
            "split": "data",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, subset=subset: _assert_inverse_scaling_sample(
            sample,
            index,
            subset=subset,
        ),
    )


def _belebele_suite_spec(
    *,
    language: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with belebele suite spec."""
    return SuiteSpec(
        suite_factory=lambda language=language: evalution.benchmarks.belebele(
            language=language,
            batch_size=4,
            max_rows=32,
        ),
        expected_name=f"belebele_{language}",
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "facebook/belebele",
            "dataset_name": language,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, language=language: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_substrings=(
                "P: ",
                "\nQ: ",
                "\nA: ",
                "\nB: ",
                "\nC: ",
                "\nD: ",
                "\nAnswer:",
            ),
            metadata_validator=_metadata_belebele_language(language),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _bangla_suite_spec(
    *,
    subset: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with bangla suite spec. Preserve the fallback order expected by the surrounding caller."""
    expected_metadata = {
        "stream": False,
        "split": "validation",
        "subset": subset,
        "scoring_mode": "multiple_choice_loglikelihood",
    }
    target_values: set[str] | None = None
    prediction_values: set[str] | None = None
    prompt_prefix: str | None = None
    prompt_suffix: str | None = None
    prompt_substrings: tuple[str, ...] = ()

    if subset == "boolqa":
        expected_metadata["dataset_path"] = "hishab/boolq_bn"
        expected_metadata["dataset_name"] = None
        target_values = {"yes", "no"}
        prediction_values = {"yes", "no"}
        prompt_prefix = "Passage:\n"
        prompt_suffix = "\n\nAnswer:"
        prompt_substrings = ("\n\nQuestion:\n",)
    elif subset == "commonsenseqa":
        expected_metadata["dataset_path"] = "hishab/commonsenseqa-bn"
        expected_metadata["dataset_name"] = None
        target_values = {"A", "B", "C", "D", "E"}
        prediction_values = {"A", "B", "C", "D", "E"}
        prompt_suffix = "\nAnswer:"
        prompt_substrings = ("\nA. ", "\nB. ", "\nC. ", "\nD. ", "\nE. ")
    elif subset == "mmlu":
        expected_metadata["dataset_path"] = "hishab/titulm-bangla-mmlu"
        expected_metadata["dataset_name"] = "all"
        expected_metadata["split"] = "test"
        target_values = {"A", "B", "C", "D"}
        prediction_values = {"A", "B", "C", "D"}
        prompt_suffix = " Answer:"
        prompt_substrings = (" A. ", " B. ", " C. ", " D. ")
    elif subset == "openbookqa":
        expected_metadata["dataset_path"] = "hishab/openbookqa-bn"
        expected_metadata["dataset_name"] = None
        expected_metadata["split"] = "test"
        target_values = {"A", "B", "C", "D"}
        prediction_values = {"A", "B", "C", "D"}
        prompt_suffix = "\nAnswer:"
        prompt_substrings = ("\nA. ", "\nB. ", "\nC. ", "\nD. ")
    elif subset == "piqa":
        expected_metadata["dataset_path"] = "hishab/piqa-bn"
        expected_metadata["dataset_name"] = None
        target_values = {"A", "B"}
        prediction_values = {"A", "B"}
        prompt_suffix = "\nAnswer:"
        prompt_substrings = ("\nA. ", "\nB. ")
    else:
        raise AssertionError(f"unsupported bangla subset spec: {subset!r}")

    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.bangla(
            subset=subset,
            batch_size=24,
            max_rows=32,
        ),
        expected_name=f"bangla_{subset}",
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata=expected_metadata,
        expected_sample_count=32,
        sample_validator=lambda sample, index, subset=subset, target_values=target_values, prediction_values=prediction_values, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, prompt_substrings=prompt_substrings: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values=target_values,
            prediction_values=prediction_values,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            prompt_substrings=prompt_substrings,
            metadata_validator=_metadata_bangla_subset(subset),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _arc_mt_suite_spec(
    *,
    language: str,
    dataset_path: str,
    dataset_name: str | None,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with ARC mt suite spec."""
    return SuiteSpec(
        suite_factory=lambda language=language: evalution.benchmarks.arc_mt(
            language=language,
            batch_size=24,
            max_rows=32,
        ),
        expected_name=f"arc_mt_{language}",
        baseline=baseline,
        expected_metrics=frozenset({"acc,exam"}),
        expected_metadata={
            "stream": False,
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "split": "test",
            "scoring_mode": "multiple_choice_exam_score",
            "scoring_reference": "clark2018arc arc-solvers calculate_scores.py",
            "language": language,
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, language=language: _assert_arc_mt_sample(
            sample,
            index,
            language=language,
        ),
        result_validator=_validate_arc_exam_result,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _kobest_suite_spec(
    *,
    subset: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with kobest suite spec. Preserve the fallback order expected by the surrounding caller."""
    target_values: set[str] | None = None
    prediction_values: set[str] | None = None
    prompt_prefix: str | None = None
    prompt_suffix: str | None = None
    prompt_substrings: tuple[str, ...] = ()

    if subset == "boolq":
        target_values = {"아니오", "예"}
        prediction_values = {"아니오", "예"}
        prompt_prefix = "지문: "
        prompt_suffix = "\n답변:"
        prompt_substrings = ("\n질문: ",)
    elif subset == "copa":
        prompt_prefix = None
    elif subset == "hellaswag":
        prompt_prefix = None
    elif subset == "sentineg":
        target_values = {"부정", "긍정"}
        prediction_values = {"부정", "긍정"}
        prompt_prefix = "문장: "
        prompt_suffix = "\n답변:"
        prompt_substrings = ("\n질문: 이 문장의 감성은 무엇입니까?\n",)
    elif subset == "wic":
        target_values = {"아니오", "예"}
        prediction_values = {"아니오", "예"}
        prompt_prefix = "문장 1: "
        prompt_suffix = "\n답변:"
        prompt_substrings = ("\n문장 2: ", "\n질문: 두 문장에서 '")
    else:
        raise AssertionError(f"unsupported kobest subset spec: {subset!r}")

    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.kobest(
            subset=subset,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=f"kobest_{subset}",
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "skt/kobest_v1",
            "dataset_name": subset,
            "split": "test",
            "subset": subset,
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, subset=subset, target_values=target_values, prediction_values=prediction_values, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, prompt_substrings=prompt_substrings: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values=target_values,
            prediction_values=prediction_values,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            prompt_substrings=prompt_substrings,
            metadata_validator=_metadata_kobest_subset(subset),
        ),
    )


def _gpqa_suite_spec(
    *,
    subset: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with GPQA suite spec."""
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.gpqa(
            subset=subset,
            batch_size=4,
            max_rows=32,
            max_new_tokens=64,
        ),
        expected_name=f"gpqa_{subset}",
        baseline=baseline,
        expected_metrics=frozenset({"em,choice_label"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "Idavidrein/gpqa",
            "dataset_name": f"gpqa_{subset}",
            "split": "train",
            "subset": subset,
            "shuffle_seed": 0,
            "prompt_variant": "author_zero_shot_label_response",
            "choice_order_mode": "seeded_shuffle",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_choice_label_exact_match",
            "primary_metric": "em,choice_label",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, subset=subset: _assert_gpqa_sample(
            sample,
            index,
            subset=subset,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _arabicmmlu_suite_spec(
    *,
    task_name: str,
    subset: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with arabicmmlu suite spec."""
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.arabicmmlu(
            subset=subset,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "MBZUAI/ArabicMMLU",
            "dataset_name": subset,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, subset=subset: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_prefix="This is a ",
            prompt_suffix="\n\nAnswer:",
            prompt_substrings=("\n\nQuestion: ",),
            metadata_validator=_metadata_arabicmmlu_subset(subset),
        ),
    )


def _blimp_suite_spec(
    *,
    subset: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with blimp suite spec."""
    task_name = f"blimp_{subset.lower()}"
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.blimp(
            subset=subset,
            batch_size=32,
            stream=True,
            max_rows=32,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "blimp",
            "dataset_name": subset,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "full_sentence_pair",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, subset=subset: _assert_blimp_sample(
            sample,
            index,
            subset=subset,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _ceval_suite_spec(
    *,
    subset: str,
    baseline: dict[str, float],
    expected_sample_count: int,
) -> SuiteSpec:
    """Support the surrounding tests with ceval suite spec."""
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.ceval(
            subset=subset,
            batch_size=24,
            stream=True,
            max_rows=32,
        ),
        expected_name=f"ceval_{subset}",
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "ceval/ceval-exam",
            "dataset_name": subset,
            "split": "val",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=expected_sample_count,
        sample_validator=lambda sample, index, subset=subset: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_suffix="\n答案：",
            metadata_validator=_metadata_ceval_subset(subset),
        ),
        abs_tolerance=2 / expected_sample_count,
    )


def _crows_pairs_suite_spec(
    task_name: str,
    *,
    baseline: dict[str, float],
    expected_sample_count: int,
) -> SuiteSpec:
    """Support the surrounding tests with crows pairs suite spec."""
    suffix = task_name.removeprefix("crows_pairs_")
    language, _, bias_type = suffix.partition("_")
    resolved_bias_type = bias_type or None
    return SuiteSpec(
        suite_factory=lambda task_name=task_name: getattr(evalution.benchmarks, task_name)(
            max_rows=32,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"pct_stereotype", "ll_diff"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "jannalu/crows_pairs_multilingual",
            "dataset_name": language,
            "split": "test",
            "language": language,
            "bias_type": resolved_bias_type,
            "scoring_mode": "pairwise_sentence_loglikelihood_bias_preference",
            "primary_metric": "pct_stereotype",
            "prompt_variant": "empty_context_full_sentence",
        },
        expected_sample_count=expected_sample_count,
        sample_validator=lambda sample, index, language=language, bias_type=resolved_bias_type: _assert_crows_pairs_sample(
            sample,
            index,
            language=language,
            bias_type=bias_type,
        ),
        abs_tolerance=max(2 / expected_sample_count, 0.15),
    )


def _aime_suite_spec(
    task_name: str,
    *,
    dataset_path: str,
    split: str,
    baseline: float,
) -> SuiteSpec:
    """Support the surrounding tests with aime suite spec."""
    return SuiteSpec(
        suite_factory=lambda task_name=task_name: getattr(evalution.benchmarks, task_name)(
            batch_size=24,
            max_new_tokens=512,
            stream=True,
            max_rows=30,
        ),
        expected_name=task_name,
        baseline={"em": baseline},
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": True,
            "dataset_path": dataset_path,
            "dataset_name": None,
            "split": split,
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_math_exact_match",
            "primary_metric": "em",
        },
        expected_sample_count=30,
        sample_validator=_assert_aime_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _cmmlu_suite_spec(
    task_name: str,
    *,
    subset: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with cmmlu suite spec."""
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.cmmlu(
            subset=subset,
            num_fewshot=5,
            batch_size=24,
            max_rows=32,
            stream=False,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "haonan-li/cmmlu",
            "dataset_name": subset,
            "split": "test",
            "fewshot_split": "dev",
            "num_fewshot": 5,
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, subset=subset: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="以下是单项选择题，请直接给出正确答案的选项。",
            prompt_suffix="答案：",
            metadata_validator=_metadata_cmmlu_subset(subset),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _kmmlu_suite_spec(
    task_name: str,
    *,
    subset: str,
    dataset_name: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with kmmlu suite spec."""
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.kmmlu(
            subset=subset,
            num_fewshot=5,
            batch_size=24,
            max_rows=32,
            stream=False,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "HAERAE-HUB/KMMLU",
            "dataset_name": dataset_name,
            "split": "test",
            "fewshot_split": "dev",
            "num_fewshot": 5,
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, subset=subset: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_suffix="정답：",
            prompt_substrings=("\nA. ", "\nD. "),
            metadata_validator=_metadata_kmmlu_subset(subset),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _mgsm_suite_spec(
    task_name: str,
    *,
    language: str,
    baseline: float,
) -> SuiteSpec:
    """Support the surrounding tests with mgsm suite spec."""
    return SuiteSpec(
        suite_factory=lambda language=language: evalution.benchmarks.mgsm(
            language=language,
            batch_size=24,
            max_new_tokens=96,
            stream=True,
            max_rows=32,
        ),
        expected_name=task_name,
        baseline={"acc,num": baseline},
        expected_metrics=frozenset({"acc,num"}),
        expected_metadata={
            "variant": "base",
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
            "stream": True,
            "generation_submission_mode": "continuous_refill",
            "num_fewshot": 0,
            "dataset_path": "juletxara/mgsm",
            "dataset_name": language,
            "split": "test",
            "language": language,
            "scoring_mode": "numeric_format_insensitive",
            "primary_metric": "acc,num",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, language=language: _assert_afrimgsm_sample(
            sample,
            index,
            language=language,
        ),
        result_validator=_validate_gsm8k_like_result,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _mmlu_cf_suite_spec(
    task_name: str,
    *,
    subject: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with MMLU cf suite spec."""
    return SuiteSpec(
        suite_factory=lambda subject=subject: evalution.benchmarks.mmlu_cf(
            subject=subject,
            num_fewshot=5,
            batch_size=24,
            max_rows=32,
            stream=False,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "microsoft/MMLU-CF",
            "dataset_name": subject,
            "split": "val",
            "fewshot_split": "dev",
            "num_fewshot": 5,
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, subject=subject: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="There is a single choice question (with answers). Answer the question by replying A, B, C or D.",
            prompt_suffix="Answer:",
            metadata_validator=_metadata_mmlu_cf_subject(subject),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _agieval_suite_spec(
    task_name: str,
    *,
    subset: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with agieval suite spec."""
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.agieval(
            subset=subset,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "RUCAIBox/AGIEval",
            "dataset_name": subset,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, subset=subset: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("Question: ", "\nAnswer:"),
            metadata_validator=_metadata_agieval_subset(subset),
        ),
    )


def _hendrycks_math_suite_spec(
    task_name: str,
    *,
    subset: str,
    baseline: float,
) -> SuiteSpec:
    """Support the surrounding tests with hendrycks math suite spec."""
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.hendrycks_math(
            subset=subset,
            batch_size=4,
            max_new_tokens=256,
            stream=True,
            max_rows=32,
        ),
        expected_name=task_name,
        baseline={"em": baseline},
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/hendrycks_math",
            "dataset_name": subset,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_math_exact_match",
            "primary_metric": "em",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index, subset=subset: _assert_hendrycks_math_sample(
            sample,
            index,
            subset=subset,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _afrimgsm_suite_spec(
    task_name: str,
    *,
    language: str,
    baseline: float,
) -> SuiteSpec:
    """Support the surrounding tests with afrimgsm suite spec."""
    return SuiteSpec(
        suite_factory=lambda language=language: evalution.benchmarks.afrimgsm(
            language=language,
            batch_size=24,
            max_new_tokens=96,
            stream=True,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline={"acc,num": baseline},
        expected_metrics=frozenset({"acc,num"}),
        expected_metadata={
            "variant": "base",
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
            "stream": True,
            "generation_submission_mode": "continuous_refill",
            "num_fewshot": 0,
            "dataset_path": "masakhane/afrimgsm",
            "dataset_name": language,
            "split": "test",
            "language": language,
            "scoring_mode": "numeric_format_insensitive",
            "primary_metric": "acc,num",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, language=language: _assert_afrimgsm_sample(
            sample,
            index,
            language=language,
        ),
        result_validator=_validate_gsm8k_like_result,
    )


def _afrimmlu_suite_spec(
    task_name: str,
    *,
    language: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with afrimmlu suite spec."""
    return SuiteSpec(
        suite_factory=lambda language=language: evalution.benchmarks.afrimmlu(
            language=language,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "masakhane/afrimmlu",
            "dataset_name": language,
            "split": "test",
            "language": language,
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, language=language: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="Question: ",
            prompt_substrings=("\nA. ", "\nB. ", "\nC. ", "\nD. ", "\nAnswer:"),
            metadata_validator=_metadata_afrimmlu_language(language),
        ),
    )


def _metadata_darijammlu_subset(subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata darijammlu subset."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subset"] == subset
        assert metadata["subject"]
        assert metadata["subject_darija"]
        assert isinstance(metadata["raw_choices"], list)
        assert 2 <= len(metadata["raw_choices"]) <= 4

    return validate


def _assert_darijammlu_sample(sample: Any, index: int, *, subset: str) -> None:
    """Assert darijammlu sample for the surrounding tests."""
    _metadata_darijammlu_subset(subset)(sample.metadata)
    choice_count = len(sample.metadata["raw_choices"])
    choice_labels = tuple(chr(ord("A") + offset) for offset in range(choice_count))
    _assert_multiple_choice_loglikelihood_sample(
        sample,
        index,
        target_values=set(choice_labels),
        prediction_values=set(choice_labels),
        prompt_prefix="This is a DarijaMMLU multiple-choice question about ",
        prompt_substrings=("\nQuestion: ", *(f"\n{label}. " for label in choice_labels), "\nAnswer:"),
    )


def _darijammlu_suite_spec(
    task_name: str,
    *,
    subset: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with darijammlu suite spec."""
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.darijammlu(
            subset=subset,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "MBZUAI-Paris/DarijaMMLU",
            "dataset_name": subset,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, subset=subset: _assert_darijammlu_sample(
            sample,
            index,
            subset=subset,
        ),
    )


def _metadata_egymmlu_subset(subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata egymmlu subset."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subset"] == subset
        assert metadata["subject"]
        assert metadata["egy_subject"]
        assert isinstance(metadata["raw_choices"], list)
        assert 2 <= len(metadata["raw_choices"]) <= 4

    return validate


def _assert_egymmlu_sample(sample: Any, index: int, *, subset: str) -> None:
    """Assert egymmlu sample for the surrounding tests."""
    _metadata_egymmlu_subset(subset)(sample.metadata)
    choice_count = len(sample.metadata["raw_choices"])
    choice_labels = tuple(chr(ord("A") + offset) for offset in range(choice_count))
    _assert_multiple_choice_loglikelihood_sample(
        sample,
        index,
        target_values=set(choice_labels),
        prediction_values=set(choice_labels),
        prompt_prefix="This is a EgyMMLU multiple-choice question about ",
        prompt_substrings=("\nQuestion: ", *(f"\n{label}. " for label in choice_labels), "\nAnswer:"),
    )


def _egymmlu_suite_spec(
    task_name: str,
    *,
    subset: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with egymmlu suite spec."""
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.egymmlu(
            subset=subset,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "UBC-NLP/EgyMMLU",
            "dataset_name": subset,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, subset=subset: _assert_egymmlu_sample(
            sample,
            index,
            subset=subset,
        ),
    )


def _metadata_eus_exams_subset(subset: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata eus exams subset."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["subset"] == subset
        assert metadata["language"] in {"eu", "es"}
        assert metadata["question_id"]
        assert metadata["link"].startswith("https://")
        assert isinstance(metadata["raw_choices"], list)
        assert len(metadata["raw_choices"]) == 4

    return validate


def _eus_exams_suite_spec(
    task_name: str,
    *,
    subset: str,
    baseline: dict[str, float],
    expected_sample_count: int,
    abs_tolerance: float | None = None,
) -> SuiteSpec:
    """Support the surrounding tests with eus exams suite spec."""
    return SuiteSpec(
        suite_factory=lambda subset=subset: evalution.benchmarks.eus_exams(
            subset=subset,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "HiTZ/EusExams",
            "dataset_name": subset,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=expected_sample_count,
        sample_validator=lambda sample, index, subset=subset: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="Question: ",
            prompt_substrings=("\nA. ", "\nB. ", "\nC. ", "\nD. ", "\nAnswer:"),
            metadata_validator=_metadata_eus_exams_subset(subset),
        ),
        abs_tolerance=(2 / expected_sample_count) if abs_tolerance is None else abs_tolerance,
    )


def _metadata_careqa_language(language: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata careqa language."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["language"] == language
        assert metadata["category"]
        assert metadata["exam_id"] > 0
        assert metadata["year"] >= 2020
        assert metadata["unique_id"]
        assert len(metadata["raw_choices"]) == 4

    return validate


def _careqa_suite_spec(
    task_name: str,
    *,
    language: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with careqa suite spec."""
    return SuiteSpec(
        suite_factory=lambda language=language: evalution.benchmarks.careqa(
            language=language,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "HPAI-BSC/CareQA",
            "dataset_name": f"CareQA_{language}",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, language=language: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="Question: ",
            prompt_substrings=("\nA. ", "\nB. ", "\nC. ", "\nD. ", "\nAnswer:"),
            metadata_validator=_metadata_careqa_language(language),
        ),
    )


def _metadata_cabbq_category(category: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata cabbq category."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["category"] == category
        assert metadata["question_polarity"]
        assert metadata["context_condition"]
        assert metadata["question_type"]
        assert len(metadata["raw_choices"]) == 3

    return validate


def _cabbq_suite_spec(
    task_name: str,
    *,
    category: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with cabbq suite spec."""
    return SuiteSpec(
        suite_factory=lambda category=category: evalution.benchmarks.cabbq(
            category=category,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "BSC-LT/CaBBQ",
            "dataset_name": category,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, category=category: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C"},
            prediction_values={"A", "B", "C"},
            prompt_prefix="Context: ",
            prompt_substrings=("\nQuestion: ", "\nA. ", "\nB. ", "\nC. ", "\nAnswer:"),
            metadata_validator=_metadata_cabbq_category(category),
        ),
    )


def _metadata_bbq_category(category: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata bbq category."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["category"] == category
        assert metadata["question_polarity"]
        assert metadata["context_condition"]
        assert metadata["question_index"]
        assert len(metadata["raw_choices"]) == 3

    return validate


def _bbq_suite_spec(
    task_name: str,
    *,
    category: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with bbq suite spec."""
    return SuiteSpec(
        suite_factory=lambda category=category: evalution.benchmarks.bbq(
            category=category,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "heegyu/bbq",
            "dataset_name": category,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, category=category: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C"},
            prediction_values={"A", "B", "C"},
            prompt_prefix="Context: ",
            prompt_substrings=("\nQuestion: ", "\nA. ", "\nB. ", "\nC. ", "\nAnswer:"),
            metadata_validator=_metadata_bbq_category(category),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    )


def _metadata_esbbq_category(category: str) -> Callable[[dict[str, Any]], None]:
    """Support the surrounding tests with metadata esbbq category."""
    def validate(metadata: dict[str, Any]) -> None:
        """Validate validate."""
        assert metadata["category"] == category
        assert metadata["question_polarity"]
        assert metadata["context_condition"]
        assert metadata["question_type"]
        assert len(metadata["raw_choices"]) == 3

    return validate


def _esbbq_suite_spec(
    task_name: str,
    *,
    category: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with esbbq suite spec."""
    return SuiteSpec(
        suite_factory=lambda category=category: evalution.benchmarks.esbbq(
            category=category,
            batch_size=24,
            max_rows=128,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "BSC-LT/EsBBQ",
            "dataset_name": category,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index, category=category: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C"},
            prediction_values={"A", "B", "C"},
            prompt_prefix="Context: ",
            prompt_substrings=("\nQuestion: ", "\nA. ", "\nB. ", "\nC. ", "\nAnswer:"),
            metadata_validator=_metadata_esbbq_category(category),
        ),
    )


def _graphwalks_sample_validator(sample: Any, index: int) -> None:
    """Support the surrounding tests with graphwalks sample validator."""
    metadata = sample.metadata
    assert sample.index == index
    assert metadata.get("problem_type")
    assert metadata.get("prompt_chars", 0) > 0
    assert isinstance(sample.extracted.get("prediction_nodes_strict"), list)
    assert isinstance(sample.extracted.get("prediction_nodes_flexible"), list)


def _graphwalks_suite_spec(
    task_name: str,
    *,
    data_file: str,
    baseline: dict[str, float],
) -> SuiteSpec:
    """Support the surrounding tests with graphwalks suite spec."""
    return SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.graphwalks_128k(
            max_rows=1,
            batch_size=1,
        ),
        expected_name=task_name,
        baseline=baseline,
        expected_metrics=frozenset({"f1", "flexible_f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "openai/graphwalks",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "graphwalks_set_f1",
            "data_file": data_file,
        },
        expected_sample_count=1,
        sample_validator=_graphwalks_sample_validator,
    )


# Keep shared test fixtures and expectations explicit at module scope.
SUITE_SPECS = {
    "cabbq_age": _cabbq_suite_spec(
        "cabbq_age",
        category="Age",
        baseline={"acc,ll": 0.3828125, "acc,ll_avg": 0.3828125},
    ),
    "cabbq_disability_status": _cabbq_suite_spec(
        "cabbq_disability_status",
        category="DisabilityStatus",
        baseline={"acc,ll": 0.546875, "acc,ll_avg": 0.546875},
    ),
    "cabbq_gender": _cabbq_suite_spec(
        "cabbq_gender",
        category="Gender",
        baseline={"acc,ll": 0.3984375, "acc,ll_avg": 0.3984375},
    ),
    "cabbq_nationality": _cabbq_suite_spec(
        "cabbq_nationality",
        category="Nationality",
        baseline={"acc,ll": 0.3984375, "acc,ll_avg": 0.3984375},
    ),
    "careqa_en": _careqa_suite_spec(
        "careqa_en",
        language="en",
        baseline={"acc,ll": 0.3359375, "acc,ll_avg": 0.3359375},
    ),
    "careqa_es": _careqa_suite_spec(
        "careqa_es",
        language="es",
        baseline={"acc,ll": 0.2890625, "acc,ll_avg": 0.2890625},
    ),
    "esbbq_age": _esbbq_suite_spec(
        "esbbq_age",
        category="Age",
        baseline={"acc,ll": 0.3828125, "acc,ll_avg": 0.3828125},
    ),
    "esbbq_disability_status": _esbbq_suite_spec(
        "esbbq_disability_status",
        category="DisabilityStatus",
        baseline={"acc,ll": 0.4609375, "acc,ll_avg": 0.4609375},
    ),
    "esbbq_gender": _esbbq_suite_spec(
        "esbbq_gender",
        category="Gender",
        baseline={"acc,ll": 0.3828125, "acc,ll_avg": 0.3828125},
    ),
    "esbbq_nationality": _esbbq_suite_spec(
        "esbbq_nationality",
        category="Nationality",
        baseline={"acc,ll": 0.4765625, "acc,ll_avg": 0.4765625},
    ),
    "graphwalks_128k": _graphwalks_suite_spec(
        "graphwalks_128k",
        data_file="graphwalks_128k_and_shorter.parquet",
        baseline={"f1": 0.0, "flexible_f1": 0.0},
    ),
    "groundcocoa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.groundcocoa(batch_size=8, max_rows=128),
        expected_name="groundcocoa",
        baseline={
            "acc,ll": 0.2265625,
            "acc,ll_avg": 0.2265625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "harsh147/GroundCocoa",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "flight_criteria_with_option_labels",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={
                "The answer is Option A",
                "The answer is Option B",
                "The answer is Option C",
                "The answer is Option D",
                "The answer is Option E",
            },
            prediction_values={
                "The answer is Option A",
                "The answer is Option B",
                "The answer is Option C",
                "The answer is Option D",
                "The answer is Option E",
            },
            prompt_prefix="A user has specified certain criteria for booking a flight.",
            metadata_validator=_metadata_has_groundcocoa_fields,
        ),
    ),
    "eus_exams_eu_opeosakiadmineu": _eus_exams_suite_spec(
        "eus_exams_eu_opeosakiadmineu",
        subset="eu_opeosakiadmineu",
        baseline={"acc,ll": 0.28125, "acc,ll_avg": 0.28125},
        expected_sample_count=128,
    ),
    "eus_exams_eu_opeosakiauxenfeu": _eus_exams_suite_spec(
        "eus_exams_eu_opeosakiauxenfeu",
        subset="eu_opeosakiauxenfeu",
        baseline={"acc,ll": 0.2734375, "acc,ll_avg": 0.2734375},
        expected_sample_count=128,
    ),
    "eus_exams_es_ejadministrativo": _eus_exams_suite_spec(
        "eus_exams_es_ejadministrativo",
        subset="es_ejadministrativo",
        baseline={"acc,ll": 0.3779527559055118, "acc,ll_avg": 0.3779527559055118},
        expected_sample_count=127,
        abs_tolerance=3 / 127,
    ),
    "eus_exams_es_ejauxiliar": _eus_exams_suite_spec(
        "eus_exams_es_ejauxiliar",
        subset="es_ejauxiliar",
        baseline={"acc,ll": 0.3700787401574803, "acc,ll_avg": 0.3700787401574803},
        expected_sample_count=127,
    ),
    "eus_reading": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.eus_reading(batch_size=1, stream=True, max_rows=128),
        expected_name="eus_reading",
        baseline={
            "acc,ll": 0.28125,
            "acc,ll_avg": 0.28125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "HiTZ/EusReading",
            "dataset_name": "default",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "order": "native",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="Pasartea: ",
            prompt_suffix="\nErantzuna:",
            metadata_validator=_metadata_has_eus_reading_fields,
        ),
    ),
    "eus_proficiency": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.eus_proficiency(batch_size=24, stream=True, max_rows=128),
        expected_name="eus_proficiency",
        baseline={
            "acc,ll": 0.21875,
            "acc,ll_avg": 0.21875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "HiTZ/EusProficiency",
            "dataset_name": "default",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "order": "native",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="Galdera: ",
            prompt_suffix="\nErantzuna:",
            metadata_validator=_metadata_has_eus_proficiency_fields,
        ),
    ),
    "eus_trivia": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.eus_trivia(batch_size=24, stream=True, max_rows=128),
        expected_name="eus_trivia",
        baseline={
            "acc,ll": 0.296875,
            "acc,ll_avg": 0.296875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "HiTZ/EusTrivia",
            "dataset_name": "default",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "order": "native",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="Galdera: ",
            prompt_suffix="\nErantzuna:",
            metadata_validator=_metadata_has_eus_trivia_fields,
        ),
    ),
    "xnli_eu": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xnli_eu(batch_size=24, max_rows=128),
        expected_name="xnli_eu",
        baseline={
            "acc,ll": 0.4140625,
            "acc,ll_avg": 0.3359375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "HiTZ/xnli-eu",
            "dataset_name": "eu",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_substrings=(" ezta? ",),
            metadata_validator=_metadata_has_xnli_eu_fields,
        ),
    ),
    "toxigen": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.toxigen(batch_size=24, max_rows=128),
        expected_name="toxigen",
        baseline={
            "acc,ll": 0.4296875,
            "acc,ll_avg": 0.4296875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "skg/toxigen-data",
            "dataset_name": "annotated",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"No", "Yes"},
            prediction_values={"No", "Yes"},
            prompt_prefix="Is the following statement hateful? Respond with either Yes or No. Statement: '",
            metadata_validator=_metadata_has_toxigen_fields,
        ),
    ),
    "polemo2_in": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.polemo2_in(batch_size=24, max_rows=128),
        expected_name="polemo2_in",
        baseline={"f1": 0.4609375},
        expected_metrics=frozenset({"f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "allegro/klej-polemo2-in",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_choice_label_micro_f1",
            "primary_metric": "f1",
            "variant": "polemo2_in",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_generated_label_micro_f1_sample(
            sample,
            index,
            prompt_prefix='Opinia: "',
            prompt_suffix="Prawidłowa odpowiedź:",
            metadata_validator=lambda metadata: _metadata_has_polemo2_fields(metadata, variant="polemo2_in"),
        ),
    ),
    "polemo2_out": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.polemo2_out(batch_size=24, max_rows=128),
        expected_name="polemo2_out",
        baseline={"f1": 0.5390625},
        expected_metrics=frozenset({"f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "allegro/klej-polemo2-out",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_choice_label_micro_f1",
            "primary_metric": "f1",
            "variant": "polemo2_out",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_generated_label_micro_f1_sample(
            sample,
            index,
            prompt_prefix='Opinia: "',
            prompt_suffix="Prawidłowa odpowiedź:",
            metadata_validator=lambda metadata: _metadata_has_polemo2_fields(metadata, variant="polemo2_out"),
        ),
    ),
    "phrases_es_va": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.phrases_es_va(
            batch_size=8,
            max_rows=32,
            max_new_tokens=64,
        ),
        expected_name="phrases_es_va",
        baseline={
            "bleu": 16.6896306627536,
            "chrf": 53.013434289807336,
            "ter": 81.21019108280255,
        },
        expected_metrics=frozenset({"bleu", "chrf", "ter"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "gplsi/ES-VA_translation_test",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_translation_corpus_metrics",
            "primary_metric": "bleu",
            "direction": "es-va",
            "source_language": "es",
            "target_language": "va",
            "upstream_task": "phrases_es-va",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_translation_corpus_sample(
            sample,
            index,
            prompt_prefix="Oració en espanyol: ",
            prompt_suffix="\n\nOració en valencià:",
            metadata_validator=_metadata_has_phrases_es_fields(
                direction="es-va",
                source_language="es",
                target_language="va",
            ),
        ),
        abs_tolerance=0.05,
    ),
    "phrases_va_es": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.phrases_va_es(
            batch_size=8,
            max_rows=32,
            max_new_tokens=64,
        ),
        expected_name="phrases_va_es",
        baseline={
            "bleu": 21.130123263852585,
            "chrf": 57.05947842395071,
            "ter": 66.33333333333333,
        },
        expected_metrics=frozenset({"bleu", "chrf", "ter"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "gplsi/ES-VA_translation_test",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_translation_corpus_metrics",
            "primary_metric": "bleu",
            "direction": "va-es",
            "source_language": "va",
            "target_language": "es",
            "upstream_task": "phrases_va-es",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_translation_corpus_sample(
            sample,
            index,
            prompt_prefix="Oració en valencià: ",
            prompt_suffix="\n\nOració en espanyol:",
            metadata_validator=_metadata_has_phrases_es_fields(
                direction="va-es",
                source_language="va",
                target_language="es",
            ),
        ),
        abs_tolerance=0.05,
    ),
    "flores_es_en_es": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.flores_es(
            direction="en-es",
            batch_size=8,
            max_rows=32,
            max_new_tokens=64,
        ),
        expected_name="flores_es_en_es",
        baseline={
            "bleu": 18.828597650639605,
            "chrf": 48.82648309055004,
            "ter": 67.75929549902153,
        },
        expected_metrics=frozenset({"bleu", "chrf", "ter"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "facebook/flores",
            "dataset_name": "all",
            "split": "devtest",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_translation_corpus_metrics",
            "primary_metric": "bleu",
            "direction": "en-es",
            "source_language": "en",
            "target_language": "es",
            "upstream_task": "spanish_bench_flores_en-es",
            "archive_url": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
            "archive_sha256": "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_translation_corpus_sample(
            sample,
            index,
            prompt_prefix="English sentence: ",
            prompt_suffix="\nSpanish sentence:",
            metadata_validator=_metadata_has_flores_fields(
                direction="en-es",
                source_language="en",
                target_language="es",
            ),
        ),
        abs_tolerance=0.1,
    ),
    "flores_pt_en_pt": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.flores_pt(
            direction="en-pt",
            batch_size=8,
            max_rows=32,
            max_new_tokens=64,
        ),
        expected_name="flores_pt_en_pt",
        baseline={
            "bleu": 29.342202780864405,
            "chrf": 59.30497881186596,
            "ter": 54.3646408839779,
        },
        expected_metrics=frozenset({"bleu", "chrf", "ter"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "facebook/flores",
            "dataset_name": "all",
            "split": "devtest",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_translation_corpus_metrics",
            "primary_metric": "bleu",
            "direction": "en-pt",
            "source_language": "en",
            "target_language": "pt",
            "upstream_task": "portuguese_bench_flores_en-pt",
            "archive_url": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
            "archive_sha256": "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_translation_corpus_sample(
            sample,
            index,
            prompt_prefix="English sentence: ",
            prompt_suffix="\nPortuguese sentence:",
            metadata_validator=_metadata_has_flores_fields(
                direction="en-pt",
                source_language="en",
                target_language="pt",
            ),
        ),
        abs_tolerance=0.05,
    ),
    "egymmlu_arabic_language": _egymmlu_suite_spec(
        "egymmlu_arabic_language",
        subset="arabic_language",
        baseline={"acc,ll": 0.3359375, "acc,ll_avg": 0.3359375},
    ),
    "egymmlu_biology": _egymmlu_suite_spec(
        "egymmlu_biology",
        subset="biology",
        baseline={"acc,ll": 0.3125, "acc,ll_avg": 0.3125},
    ),
    "egymmlu_computer_science": _egymmlu_suite_spec(
        "egymmlu_computer_science",
        subset="computer_science",
        baseline={"acc,ll": 0.4296875, "acc,ll_avg": 0.4296875},
    ),
    "egymmlu_driving_test": _egymmlu_suite_spec(
        "egymmlu_driving_test",
        subset="driving_test",
        baseline={"acc,ll": 0.359375, "acc,ll_avg": 0.359375},
    ),
    "darijammlu_arabic_language": _darijammlu_suite_spec(
        "darijammlu_arabic_language",
        subset="arabic_language",
        baseline={"acc,ll": 0.3203125, "acc,ll_avg": 0.3203125},
    ),
    "darijammlu_biology": _darijammlu_suite_spec(
        "darijammlu_biology",
        subset="biology",
        baseline={"acc,ll": 0.28125, "acc,ll_avg": 0.28125},
    ),
    "darijammlu_computer_science": _darijammlu_suite_spec(
        "darijammlu_computer_science",
        subset="computer_science",
        baseline={"acc,ll": 0.5, "acc,ll_avg": 0.5},
    ),
    "darijammlu_driving_test": _darijammlu_suite_spec(
        "darijammlu_driving_test",
        subset="driving_test",
        baseline={"acc,ll": 0.3203125, "acc,ll_avg": 0.3203125},
    ),
    "afrimmlu_eng": _afrimmlu_suite_spec(
        "afrimmlu_eng",
        language="eng",
        baseline={
            "acc,ll": 0.3828125,
            "acc,ll_avg": 0.3828125,
        },
    ),
    "afrimmlu_fra": _afrimmlu_suite_spec(
        "afrimmlu_fra",
        language="fra",
        baseline={
            "acc,ll": 0.28125,
            "acc,ll_avg": 0.28125,
        },
    ),
    "afrimmlu_hau": _afrimmlu_suite_spec(
        "afrimmlu_hau",
        language="hau",
        baseline={
            "acc,ll": 0.2578125,
            "acc,ll_avg": 0.2578125,
        },
    ),
    "afrimmlu_swa": _afrimmlu_suite_spec(
        "afrimmlu_swa",
        language="swa",
        baseline={
            "acc,ll": 0.2890625,
            "acc,ll_avg": 0.2890625,
        },
    ),
    "afrimgsm_eng": _afrimgsm_suite_spec(
        "afrimgsm_eng",
        language="eng",
        baseline=0.1484375,
    ),
    "afrimgsm_fra": _afrimgsm_suite_spec(
        "afrimgsm_fra",
        language="fra",
        baseline=0.0859375,
    ),
    "afrimgsm_swa": _afrimgsm_suite_spec(
        "afrimgsm_swa",
        language="swa",
        baseline=0.046875,
    ),
    "afrimgsm_yor": _afrimgsm_suite_spec(
        "afrimgsm_yor",
        language="yor",
        baseline=0.015625,
    ),
    "agieval_aqua_rat": _agieval_suite_spec(
        "agieval_aqua_rat",
        subset="aqua-rat",
        baseline={
            "acc,ll": 0.296875,
            "acc,ll_avg": 0.296875,
        },
    ),
    "agieval_logiqa_en": _agieval_suite_spec(
        "agieval_logiqa_en",
        subset="logiqa-en",
        baseline={
            "acc,ll": 0.3203125,
            "acc,ll_avg": 0.3203125,
        },
    ),
    "agieval_sat_math": _agieval_suite_spec(
        "agieval_sat_math",
        subset="sat-math",
        baseline={
            "acc,ll": 0.3046875,
            "acc,ll_avg": 0.3046875,
        },
    ),
    "agieval_gaokao_english": _agieval_suite_spec(
        "agieval_gaokao_english",
        subset="gaokao-english",
        baseline={
            "acc,ll": 0.609375,
            "acc,ll_avg": 0.609375,
        },
    ),
    "aime": _aime_suite_spec(
        "aime",
        dataset_path="gneubig/aime-1983-2024",
        split="train",
        baseline=0.06666666666666667,
    ),
    "aime24": _aime_suite_spec(
        "aime24",
        dataset_path="Maxwell-Jia/AIME_2024",
        split="train",
        baseline=0.03333333333333333,
    ),
    "aime25": _aime_suite_spec(
        "aime25",
        dataset_path="math-ai/aime25",
        split="test",
        baseline=0.0,
    ),
    "aime26": _aime_suite_spec(
        "aime26",
        dataset_path="math-ai/aime26",
        split="test",
        baseline=0.0,
    ),
    "cmmlu_agronomy": _cmmlu_suite_spec(
        "cmmlu_agronomy",
        subset="agronomy",
        baseline={
            "acc,ll": 0.28125,
            "acc,ll_avg": 0.28125,
        },
    ),
    "kmmlu_accounting": _kmmlu_suite_spec(
        "kmmlu_accounting",
        subset="accounting",
        dataset_name="Accounting",
        baseline={
            "acc,ll": 0.21875,
            "acc,ll_avg": 0.21875,
        },
    ),
    "mgsm_direct_en": _mgsm_suite_spec(
        "mgsm_direct_en",
        language="en",
        baseline=0.0625,
    ),
    "mgsm_direct_es_spanish_bench": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mgsm_direct_es_spanish_bench(
            batch_size=24,
            max_new_tokens=96,
            stream=True,
            max_rows=32,
        ),
        expected_name="mgsm_direct_es_spanish_bench",
        baseline={"acc,num": 0.0625},
        expected_metrics=frozenset({"acc,num"}),
        expected_metadata={
            "variant": "base",
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
            "stream": True,
            "generation_submission_mode": "continuous_refill",
            "num_fewshot": 0,
            "dataset_path": "juletxara/mgsm",
            "dataset_name": "es",
            "split": "test",
            "language": "es",
            "scoring_mode": "numeric_format_insensitive",
            "primary_metric": "acc,num",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_afrimgsm_sample(
            sample,
            index,
            language="es",
        ),
        result_validator=_validate_gsm8k_like_result,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "mmlu_cf_biology": _mmlu_cf_suite_spec(
        "mmlu_cf_biology",
        subject="biology",
        baseline={
            "acc,ll": 0.53125,
            "acc,ll_avg": 0.53125,
        },
    ),
    "hendrycks_math_algebra": _hendrycks_math_suite_spec(
        "hendrycks_math_algebra",
        subset="algebra",
        baseline=0.21875,
    ),
    "asdiv": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.asdiv(batch_size=24, stream=True, max_rows=128),
        expected_name="asdiv",
        baseline={
            "acc,ll": 0.0625,
        },
        expected_metrics=frozenset({"acc,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/asdiv",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            prompt_substrings=("\nQuestion:", "\nAnswer:"),
            metadata_validator=_metadata_fields_truthy(
                "body",
                "question",
                "answer",
                "solution_type",
                "formula",
            ),
            expected_scores=frozenset({"acc,ll"}),
            require_leading_space_target=False,
        ),
    ),
    "assin_entailment": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.assin_entailment(
            batch_size=24,
            stream=True,
            max_rows=32,
        ),
        expected_name="assin_entailment",
        baseline={
            "acc,ll": 0.3125,
            "acc,ll_avg": 0.34375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "nilc-nlp/assin",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_assin_sample(
            sample,
            index,
            variant="assin_entailment",
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "assin_paraphrase": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.assin_paraphrase(
            batch_size=24,
            stream=True,
            max_rows=32,
        ),
        expected_name="assin_paraphrase",
        baseline={
            "acc,ll": 0.40625,
            "acc,ll_avg": 0.40625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "nilc-nlp/assin",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_assin_sample(
            sample,
            index,
            variant="assin_paraphrase",
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "asdiv_cot_llama": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.asdiv_cot_llama(
            apply_chat_template=True,
            batch_size=24,
            max_new_tokens=96,
            stream=True,
            max_rows=128,
        ),
        expected_name="asdiv_cot_llama",
        baseline={
            "acc,num": 0.8984375,
        },
        expected_metrics=frozenset({"acc,num"}),
        expected_metadata={
            "variant": "cot_llama",
            "apply_chat_template": True,
            "fewshot_as_multiturn": True,
            "stream": True,
            "generation_submission_mode": "continuous_refill",
            "num_fewshot": 8,
            "dataset_path": "EleutherAI/asdiv",
            "scoring_mode": "numeric_format_insensitive",
            "primary_metric": "acc,num",
        },
        expected_sample_count=128,
        sample_validator=_assert_asdiv_cot_llama_sample,
        result_validator=_validate_gsm8k_like_result,
    ),
    "aexams_biology": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.aexams_biology(batch_size=24),
        expected_name="aexams_biology",
        baseline={"acc,ll": 0.34285714285714286, "acc,ll_avg": 0.34285714285714286},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "Hennara/aexams",
            "dataset_name": "Biology",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=35,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_suffix="\nالجواب:",
            metadata_validator=_metadata_field_in("subject", {"biology"}),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_35,
    ),
    "aexams_islamic_studies": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.aexams_islamic_studies(batch_size=24),
        expected_name="aexams_islamic_studies",
        baseline={"acc,ll": 0.4246575342465753, "acc,ll_avg": 0.4246575342465753},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "Hennara/aexams",
            "dataset_name": "IslamicStudies",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=73,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_suffix="\nالجواب:",
            metadata_validator=_metadata_field_in("subject", {"islamic_studies"}),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_73,
    ),
    "aexams_physics": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.aexams_physics(batch_size=24),
        expected_name="aexams_physics",
        baseline={"acc,ll": 0.2857142857142857, "acc,ll_avg": 0.2857142857142857},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "Hennara/aexams",
            "dataset_name": "Physics",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=42,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_suffix="\nالجواب:",
            metadata_validator=_metadata_field_in("subject", {"physics"}),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_42,
    ),
    "aexams_science": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.aexams_science(batch_size=24),
        expected_name="aexams_science",
        baseline={"acc,ll": 0.34782608695652173, "acc,ll_avg": 0.34782608695652173},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "Hennara/aexams",
            "dataset_name": "Science",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=115,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_suffix="\nالجواب:",
            metadata_validator=_metadata_field_in("subject", {"science"}),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_115,
    ),
    "aexams_social": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.aexams_social(batch_size=24),
        expected_name="aexams_social",
        baseline={"acc,ll": 0.29044117647058826, "acc,ll_avg": 0.29044117647058826},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "Hennara/aexams",
            "dataset_name": "Social",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=272,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_suffix="\nالجواب:",
            metadata_validator=_metadata_field_in("subject", {"social"}),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_272,
    ),
    "babi": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.babi(
            batch_size=24,
            max_new_tokens=16,
            stream=True,
            max_rows=128,
        ),
        expected_name="babi",
        baseline={
            "em": 0.0,
        },
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "Muennighoff/babi",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_generated_exact_match_sample(
            sample,
            index,
            prompt_prefix="Passage: ",
            prompt_suffix="\nAnswer:",
            prompt_substrings=("Question: ",),
            metadata_validator=_metadata_field_truthy("task"),
        ),
    ),
    "bear": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.bear(
            batch_size=8,
            stream=True,
            max_rows=32,
        ),
        expected_name="bear",
        baseline={
            "acc,ll": 0.40625,
            "acc,ll_avg": 0.28125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "lm-pub-quiz/BEAR",
            "dataset_name": "BEAR",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "empty_context_full_statement",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_bear_sample(
            sample,
            index,
            variant="bear",
            min_choice_count=50,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "bear_big": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.bear_big(
            batch_size=8,
            stream=True,
            max_rows=32,
        ),
        expected_name="bear_big",
        baseline={
            "acc,ll": 0.34375,
            "acc,ll_avg": 0.03125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "lm-pub-quiz/BEAR",
            "dataset_name": "BEAR_big",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "empty_context_full_statement",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_bear_sample(
            sample,
            index,
            variant="bear_big",
            min_choice_count=150,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "gsm8k": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.gsm8k(
            variant="cot",
            apply_chat_template=True,
            batch_size=24,
            max_new_tokens=96,
            stream=True,
            max_rows=128,
        ),
        expected_name="gsm8k_cot",
        baseline={
            "acc,num": 0.4296875,
        },
        expected_metrics=frozenset({"acc,num"}),
        expected_metadata={
            "variant": "cot",
            "apply_chat_template": True,
            "fewshot_as_multiturn": True,
            "stream": True,
            "generation_submission_mode": "continuous_refill",
            "num_fewshot": 8,
            "dataset_path": "openai/gsm8k",
            "scoring_mode": "numeric_format_insensitive",
            "primary_metric": "acc,num",
        },
        expected_sample_count=128,
        sample_validator=_assert_gsm8k_sample,
        result_validator=_validate_gsm8k_like_result,
        abs_tolerance=4 / 128,
    ),
    "gsm8k_fr": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.gsm8k_fr(
            batch_size=24,
            max_new_tokens=96,
            stream=True,
            max_rows=128,
        ),
        expected_name="gsm8k_fr",
        baseline={
            "acc,num": 0.109375,
        },
        expected_metrics=frozenset({"acc,num"}),
        expected_metadata={
            "variant": "base",
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
            "stream": True,
            "generation_submission_mode": "continuous_refill",
            "num_fewshot": 5,
            "dataset_path": "cmh/gsm8k_fr",
            "scoring_mode": "numeric_format_insensitive",
            "primary_metric": "acc,num",
        },
        expected_sample_count=128,
        sample_validator=_assert_gsm8k_translated_sample,
        result_validator=_validate_gsm8k_like_result,
    ),
    "gsm8k_ko": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.gsm8k_ko(
            batch_size=24,
            max_new_tokens=96,
            stream=True,
            max_rows=128,
        ),
        expected_name="gsm8k_ko",
        baseline={
            "acc,num": 0.109375,
        },
        expected_metrics=frozenset({"acc,num"}),
        expected_metadata={
            "variant": "base",
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
            "stream": True,
            "generation_submission_mode": "continuous_refill",
            "num_fewshot": 5,
            "dataset_path": "kuotient/gsm8k-ko",
            "scoring_mode": "numeric_format_insensitive",
            "primary_metric": "acc,num",
        },
        expected_sample_count=128,
        sample_validator=_assert_gsm8k_translated_sample,
        result_validator=_validate_gsm8k_like_result,
    ),
    "gsm8k_platinum": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.gsm8k_platinum(
            variant="cot",
            apply_chat_template=True,
            batch_size=24,
            max_new_tokens=96,
            stream=True,
            max_rows=128,
        ),
        expected_name="gsm8k_platinum_cot",
        baseline={
            "acc,num": 0.4296875,
        },
        expected_metrics=frozenset({"acc,num"}),
        expected_metadata={
            "variant": "cot",
            "apply_chat_template": True,
            "fewshot_as_multiturn": True,
            "stream": True,
            "generation_submission_mode": "continuous_refill",
            "num_fewshot": 8,
            "scoring_mode": "numeric_format_insensitive",
            "primary_metric": "acc,num",
        },
        expected_sample_count=128,
        sample_validator=_assert_gsm8k_sample,
        result_validator=_validate_gsm8k_like_result,
        abs_tolerance=3 / 128,
    ),
    "anli_r1": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.anli_r1(batch_size=24, stream=True, max_rows=128),
        expected_name="anli_r1",
        baseline={
            "acc,ll": 0.328125,
            "acc,ll_avg": 0.328125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "facebook/anli",
            "dataset_name": None,
            "split": "test_r1",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"True", "Neither", "False"},
            prediction_values={"True", "Neither", "False"},
            prompt_substrings=("Question: ", " True, False, or Neither?\nAnswer:"),
            metadata_validator=_metadata_has_choice_labels(exact_count=3),
        ),
    ),
    "anli_r2": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.anli_r2(batch_size=24, stream=True, max_rows=128),
        expected_name="anli_r2",
        baseline={
            "acc,ll": 0.375,
            "acc,ll_avg": 0.375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "facebook/anli",
            "dataset_name": None,
            "split": "test_r2",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"True", "Neither", "False"},
            prediction_values={"True", "Neither", "False"},
            prompt_substrings=("Question: ", " True, False, or Neither?\nAnswer:"),
            metadata_validator=_metadata_has_choice_labels(exact_count=3),
        ),
    ),
    "anli_r3": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.anli_r3(batch_size=24, stream=True, max_rows=128),
        expected_name="anli_r3",
        baseline={
            "acc,ll": 0.3671875,
            "acc,ll_avg": 0.3671875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "facebook/anli",
            "dataset_name": None,
            "split": "test_r3",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"True", "Neither", "False"},
            prediction_values={"True", "Neither", "False"},
            prompt_substrings=("Question: ", " True, False, or Neither?\nAnswer:"),
            metadata_validator=_metadata_has_choice_labels(exact_count=3),
        ),
    ),
    "boolq": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.boolq(batch_size=24, stream=True, max_rows=128),
        expected_name="boolq",
        baseline={
            "acc,ll": 0.734375,
            "acc,ll_avg": 0.734375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
        suite_factory=lambda: evalution.benchmarks.cb(batch_size=24, stream=True, max_rows=56),
        expected_name="cb",
        baseline={
            "acc,ll": 0.6428571428571429,
            "acc,ll_avg": 0.6428571428571429,
            "f1,ll_macro": 0.44734299516908216,
            "f1,ll_avg_macro": 0.44734299516908216,
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
            "stream": True,
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
        suite_factory=lambda: evalution.benchmarks.cola(batch_size=24, stream=True, max_rows=128),
        expected_name="cola",
        baseline={
            "acc,ll": 0.65625,
            "acc,ll_avg": 0.65625,
            "mcc,ll": 0.11075904894206384,
            "mcc,ll_avg": 0.11075904894206384,
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
            "stream": True,
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
    "cnn_dailymail": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.cnn_dailymail(
            batch_size=8,
            max_new_tokens=96,
            max_rows=32,
        ),
        expected_name="cnn_dailymail",
        baseline={
            "rouge1": 0.3200104088411268,
            "rouge2": 0.11965895439917622,
            "rougeLsum": 0.2568289407320166,
        },
        expected_metrics=frozenset({"rouge1", "rouge2", "rougeLsum"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "cnn_dailymail",
            "dataset_name": "3.0.0",
            "split": "validation",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_summary_rouge",
            "primary_metric": "rougeLsum",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_generated_summary_sample(
            sample,
            index,
            prompt_prefix="Summarize the following news article.\n\nArticle:\n",
            prompt_suffix="\n\nSummary:",
            metadata_validator=_assert_cnn_dailymail_metadata,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "cocoteros_es": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.cocoteros_es(
            batch_size=8,
            max_new_tokens=40,
            max_rows=32,
        ),
        expected_name="cocoteros_es",
        baseline={
            "bleu": 0.4089394499961097,
            "rouge1": 0.051929660753190166,
        },
        expected_metrics=frozenset({"bleu", "rouge1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "gplsi/cocoteros",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_corpus_bleu_mean_rouge1",
            "primary_metric": "bleu",
        },
        expected_sample_count=32,
        sample_validator=_assert_cocoteros_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "escola": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.escola(
            batch_size=24,
            max_rows=32,
        ),
        expected_name="escola",
        baseline={
            "acc,ll": 0.375,
            "acc,ll_avg": 0.375,
            "mcc,ll": 0.09245003270420485,
            "mcc,ll_avg": 0.09245003270420485,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg", "mcc,ll", "mcc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "nbel/EsCoLA",
            "dataset_name": None,
            "split": "validation",
            "order": "native",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"no", "sí"},
            prediction_values={"no", "sí"},
            prompt_suffix="Respuesta:",
            prompt_substrings=("\nPregunta: ¿Tiene sentido esta frase?\n",),
            metadata_validator=_metadata_has_escola_fields,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "code2text_go": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.code2text_go(batch_size=1, max_rows=16),
        expected_name="code2text_go",
        baseline={"bleu4": 0.12701024051547405},
        expected_metrics=frozenset({"bleu4"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "CM/codexglue_code2text_go",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_docstring_corpus_bleu4",
            "primary_metric": "bleu4",
            "language": "go",
            "num_beams": 10,
        },
        expected_sample_count=16,
        sample_validator=lambda sample, index: _assert_code2text_sample(sample, index, language="go"),
        abs_tolerance=0.05,
    ),
    "code2text_java": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.code2text_java(batch_size=1, max_rows=16),
        expected_name="code2text_java",
        baseline={"bleu4": 0.7372980967711409},
        expected_metrics=frozenset({"bleu4"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "CM/codexglue_code2text_java",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_docstring_corpus_bleu4",
            "primary_metric": "bleu4",
            "language": "java",
            "num_beams": 10,
        },
        expected_sample_count=16,
        sample_validator=lambda sample, index: _assert_code2text_sample(sample, index, language="java"),
        abs_tolerance=0.05,
    ),
    "code2text_javascript": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.code2text_javascript(batch_size=1, max_rows=16),
        expected_name="code2text_javascript",
        baseline={"bleu4": 0.24753010988524},
        expected_metrics=frozenset({"bleu4"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "CM/codexglue_code2text_javascript",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_docstring_corpus_bleu4",
            "primary_metric": "bleu4",
            "language": "javascript",
            "num_beams": 10,
        },
        expected_sample_count=16,
        sample_validator=lambda sample, index: _assert_code2text_sample(sample, index, language="javascript"),
        abs_tolerance=0.05,
    ),
    "code2text_php": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.code2text_php(batch_size=1, max_rows=16),
        expected_name="code2text_php",
        baseline={"bleu4": 0.2931185120841358},
        expected_metrics=frozenset({"bleu4"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "CM/codexglue_code2text_php",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_docstring_corpus_bleu4",
            "primary_metric": "bleu4",
            "language": "php",
            "num_beams": 10,
        },
        expected_sample_count=16,
        sample_validator=lambda sample, index: _assert_code2text_sample(sample, index, language="php"),
        abs_tolerance=0.05,
    ),
    "code2text_python": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.code2text_python(batch_size=1, max_rows=16),
        expected_name="code2text_python",
        baseline={"bleu4": 0.14476427430734137},
        expected_metrics=frozenset({"bleu4"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "CM/codexglue_code2text_python",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_docstring_corpus_bleu4",
            "primary_metric": "bleu4",
            "language": "python",
            "num_beams": 10,
        },
        expected_sample_count=16,
        sample_validator=lambda sample, index: _assert_code2text_sample(sample, index, language="python"),
        abs_tolerance=0.05,
    ),
    "code2text_ruby": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.code2text_ruby(batch_size=1, max_rows=16),
        expected_name="code2text_ruby",
        baseline={"bleu4": 0.3812030669921292},
        expected_metrics=frozenset({"bleu4"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "CM/codexglue_code2text_ruby",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_docstring_corpus_bleu4",
            "primary_metric": "bleu4",
            "language": "ruby",
            "num_beams": 10,
        },
        expected_sample_count=16,
        sample_validator=lambda sample, index: _assert_code2text_sample(sample, index, language="ruby"),
        abs_tolerance=0.05,
    ),
    "gpqa_main": _gpqa_suite_spec(
        subset="main",
        baseline={"em,choice_label": 0.0},
    ),
    "gpqa_diamond": _gpqa_suite_spec(
        subset="diamond",
        baseline={"em,choice_label": 0.0},
    ),
    "gpqa_extended": _gpqa_suite_spec(
        subset="extended",
        baseline={"em,choice_label": 0.0},
    ),
    "commonsense_qa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.commonsense_qa(
            batch_size=24,
            stream=True,
            max_rows=128,
        ),
        expected_name="commonsense_qa",
        baseline={
            "acc,ll": 0.6328125,
            "acc,ll_avg": 0.6328125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
    "darijahellaswag": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.darijahellaswag(
            batch_size=24,
            stream=True,
            max_rows=32,
        ),
        expected_name="darijahellaswag",
        baseline={
            "acc,ll": 0.28125,
            "acc,ll_avg": 0.28125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "MBZUAI-Paris/DarijaHellaSwag",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_substrings=(":",),
            metadata_validator=_metadata_fields_truthy("activity_label", "source_id", "split_type"),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "egyhellaswag": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.egyhellaswag(
            batch_size=24,
            stream=True,
            max_rows=32,
        ),
        expected_name="egyhellaswag",
        baseline={
            "acc,ll": 0.25,
            "acc,ll_avg": 0.1875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "UBC-NLP/EgyHellaSwag",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_substrings=(":",),
            metadata_validator=_metadata_fields_truthy("activity_label", "source_id", "split_type"),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "copa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.copa(batch_size=24, stream=True, max_rows=100),
        expected_name="copa",
        baseline={
            "acc,ll": 0.79,
            "acc,ll_avg": 0.7,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
    "copa_ar": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.copa_ar(batch_size=24, stream=True, max_rows=89),
        expected_name="copa_ar",
        baseline={
            "acc,ll": 0.5730337078651685,
            "acc,ll_avg": 0.4943820224719101,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "Hennara/copa_ar",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=89,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="السؤال: ",
            prompt_suffix="\nالجواب:",
            metadata_validator=_metadata_field_in("source_benchmark", {"copa"}),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_89,
    ),
    "copa_es": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.copa_es(
            batch_size=24,
            max_rows=32,
        ),
        expected_name="copa_es",
        baseline={
            "acc,ll": 0.6875,
            "acc,ll_avg": 0.71875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "BSC-LT/COPA-es",
            "dataset_name": None,
            "split": "test",
            "order": "native",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=_assert_copa_es_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "copal_id_standard": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.copal_id_standard(
            batch_size=24,
            stream=True,
            max_rows=128,
        ),
        expected_name="copal_id_standard",
        baseline={
            "acc,ll": 0.5078125,
            "acc,ll_avg": 0.546875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "haryoaw/COPAL",
            "dataset_name": "id",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_question_and_variant("standard"),
        ),
    ),
    "copal_id_colloquial": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.copal_id_colloquial(
            batch_size=24,
            stream=True,
            max_rows=128,
        ),
        expected_name="copal_id_colloquial",
        baseline={
            "acc,ll": 0.4140625,
            "acc,ll_avg": 0.4140625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "haryoaw/COPAL",
            "dataset_name": "id",
            "split": "test_colloquial",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_question_and_variant("colloquial"),
        ),
    ),
    "ethics_cm": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.ethics_cm(batch_size=24, stream=True, max_rows=128),
        expected_name="ethics_cm",
        baseline={
            "acc,ll": 0.53125,
            "acc,ll_avg": 0.53125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/hendrycks_ethics",
            "dataset_name": "commonsense",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"no", "yes"},
            prediction_values={"no", "yes"},
            prompt_suffix="\nAnswer:",
            prompt_substrings=("\nQuestion: Is this wrong?\nAnswer:",),
            metadata_validator=_metadata_has_choice_labels(exact_count=2),
        ),
    ),
    "ethics_deontology": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.ethics_deontology(batch_size=24, stream=True, max_rows=128),
        expected_name="ethics_deontology",
        baseline={
            "acc,ll": 0.546875,
            "acc,ll_avg": 0.546875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/hendrycks_ethics",
            "dataset_name": "deontology",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"unreasonable", "reasonable"},
            prediction_values={"unreasonable", "reasonable"},
            prompt_prefix='Question: Would most people believe this reasonable or unreasonable to say? "',
            prompt_suffix='"\nAnswer:',
            metadata_validator=_metadata_has_choice_labels(exact_count=2),
        ),
    ),
    "ethics_justice": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.ethics_justice(batch_size=24, stream=True, max_rows=128),
        expected_name="ethics_justice",
        baseline={
            "acc,ll": 0.4921875,
            "acc,ll_avg": 0.4921875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/hendrycks_ethics",
            "dataset_name": "justice",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"unreasonable", "reasonable"},
            prediction_values={"unreasonable", "reasonable"},
            prompt_prefix='Question: Would most people believe this reasonable or unreasonable to say? "',
            prompt_suffix='"\nAnswer:',
            metadata_validator=_metadata_has_choice_labels(exact_count=2),
        ),
    ),
    "ethics_utilitarianism": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.ethics_utilitarianism(batch_size=24, stream=True, max_rows=128),
        expected_name="ethics_utilitarianism",
        baseline={
            "acc,ll": 0.4453125,
            "acc,ll_avg": 0.4453125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/hendrycks_ethics",
            "dataset_name": "utilitarianism",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"no", "yes"},
            prediction_values={"no", "yes"},
            prompt_prefix="Scenario 1: ",
            prompt_suffix="\nAnswer:",
            prompt_substrings=("\nScenario 2: ", "\nQuestion: Is Scenario 1 preferable?\nAnswer:"),
            metadata_validator=_metadata_has_choice_labels(exact_count=2),
        ),
    ),
    "ethics_virtue": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.ethics_virtue(batch_size=24, stream=True, max_rows=128),
        expected_name="ethics_virtue",
        baseline={
            "acc,ll": 0.3359375,
            "acc,ll_avg": 0.3359375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/hendrycks_ethics",
            "dataset_name": "virtue",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"no", "yes"},
            prediction_values={"no", "yes"},
            prompt_prefix="Sentence: ",
            prompt_suffix="\nAnswer:",
            prompt_substrings=('Question: Does the character in this sentence exhibit the trait "',),
            metadata_validator=_metadata_has_choice_labels(exact_count=2),
        ),
    ),
    "arc_easy": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.arc_easy(batch_size=24, stream=True, max_rows=128),
        expected_name="arc_easy",
        baseline={
            "acc,exam": 0.64453125,
        },
        expected_metrics=frozenset({"acc,exam"}),
        expected_metadata={
            "stream": True,
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
            stream=True,
            max_rows=128,
        ),
        expected_name="arc_challenge",
        baseline={
            "acc,exam": 0.3671875,
        },
        expected_metrics=frozenset({"acc,exam"}),
        expected_metadata={
            "stream": True,
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
            stream=True,
            max_rows=128,
            label_permutations=0.25,
        ),
        expected_name="arc_challenge",
        baseline={
            "acc,exam": 0.3671875,
            "acc,label_perm:0.25": 0.5234375,
        },
        expected_metrics=frozenset(
            {
                "acc,exam",
                "acc,label_perm:0.25",
            }
        ),
        expected_metadata={
            "stream": True,
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
        suite_factory=lambda: evalution.benchmarks.hellaswag(batch_size=24, stream=True, max_rows=128),
        expected_name="hellaswag",
        baseline={
            "acc,ll": 0.4375,
            "acc,ll_avg": 0.5390625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
    "histoires_morales": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.histoires_morales(batch_size=24, max_rows=32),
        expected_name="histoires_morales",
        baseline={
            "acc,ll": 0.53125,
            "acc,ll_avg": 0.5,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "LabHC/histoires_morales",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy(
                "guid",
                "norm",
                "situation",
                "intention",
                "moral_action",
                "immoral_action",
                "moral_consequence",
                "immoral_consequence",
            ),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "moral_stories": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.moral_stories(batch_size=24, max_rows=32),
        expected_name="moral_stories",
        baseline={
            "acc,ll": 0.59375,
            "acc,ll_avg": 0.59375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "LabHC/moral_stories",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_has_moral_stories_fields,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "mbpp": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mbpp(batch_size=4, max_rows=64, max_new_tokens=512),
        expected_name="mbpp",
        baseline={
            "pass@1": 0.0,
        },
        expected_metrics=frozenset({"pass@1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "mbpp",
            "dataset_name": "sanitized",
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_code_execution",
            "primary_metric": "pass@1",
        },
        expected_sample_count=64,
        sample_validator=_assert_mbpp_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "humaneval": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.humaneval(batch_size=4, max_rows=32, max_new_tokens=512),
        expected_name="humaneval",
        baseline={
            "pass@1": 0.0,
        },
        expected_metrics=frozenset({"pass@1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "openai/openai_humaneval",
            "dataset_name": "openai_humaneval",
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_code_execution",
            "primary_metric": "pass@1",
        },
        expected_sample_count=32,
        sample_validator=_assert_humaneval_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "ifeval": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.ifeval(batch_size=8, max_rows=64),
        expected_name="ifeval",
        baseline={
            "prompt_level_strict_acc": 0.25,
            "prompt_level_loose_acc": 0.328125,
            "inst_level_strict_acc": 0.42424242424242425,
            "inst_level_loose_acc": 0.48484848484848486,
        },
        expected_metrics=frozenset({
            "prompt_level_strict_acc",
            "prompt_level_loose_acc",
            "inst_level_strict_acc",
            "inst_level_loose_acc",
        }),
        expected_metadata={
            "stream": False,
            "dataset_path": "google/IFEval",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "instruction_following",
            "primary_metric": "prompt_level_strict_acc",
        },
        expected_sample_count=64,
        sample_validator=_assert_ifeval_sample,
        abs_tolerance=2 / 64,
    ),
    "ifeval_pt": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.ifeval_pt(batch_size=8, max_rows=64),
        expected_name="ifeval_pt",
        baseline={
            "prompt_level_strict_acc": 0.203125,
            "prompt_level_loose_acc": 0.25,
            "inst_level_strict_acc": 0.38,
            "inst_level_loose_acc": 0.41,
        },
        expected_metrics=frozenset({
            "prompt_level_strict_acc",
            "prompt_level_loose_acc",
            "inst_level_strict_acc",
            "inst_level_loose_acc",
        }),
        expected_metadata={
            "stream": False,
            "dataset_path": "Polygl0t/IFEval-PT",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "instruction_following",
            "primary_metric": "prompt_level_strict_acc",
        },
        expected_sample_count=64,
        sample_validator=_assert_ifeval_sample,
        abs_tolerance=2 / 64,
    ),
    "multirc": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.multirc(batch_size=8, max_rows=16),
        expected_name="multirc",
        baseline={
            "em": 0.0,
            "f1a": 0.0,
        },
        expected_metrics=frozenset({"em", "f1a"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "super_glue",
            "dataset_name": "multirc",
            "split": "validation",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "multi_label_extraction",
            "primary_metric": "em",
        },
        expected_sample_count=16,
        sample_validator=lambda sample, index: (
            _assert_dict_subset(
                sample["metadata"],
                {"paragraph_idx": int, "question_idx": int, "num_answers": int},
            )
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "icelandic_winogrande": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.icelandic_winogrande(batch_size=24, max_rows=32),
        expected_name="icelandic_winogrande",
        baseline={
            "acc,ll": 0.5625,
            "acc,ll_avg": 0.5625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "mideind/icelandic-winogrande",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "partial_evaluation_blank_replacement",
        },
        expected_sample_count=32,
        sample_validator=_assert_icelandic_winogrande_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "hellaswag_label_perm_0_25": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.hellaswag(
            batch_size=24,
            stream=True,
            max_rows=128,
            label_permutations=0.25,
        ),
        expected_name="hellaswag",
        baseline={
            "acc,ll": 0.453125,
            "acc,ll_avg": 0.5390625,
            "acc,label_perm:0.25": 0.421875,
        },
        expected_metrics=frozenset(
            {
                "acc,ll",
                "acc,ll_avg",
                "acc,label_perm:0.25",
            }
        ),
        expected_metadata={
            "stream": True,
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
    "kobest_boolq": _kobest_suite_spec(
        subset="boolq",
        baseline={
            "acc,ll": 0.546875,
            "acc,ll_avg": 0.5546875,
        },
    ),
    "kobest_copa": _kobest_suite_spec(
        subset="copa",
        baseline={
            "acc,ll": 0.546875,
            "acc,ll_avg": 0.53125,
        },
    ),
    "kobest_hellaswag": _kobest_suite_spec(
        subset="hellaswag",
        baseline={
            "acc,ll": 0.359375,
            "acc,ll_avg": 0.4609375,
        },
    ),
    "kobest_sentineg": _kobest_suite_spec(
        subset="sentineg",
        baseline={
            "acc,ll": 0.5078125,
            "acc,ll_avg": 0.515625,
        },
    ),
    "kobest_wic": _kobest_suite_spec(
        subset="wic",
        baseline={
            "acc,ll": 0.4375,
            "acc,ll_avg": 0.5,
        },
    ),
    "headqa_en": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.headqa_en(batch_size=24, stream=True, max_rows=128),
        expected_name="headqa_en",
        baseline={
            "acc,ll": 0.40625,
            "acc,ll_avg": 0.4140625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/headqa",
            "dataset_name": "en",
            "split": "test",
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
    "headqa_es": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.headqa_es(batch_size=24, stream=True, max_rows=128),
        expected_name="headqa_es",
        baseline={
            "acc,ll": 0.2578125,
            "acc,ll_avg": 0.328125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/headqa",
            "dataset_name": "es",
            "split": "test",
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
    "lambada_openai": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai",
        baseline={
            "acc,ll": 0.59375,
            "ppl,ll": 6.862626502040115,
        },
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_openai",
            "dataset_name": "default",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_de": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_de(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_de",
        baseline={"acc,ll": 0.25, "ppl,ll": 158.80795113445453},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_openai",
            "dataset_name": "de",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_en": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_en(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_en",
        baseline={"acc,ll": 0.59375, "ppl,ll": 6.862626502040115},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_openai",
            "dataset_name": "en",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_es": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_es(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_es",
        baseline={"acc,ll": 0.2109375, "ppl,ll": 206.32852515616028},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_openai",
            "dataset_name": "es",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_fr": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_fr(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_fr",
        baseline={"acc,ll": 0.3359375, "ppl,ll": 117.63198253978209},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_openai",
            "dataset_name": "fr",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_it": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_it(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_it",
        baseline={"acc,ll": 0.34375, "ppl,ll": 86.66940887099368},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_openai",
            "dataset_name": "it",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_stablelm_de": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_stablelm_de(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_stablelm_de",
        baseline={"acc,ll": 0.4140625, "ppl,ll": 27.918869538919996},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_multilingual_stablelm",
            "dataset_name": "de",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_stablelm_en": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_stablelm_en(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_stablelm_en",
        baseline={"acc,ll": 0.59375, "ppl,ll": 6.862626502040115},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_multilingual_stablelm",
            "dataset_name": "en",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_stablelm_es": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_stablelm_es(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_stablelm_es",
        baseline={"acc,ll": 0.3515625, "ppl,ll": 241.99668470319156},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_multilingual_stablelm",
            "dataset_name": "es",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_stablelm_fr": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_stablelm_fr(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_stablelm_fr",
        baseline={"acc,ll": 0.359375, "ppl,ll": 49.21898066726227},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_multilingual_stablelm",
            "dataset_name": "fr",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_present("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_stablelm_it": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_stablelm_it(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_stablelm_it",
        baseline={"acc,ll": 0.3203125, "ppl,ll": 74.89661557691544},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_multilingual_stablelm",
            "dataset_name": "it",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_stablelm_nl": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_stablelm_nl(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_stablelm_nl",
        baseline={"acc,ll": 0.3359375, "ppl,ll": 347.65201100927254},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_multilingual_stablelm",
            "dataset_name": "nl",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_mt_stablelm_pt": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_mt_stablelm_pt(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_openai_mt_stablelm_pt",
        baseline={"acc,ll": 0.40625, "ppl,ll": 22.182617076504428},
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_multilingual_stablelm",
            "dataset_name": "pt",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_openai_cloze": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_openai_cloze(
            batch_size=24,
            stream=True,
            max_rows=128,
        ),
        expected_name="lambada_openai_cloze",
        baseline={
            "acc,ll": 0.015625,
            "ppl,ll": 3349.9450401193135,
        },
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/lambada_openai",
            "dataset_name": "default",
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
            "prompt_variant": "cloze",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            prompt_suffix=" ____. ->",
            metadata_validator=_metadata_fields_truthy("text", "target_token", "prompt_variant"),
        ),
    ),
    "lambada_standard": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_standard(batch_size=24, stream=True, max_rows=128),
        expected_name="lambada_standard",
        baseline={
            "acc,ll": 0.5078125,
            "ppl,ll": 9.846866118664849,
        },
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "cimec/lambada",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_fields_truthy("text", "target_token"),
        ),
    ),
    "lambada_standard_cloze": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.lambada_standard_cloze(
            batch_size=24,
            stream=True,
            max_rows=128,
        ),
        expected_name="lambada_standard_cloze",
        baseline={
            "acc,ll": 0.0078125,
            "ppl,ll": 6040.105500898565,
        },
        expected_metrics=frozenset({"acc,ll", "ppl,ll"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "cimec/lambada",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "single_continuation_loglikelihood",
            "prompt_variant": "cloze",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_single_continuation_loglikelihood_sample(
            sample,
            index,
            prompt_suffix=" ____. ->",
            metadata_validator=_metadata_fields_truthy("text", "target_token", "prompt_variant"),
        ),
    ),
    "logiqa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.logiqa(batch_size=24, max_rows=128),
        expected_name="logiqa",
        baseline={
            "acc,ll": 0.1875,
            "acc,ll_avg": 0.34375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EleutherAI/logiqa",
            "dataset_name": "logiqa",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Passage: ",
            prompt_substrings=("\nQuestion: ", "\nChoices:\n", "\nAnswer:"),
            metadata_validator=_metadata_has_choice_labels(exact_count=4),
        ),
    ),
    "logiqa2": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.logiqa2(batch_size=24, max_rows=128),
        expected_name="logiqa2",
        baseline={
            "acc,ll": 0.28125,
            "acc,ll_avg": 0.3359375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "datatune/LogiQA2.0",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Passage: ",
            prompt_substrings=("\nQuestion: ", "\nChoices:\n", "\nAnswer:"),
            metadata_validator=_metadata_has_choice_labels(exact_count=4),
        ),
    ),
    "longbench_trec": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.longbench_trec(batch_size=4, max_rows=32),
        expected_name="longbench_trec",
        baseline={
            "score": 0.59375,
            "classification_score": 0.59375,
        },
        expected_metrics=frozenset({"score", "classification_score"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "Xnhyacinth/LongBench",
            "dataset_name": "trec",
            "split": "test",
            "order": "native",
            "generation_submission_mode": "continuous_refill",
            "subset": "longbench_trec",
            "task_root": "trec",
            "variant": "base",
            "scoring_mode": "generated_longbench_classification",
            "primary_metric": "score",
            "metric_name": "classification_score",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_longbench_sample(
            sample,
            index,
            task_root="trec",
            metric_name="classification_score",
            language="en",
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "longbench2_legal_single": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.longbench2_legal_single(batch_size=1, max_rows=9),
        expected_name="longbench2_legal_single",
        baseline={
            "acc,ll": 0.5555555555555556,
            "acc,ll_avg": 0.5555555555555556,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "recursal/longbench-v2",
            "dataset_name": "legal_single",
            "split": "train",
            "order": "native",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=9,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="Please read the following text and answer the question below.\n\n<text>\n",
            prompt_substrings=(
                "\n</text>\n\nWhat is the correct answer to this question: ",
                "\nChoices:\n(A) ",
                "\n(B) ",
                "\n(C) ",
                "\n(D) ",
            ),
            prompt_suffix="\n\nAnswer:",
            metadata_validator=_metadata_fields_truthy("dataset_name", "domain", "difficulty", "length", "choice_texts"),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_9,
    ),
    "mathqa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mathqa(batch_size=24, max_rows=128),
        expected_name="mathqa",
        baseline={
            "acc,ll": 0.28125,
            "acc,ll_avg": 0.2890625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "math_qa",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Question: ",
            prompt_suffix="\nAnswer:",
            metadata_validator=_metadata_has_choice_labels(exact_count=5),
        ),
    ),
    "click": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click")(batch_size=24, max_rows=128),
        expected_name="click",
        baseline={"acc,ll": 0.25, "acc,ll_avg": 0.25},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click"),
        ),
    ),
    "click_lang": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_lang")(batch_size=4, max_rows=128),
        expected_name="click_lang",
        baseline={"acc,ll": 0.2265625, "acc,ll_avg": 0.2265625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_lang",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_lang"),
        ),
    ),
    "click_lang_text": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_lang_text")(batch_size=4, max_rows=128),
        expected_name="click_lang_text",
        baseline={"acc,ll": 0.25, "acc,ll_avg": 0.25},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_lang_text",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_lang_text"),
        ),
    ),
    "click_lang_grammar": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_lang_grammar")(batch_size=24, max_rows=128),
        expected_name="click_lang_grammar",
        baseline={"acc,ll": 0.2421875, "acc,ll_avg": 0.2421875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_lang_grammar",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_lang_grammar"),
        ),
    ),
    "click_lang_function": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_lang_function")(batch_size=24, max_rows=128),
        expected_name="click_lang_function",
        baseline={"acc,ll": 0.2421875, "acc,ll_avg": 0.2421875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_lang_function",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_lang_function"),
        ),
    ),
    "click_cul": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_cul")(batch_size=24, max_rows=128),
        expected_name="click_cul",
        baseline={"acc,ll": 0.25, "acc,ll_avg": 0.25},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_cul",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_cul"),
        ),
    ),
    "click_cul_economy": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_cul_economy")(batch_size=24, max_rows=128),
        expected_name="click_cul_economy",
        baseline={"acc,ll": 0.2542372881355932, "acc,ll_avg": 0.2542372881355932},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_cul_economy",
        },
        expected_sample_count=59,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_cul_economy"),
        ),
    ),
    "click_cul_geography": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_cul_geography")(batch_size=24, max_rows=128),
        expected_name="click_cul_geography",
        baseline={"acc,ll": 0.3125, "acc,ll_avg": 0.3125},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_cul_geography",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_cul_geography"),
        ),
    ),
    "click_cul_history": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_cul_history")(batch_size=24, max_rows=128),
        expected_name="click_cul_history",
        baseline={"acc,ll": 0.265625, "acc,ll_avg": 0.265625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_cul_history",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_cul_history"),
        ),
    ),
    "click_cul_kpop": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_cul_kpop")(batch_size=24, max_rows=128),
        expected_name="click_cul_kpop",
        baseline={"acc,ll": 0.34146341463414637, "acc,ll_avg": 0.34146341463414637},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_cul_kpop",
        },
        expected_sample_count=41,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_cul_kpop"),
        ),
    ),
    "click_cul_law": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_cul_law")(batch_size=24, max_rows=128),
        expected_name="click_cul_law",
        baseline={"acc,ll": 0.234375, "acc,ll_avg": 0.234375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_cul_law",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_cul_law"),
        ),
    ),
    "click_cul_politics": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_cul_politics")(batch_size=24, max_rows=128),
        expected_name="click_cul_politics",
        baseline={"acc,ll": 0.2261904761904762, "acc,ll_avg": 0.2261904761904762},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_cul_politics",
        },
        expected_sample_count=84,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_cul_politics"),
        ),
    ),
    "click_cul_society": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_cul_society")(batch_size=24, max_rows=128),
        expected_name="click_cul_society",
        baseline={"acc,ll": 0.2421875, "acc,ll_avg": 0.2421875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_cul_society",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_cul_society"),
        ),
    ),
    "click_cul_tradition": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "click_cul_tradition")(batch_size=24, max_rows=128),
        expected_name="click_cul_tradition",
        baseline={"acc,ll": 0.3984375, "acc,ll_avg": 0.3984375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EunsuKim/CLIcK",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "click_cul_tradition",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D", "E"},
            prediction_values={"A", "B", "C", "D", "E"},
            prompt_substrings=("질문: ", '\n보기:\n'),
            prompt_suffix='\n정답:',
            metadata_validator=_metadata_has_click_fields(subset="click_cul_tradition"),
        ),
    ),
    "haerae": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "haerae")(batch_size=24, max_rows=128),
        expected_name="haerae",
        baseline={"acc,ll": 0.3359375, "acc,ll_avg": 0.3359375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "HAERAE-HUB/HAE_RAE_BENCH",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "haerae",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prediction_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prompt_substrings=("### 질문:", "### 선택지:"),
            prompt_suffix="### 정답:",
            metadata_validator=_metadata_has_haerae_fields(subset="haerae"),
        ),
    ),
    "haerae_general_knowledge": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "haerae_general_knowledge")(batch_size=24, max_rows=128),
        expected_name="haerae_general_knowledge",
        baseline={"acc,ll": 0.296875, "acc,ll_avg": 0.296875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "HAERAE-HUB/HAE_RAE_BENCH",
            "dataset_name": "general_knowledge",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "general_knowledge",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prediction_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prompt_substrings=("### 질문:", "### 선택지:"),
            prompt_suffix="### 정답:",
            metadata_validator=_metadata_has_haerae_fields(subset="general_knowledge"),
        ),
    ),
    "haerae_history": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "haerae_history")(batch_size=24, max_rows=128),
        expected_name="haerae_history",
        baseline={"acc,ll": 0.2265625, "acc,ll_avg": 0.2265625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "HAERAE-HUB/HAE_RAE_BENCH",
            "dataset_name": "history",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "history",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prediction_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prompt_substrings=("### 질문:", "### 선택지:"),
            prompt_suffix="### 정답:",
            metadata_validator=_metadata_has_haerae_fields(subset="history"),
        ),
    ),
    "haerae_loan_word": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "haerae_loan_word")(batch_size=24, max_rows=128),
        expected_name="haerae_loan_word",
        baseline={"acc,ll": 0.3125, "acc,ll_avg": 0.3125},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "HAERAE-HUB/HAE_RAE_BENCH",
            "dataset_name": "loan_words",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "loan_word",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prediction_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prompt_substrings=("### 질문:", "### 선택지:"),
            prompt_suffix="### 정답:",
            metadata_validator=_metadata_has_haerae_fields(subset="loan_word"),
        ),
    ),
    "haerae_rare_word": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "haerae_rare_word")(batch_size=24, max_rows=128),
        expected_name="haerae_rare_word",
        baseline={"acc,ll": 0.3671875, "acc,ll_avg": 0.3671875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "HAERAE-HUB/HAE_RAE_BENCH",
            "dataset_name": "rare_words",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "rare_word",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prediction_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prompt_substrings=("### 질문:", "### 선택지:"),
            prompt_suffix="### 정답:",
            metadata_validator=_metadata_has_haerae_fields(subset="rare_word"),
        ),
    ),
    "haerae_standard_nomenclature": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "haerae_standard_nomenclature")(batch_size=24, max_rows=128),
        expected_name="haerae_standard_nomenclature",
        baseline={"acc,ll": 0.3046875, "acc,ll_avg": 0.3046875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "HAERAE-HUB/HAE_RAE_BENCH",
            "dataset_name": "standard_nomenclature",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subset": "standard_nomenclature",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prediction_values={"(A)", "(B)", "(C)", "(D)", "(E)"},
            prompt_substrings=("### 질문:", "### 선택지:"),
            prompt_suffix="### 정답:",
            metadata_validator=_metadata_has_haerae_fields(subset="standard_nomenclature"),
        ),
    ),
    "kormedmcqa": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "kormedmcqa")(batch_size=24, max_rows=128),
        expected_name="kormedmcqa",
        baseline={"em": 0.3203125},
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "sean0042/KorMedMCQA",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
            "subset": "kormedmcqa",
            "fewshot_split": "fewshot",
            "num_fewshot": 5,
            "apply_chat_template": False,
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_generated_exact_match_sample(
            sample,
            index,
            prompt_suffix="정답：",
            prompt_substrings=("\nA. ", "\nE. "),
            metadata_validator=_metadata_has_kormedmcqa_fields(
                allowed_subsets={"doctor", "nurse", "pharm", "dentist"}
            ),
            allow_empty_prediction=True,
        ),
    ),
    "kormedmcqa_doctor": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "kormedmcqa_doctor")(batch_size=24, max_rows=128),
        expected_name="kormedmcqa_doctor",
        baseline={"em": 0.1953125},
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "sean0042/KorMedMCQA",
            "dataset_name": "doctor",
            "split": "test",
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
            "subset": "doctor",
            "fewshot_split": "fewshot",
            "num_fewshot": 5,
            "apply_chat_template": False,
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_generated_exact_match_sample(
            sample,
            index,
            prompt_suffix="정답：",
            prompt_substrings=("\nA. ", "\nE. "),
            metadata_validator=_metadata_has_kormedmcqa_fields(subset="doctor"),
            allow_empty_prediction=True,
        ),
        abs_tolerance=3 / 128,
    ),
    "kormedmcqa_nurse": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "kormedmcqa_nurse")(batch_size=24, max_rows=128),
        expected_name="kormedmcqa_nurse",
        baseline={"em": 0.34375},
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "sean0042/KorMedMCQA",
            "dataset_name": "nurse",
            "split": "test",
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
            "subset": "nurse",
            "fewshot_split": "fewshot",
            "num_fewshot": 5,
            "apply_chat_template": False,
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_generated_exact_match_sample(
            sample,
            index,
            prompt_suffix="정답：",
            prompt_substrings=("\nA. ", "\nE. "),
            metadata_validator=_metadata_has_kormedmcqa_fields(subset="nurse"),
            allow_empty_prediction=True,
        ),
        abs_tolerance=3 / 128,
    ),
    "kormedmcqa_pharm": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "kormedmcqa_pharm")(batch_size=24, max_rows=128),
        expected_name="kormedmcqa_pharm",
        baseline={"em": 0.296875},
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "sean0042/KorMedMCQA",
            "dataset_name": "pharm",
            "split": "test",
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
            "subset": "pharm",
            "fewshot_split": "fewshot",
            "num_fewshot": 5,
            "apply_chat_template": False,
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_generated_exact_match_sample(
            sample,
            index,
            prompt_suffix="정답：",
            prompt_substrings=("\nA. ", "\nE. "),
            metadata_validator=_metadata_has_kormedmcqa_fields(subset="pharm"),
            allow_empty_prediction=True,
        ),
    ),
    "kormedmcqa_dentist": SuiteSpec(
        suite_factory=lambda: getattr(evalution.benchmarks, "kormedmcqa_dentist")(batch_size=24, max_rows=128),
        expected_name="kormedmcqa_dentist",
        baseline={"em": 0.3125},
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "sean0042/KorMedMCQA",
            "dataset_name": "dentist",
            "split": "test",
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
            "subset": "dentist",
            "fewshot_split": "fewshot",
            "num_fewshot": 5,
            "apply_chat_template": False,
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_generated_exact_match_sample(
            sample,
            index,
            prompt_suffix="정답：",
            prompt_substrings=("\nA. ", "\nE. "),
            metadata_validator=_metadata_has_kormedmcqa_fields(subset="dentist"),
            allow_empty_prediction=True,
        ),
        abs_tolerance=3 / 128,
    ),
    "fda": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.fda(batch_size=24, max_rows=128),
        expected_name="fda",
        baseline={"contains": 0.5703125},
        expected_metrics=frozenset({"contains"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "hazyresearch/based-fda",
            "dataset_name": "default",
            "split": "validation",
            "scoring_mode": "generated_contains_match",
            "primary_metric": "contains",
            "generation_submission_mode": "continuous_refill",
        },
        expected_sample_count=128,
        sample_validator=_assert_generated_contains_sample,
    ),
    "gsm_plus": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.gsm_plus(batch_size=24, max_rows=128),
        expected_name="gsm_plus",
        baseline={
            "em,strict": 0.171875,
            "em,flex": 0.171875,
        },
        expected_metrics=frozenset({"em,strict", "em,flex"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "qintongli/GSM-Plus",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_regex_extract_exact_match",
            "primary_metric": "em,strict",
            "variant": "base",
            "num_fewshot": 5,
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_generated_regex_extract_exact_match_sample(
            sample,
            index,
            prompt_prefix="Question: ",
            prompt_suffix="\nAnswer:",
            metadata_validator=_metadata_has_gsm_plus_fields,
        ),
    ),
    "gsm_plus_mini": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.gsm_plus_mini(batch_size=24, max_rows=128),
        expected_name="gsm_plus_mini",
        baseline={
            "em,strict": 0.328125,
            "em,flex": 0.328125,
        },
        expected_metrics=frozenset({"em,strict", "em,flex"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "qintongli/GSM-Plus",
            "dataset_name": None,
            "split": "testmini",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_regex_extract_exact_match",
            "primary_metric": "em,strict",
            "variant": "base",
            "num_fewshot": 5,
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_generated_regex_extract_exact_match_sample(
            sample,
            index,
            prompt_prefix="Question: ",
            prompt_suffix="\nAnswer:",
            metadata_validator=_metadata_has_gsm_plus_fields,
        ),
    ),
    "mastermind_24_easy": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mastermind_24_easy(batch_size=24, max_rows=128),
        expected_name="mastermind_24_easy",
        baseline={"acc,ll": 0.375, "acc,ll_avg": 0.375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "flair/mastermind_24_mcq_random",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "variant": "mastermind_24_easy",
            "code_shape": "24",
            "difficulty": "easy",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_suffix="\n\nThe secret code is:",
            metadata_validator=lambda metadata: _metadata_has_mastermind_fields(
                metadata,
                variant="mastermind_24_easy",
                code_shape="24",
                difficulty="easy",
            ),
        ),
    ),
    "mastermind_24_hard": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mastermind_24_hard(batch_size=24, max_rows=128),
        expected_name="mastermind_24_hard",
        baseline={"acc,ll": 0.4140625, "acc,ll_avg": 0.4140625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "flair/mastermind_24_mcq_close",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "variant": "mastermind_24_hard",
            "code_shape": "24",
            "difficulty": "hard",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_suffix="\n\nThe secret code is:",
            metadata_validator=lambda metadata: _metadata_has_mastermind_fields(
                metadata,
                variant="mastermind_24_hard",
                code_shape="24",
                difficulty="hard",
            ),
        ),
    ),
    "mastermind_35_easy": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mastermind_35_easy(batch_size=24, max_rows=128),
        expected_name="mastermind_35_easy",
        baseline={"acc,ll": 0.4296875, "acc,ll_avg": 0.4296875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "flair/mastermind_35_mcq_random",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "variant": "mastermind_35_easy",
            "code_shape": "35",
            "difficulty": "easy",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_suffix="\n\nThe secret code is:",
            metadata_validator=lambda metadata: _metadata_has_mastermind_fields(
                metadata,
                variant="mastermind_35_easy",
                code_shape="35",
                difficulty="easy",
            ),
        ),
    ),
    "mastermind_35_hard": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mastermind_35_hard(batch_size=24, max_rows=128),
        expected_name="mastermind_35_hard",
        baseline={"acc,ll": 0.4609375, "acc,ll_avg": 0.4609375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "flair/mastermind_35_mcq_close",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "variant": "mastermind_35_hard",
            "code_shape": "35",
            "difficulty": "hard",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_suffix="\n\nThe secret code is:",
            metadata_validator=lambda metadata: _metadata_has_mastermind_fields(
                metadata,
                variant="mastermind_35_hard",
                code_shape="35",
                difficulty="hard",
            ),
        ),
    ),
    "mastermind_46_easy": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mastermind_46_easy(batch_size=24, max_rows=128),
        expected_name="mastermind_46_easy",
        baseline={"acc,ll": 0.625, "acc,ll_avg": 0.625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "flair/mastermind_46_mcq_random",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "variant": "mastermind_46_easy",
            "code_shape": "46",
            "difficulty": "easy",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_suffix="\n\nThe secret code is:",
            metadata_validator=lambda metadata: _metadata_has_mastermind_fields(
                metadata,
                variant="mastermind_46_easy",
                code_shape="46",
                difficulty="easy",
            ),
        ),
    ),
    "mastermind_46_hard": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mastermind_46_hard(batch_size=24, max_rows=128),
        expected_name="mastermind_46_hard",
        baseline={"acc,ll": 0.6171875, "acc,ll_avg": 0.6171875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "flair/mastermind_46_mcq_close",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "variant": "mastermind_46_hard",
            "code_shape": "46",
            "difficulty": "hard",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_suffix="\n\nThe secret code is:",
            metadata_validator=lambda metadata: _metadata_has_mastermind_fields(
                metadata,
                variant="mastermind_46_hard",
                code_shape="46",
                difficulty="hard",
            ),
        ),
    ),
    "mutual": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mutual(batch_size=24, max_rows=128),
        expected_name="mutual",
        baseline={
            "acc,ll": 0.3984375,
            "acc,ll_avg": 0.2734375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "tasksource/mutual",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Dialogue: ",
            prompt_substrings=("\nReply options:\n", "\nA. ", "\nB. ", "\nC. ", "\nD. ", "\nAnswer:"),
            metadata_validator=_metadata_has_choice_labels(exact_count=4),
        ),
        abs_tolerance=3 / 128,
    ),
    "mc_taco": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mc_taco(batch_size=24, max_rows=128),
        expected_name="mc_taco",
        baseline={
            "acc,ll": 0.4609375,
            "acc,ll_avg": 0.4609375,
            "f1,ll_boolean": 0.488888888888889,
            "f1,ll_avg_boolean": 0.488888888888889,
        },
        expected_metrics=frozenset(
            {
                "acc,ll",
                "acc,ll_avg",
                "f1,ll_boolean",
                "f1,ll_avg_boolean",
            }
        ),
        expected_metadata={
            "stream": False,
            "dataset_path": "CogComp/mc_taco",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"no", "yes"},
            prediction_values={"no", "yes"},
            prompt_substrings=("\nQuestion: ", "\nAnswer: ", "\nPlausible:"),
            metadata_validator=_metadata_field_truthy("category"),
        ),
    ),
    "medmcqa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.medmcqa(batch_size=24, stream=True, max_rows=128),
        expected_name="medmcqa",
        baseline={
            "acc,ll": 0.5,
            "acc,ll_avg": 0.5,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "openlifescienceai/medmcqa",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="Question: ",
            prompt_suffix="\nAnswer:",
            prompt_substrings=("\nChoices:\nA. ", "\nD. "),
            metadata_validator=_metadata_has_choice_labels(exact_count=4),
        ),
    ),
    "medqa_4options": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.medqa_4options(batch_size=24, stream=True, max_rows=128),
        expected_name="medqa_4options",
        baseline={
            "acc,ll": 0.4140625,
            "acc,ll_avg": 0.4140625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "GBaker/MedQA-USMLE-4-options-hf",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_prefix="Question: ",
            prompt_suffix="\nAnswer:",
            prompt_substrings=("\nA. ", "\nD. "),
            metadata_validator=_metadata_has_choice_labels(exact_count=4),
        ),
    ),
    "mediqa_qa2019": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mediqa_qa2019(
            batch_size=4,
            max_rows=32,
            max_new_tokens=256,
        ),
        expected_name="mediqa_qa2019",
        baseline={
            "bleu": 0.008962710112574834,
            "rouge1": 0.14323768511539267,
            "rouge2": 0.030806437286732154,
            "rougeL": 0.09509368378967327,
        },
        expected_metrics=frozenset({"bleu", "rouge1", "rouge2", "rougeL"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "bigbio/mediqa_qa",
            "dataset_name": "mediqa_qa_source",
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_medical_answer_quality",
            "primary_metric": "rouge1",
            "prompt_variant": "instruction_plus_patient_question",
            "source_url": "https://github.com/abachaa/MEDIQA2019/archive/refs/heads/master.zip",
            "source_sha256": "c078ff1fe9132cf5eb89234233b1c61edeb39af21444cf5f8d04b737df52c11f",
            "omitted_upstream_metrics": ["bert_score", "bleurt"],
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_generated_bleu_rouge_sample(
            sample,
            index,
            prompt_prefix=(
                "Instructions: The following text is a question asked by a patient. "
                "Answer how a doctor would, while trying to be as informative and helpful "
                "as possible."
            ),
            prompt_substrings=("\n\nQuestion: ", "\n\nAnswer:"),
            metadata_validator=_metadata_has_mediqa_qa2019_fields,
            allow_empty_prediction=True,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "meqsum": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.meqsum(
            batch_size=8,
            max_rows=32,
            max_new_tokens=64,
        ),
        expected_name="meqsum",
        baseline={
            "bleu": 0.007607246198813463,
            "rouge1": 0.05866326457095538,
            "rouge2": 0.006186420521077284,
            "rougeL": 0.055913160793454354,
        },
        expected_metrics=frozenset({"bleu", "rouge1", "rouge2", "rougeL"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "bigbio/meqsum",
            "dataset_name": "meqsum_source",
            "split": "train",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_medical_question_summary",
            "primary_metric": "rouge1",
            "prompt_variant": "instruction_plus_source_question",
            "source_url": "https://github.com/abachaa/MeQSum/raw/master/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx",
            "source_sha256": "abedd939d5014306ca576522416bf69103e85dc8fcf1668a4099e8b84a39eeea",
            "omitted_upstream_metrics": ["bert_score", "bleurt"],
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_generated_bleu_rouge_sample(
            sample,
            index,
            prompt_prefix=(
                "Instructions: The following text is contains a medical question. "
                "Extract and summarize the question."
            ),
            metadata_validator=_metadata_has_meqsum_fields,
            allow_empty_prediction=True,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "mlqa_en_en": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mlqa_en_en(
            batch_size=4,
            max_rows=32,
        ),
        expected_name="mlqa_en_en",
        baseline={
            "em": 0.21875,
            "f1": 0.41679615383717333,
        },
        expected_metrics=frozenset({"em", "f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "facebook/mlqa",
            "dataset_name": "mlqa.en.en",
            "split": "test",
            "order": "native",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_mlqa_exact_match_f1",
            "primary_metric": "f1",
            "context_language": "en",
            "question_language": "en",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_mlqa_sample(
            sample,
            index,
            context_language="en",
            question_language="en",
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "mmlu_redux_stem_abstract_algebra": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mmlu_redux(
            subsets="stem.abstract_algebra",
            batch_size=24,
            max_rows=32,
        ),
        expected_name="mmlu_redux_stem_abstract_algebra",
        baseline={
            "acc,ll": 0.15625,
            "acc,ll_avg": 0.15625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "fxmarty/mmlu-redux-2.0-ok",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "subsets": ["stem.abstract_algebra"],
            "subset_paths": [["stem", "abstract_algebra"]],
            "subset_kinds": ["leaf"],
            "selection_mode": "single",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"A", "B", "C", "D"},
            prediction_values={"A", "B", "C", "D"},
            prompt_substrings=(
                "\nA. ",
                "\nB. ",
                "\nC. ",
                "\nD. ",
                "\nPlease respond with the correct letter (A, B, C or D) without any additional comments, only the correct letter:",
            ),
            prompt_suffix="only the correct letter:",
            metadata_validator=_metadata_has_mmlu_redux_fields(
                subset="stem.abstract_algebra",
                subject="abstract_algebra",
            ),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "mmlu_pro_plus_stem_math": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mmlu_pro_plus(
            subsets="stem.math",
            num_fewshot=5,
            batch_size=1,
            max_new_tokens=128,
            stream=True,
            max_rows=32,
        ),
        expected_name="mmlu_pro_plus_stem_math",
        baseline={"em,choice_label": 0.1875},
        expected_metrics=frozenset({"em,choice_label"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "saeidasgari/mmlu-pro-plus",
            "dataset_name": None,
            "split": "test",
            "fewshot_split": "validation",
            "num_fewshot": 5,
            "subsets": ["stem.math"],
            "subset_paths": [["stem", "math"]],
            "subset_kinds": ["leaf"],
            "selection_mode": "single",
            "apply_chat_template": False,
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_choice_label_exact_match",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_mmlu_pro_sample(
            sample,
            index,
            allowed_subsets={"stem.math"},
            max_choice_count=16,
        ),
        result_validator=_validate_mmlu_pro_result,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "mmlu_all": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.mmlu(
            subsets="all",
            num_fewshot=5,
            batch_size=24,
            stream=True,
            max_rows=128,
        ),
        expected_name="mmlu",
        baseline={
            "acc,ll": 0.3125,
            "acc,ll_avg": 0.3125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "cais/mmlu",
            "dataset_name": "all",
            "subsets": ["all"],
            "subset_paths": [["all"]],
            "subset_kinds": ["all"],
            "selection_mode": "single",
            "split": "test",
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
            stream=True,
            max_rows=32,
        ),
        expected_name="mmlu_stem",
        baseline={
            "acc,ll": 0.1875,
            "acc,ll_avg": 0.1875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "cais/mmlu",
            "dataset_name": "all",
            "subsets": ["stem"],
            "subset_paths": [["stem"]],
            "subset_kinds": ["node"],
            "selection_mode": "single",
            "split": "test",
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
        suite_factory=lambda: evalution.benchmarks.mnli(batch_size=24, stream=True, max_rows=128),
        expected_name="mnli",
        baseline={
            "acc,ll": 0.4765625,
            "acc,ll_avg": 0.4765625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
        suite_factory=lambda: evalution.benchmarks.mrpc(batch_size=24, stream=True, max_rows=128),
        expected_name="mrpc",
        baseline={
            "acc,ll": 0.6015625,
            "acc,ll_avg": 0.6015625,
            "f1,ll_boolean": 0.7301587301587302,
            "f1,ll_avg_boolean": 0.7301587301587302,
        },
        expected_metrics=frozenset(
            {
                "acc,ll",
                "acc,ll_avg",
                "f1,ll_boolean",
                "f1,ll_avg_boolean",
            }
        ),
        expected_metadata={
            "stream": True,
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
    "multirc": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.multirc(batch_size=24, stream=True, max_rows=128),
        expected_name="multirc",
        baseline={
            "acc,ll": 0.421875,
            "acc,ll_avg": 0.421875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "super_glue",
            "dataset_name": "multirc",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
            "order": "native",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_substrings=("\nQuestion: ", "\nAnswer:"),
            metadata_validator=_metadata_has_multirc_fields,
        ),
    ),
    "openbookqa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.openbookqa(batch_size=24, stream=True, max_rows=128),
        expected_name="openbookqa",
        baseline={
            "acc,ll": 0.28125,
            "acc,ll_avg": 0.4453125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "allenai/openbookqa",
            "dataset_name": "main",
            "split": "test",
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
            stream=True,
            max_rows=128,
            label_permutations=0.25,
        ),
        expected_name="openbookqa",
        baseline={
            "acc,ll": 0.28125,
            "acc,ll_avg": 0.4453125,
            "acc,label_perm:0.25": 0.625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg", "acc,label_perm:0.25"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "allenai/openbookqa",
            "dataset_name": "main",
            "split": "test",
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
        suite_factory=lambda: evalution.benchmarks.piqa(batch_size=24, stream=True, max_rows=128),
        expected_name="piqa",
        baseline={
            "acc,ll": 0.703125,
            "acc,ll_avg": 0.7578125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
    "piqa_ar": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.piqa_ar(batch_size=24, stream=True, max_rows=128),
        expected_name="piqa_ar",
        baseline={
            "acc,ll": 0.5625,
            "acc,ll_avg": 0.5390625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "Hennara/pica_ar",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="السؤال: ",
            prompt_suffix="\nالجواب:",
            metadata_validator=_metadata_field_in("source_benchmark", {"piqa"}),
        ),
    ),
    "pile_10k": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.pile_10k(batch_size=1, max_rows=32),
        expected_name="pile_10k",
        baseline={
            "word_perplexity": 49.77654296230349,
            "byte_perplexity": 1.8463441469459794,
            "bits_per_byte": 0.884671487337572,
        },
        expected_metrics=frozenset({"word_perplexity", "byte_perplexity", "bits_per_byte"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "monology/pile-uncopyrighted",
            "dataset_name": None,
            "split": "train",
            "scoring_mode": "rolling_loglikelihood_perplexity",
            "primary_metric": "word_perplexity",
        },
        expected_sample_count=32,
        sample_validator=_assert_pile_10k_sample,
        abs_tolerance=0.05,
    ),
    "prost": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.prost(batch_size=24, max_rows=128),
        expected_name="prost",
        baseline={
            "acc,ll": 0.1640625,
            "acc,ll_avg": 0.1640625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "corypaik/prost",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_substrings=("\nQuestion: ",),
            prompt_suffix="\nAnswer:",
            metadata_validator=_metadata_fields_truthy("group", "name"),
        ),
    ),
    "pubmedqa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.pubmedqa(batch_size=24, max_rows=128),
        expected_name="pubmedqa",
        baseline={
            "acc,ll": 0.9765625,
            "acc,ll_avg": 0.9765625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "bigbio/pubmed_qa",
            "dataset_name": "pubmed_qa_labeled_fold0_source",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"yes", "no", "maybe"},
            prediction_values={"yes", "no", "maybe"},
            prompt_prefix="Abstract: ",
            prompt_substrings=("\nQuestion: ",),
            prompt_suffix="\nAnswer:",
            metadata_validator=_metadata_fields_truthy("pubid", "long_answer"),
        ),
    ),
    "qa4mre_2011": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.qa4mre_2011(batch_size=1, max_rows=32),
        expected_name="qa4mre_2011",
        baseline={"acc,ll": 0.40625, "acc,ll_avg": 0.5},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "qa4mre",
            "dataset_name": "2011.main.EN",
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_suffix="\nAnswer:",
            metadata_validator=_metadata_fields_truthy(
                "year", "topic_id", "topic_name", "test_id", "document_id", "question_id", "question", "correct_answer_id"
            ),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "qa4mre_2012": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.qa4mre_2012(batch_size=1, max_rows=32),
        expected_name="qa4mre_2012",
        baseline={"acc,ll": 0.5, "acc,ll_avg": 0.46875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "qa4mre",
            "dataset_name": "2012.main.EN",
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_suffix="\nAnswer:",
            metadata_validator=_metadata_fields_truthy(
                "year", "topic_id", "topic_name", "test_id", "document_id", "question_id", "question", "correct_answer_id"
            ),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "qa4mre_2013": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.qa4mre_2013(batch_size=1, max_rows=32),
        expected_name="qa4mre_2013",
        baseline={"acc,ll": 0.46875, "acc,ll_avg": 0.59375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "qa4mre",
            "dataset_name": "2013.main.EN",
            "split": "train",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_suffix="\nAnswer:",
            metadata_validator=_metadata_fields_truthy(
                "year", "topic_id", "topic_name", "test_id", "document_id", "question_id", "question", "correct_answer_id"
            ),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "qasper_bool": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.qasper_bool(batch_size=8, max_rows=32),
        expected_name="qasper_bool",
        baseline={
            "acc,ll": 0.59375,
            "acc,ll_avg": 0.59375,
            "f1,ll_boolean": 0.7450980392156863,
            "f1,ll_avg_boolean": 0.7450980392156863,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg", "f1,ll_boolean", "f1,ll_avg_boolean"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "allenai/qasper",
            "dataset_name": None,
            "split": "validation",
            "order": "native",
            "variant": "bool",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"yes", "no"},
            prediction_values={"yes", "no"},
            prompt_prefix="TITLE: ",
            prompt_substrings=("\nABSTRACT: ", "\n\nQ: "),
            prompt_suffix="\n\nA:",
            metadata_validator=_metadata_has_qasper_fields(answer_type="bool"),
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "qasper_freeform": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.qasper_freeform(batch_size=4, max_rows=32, max_new_tokens=64),
        expected_name="qasper_freeform",
        baseline={
            "f1": 0.09263123828222714,
        },
        expected_metrics=frozenset({"f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "allenai/qasper",
            "dataset_name": None,
            "split": "validation",
            "order": "native",
            "generation_submission_mode": "continuous_refill",
            "variant": "freeform",
            "scoring_mode": "generated_qasper_abstractive_f1",
            "primary_metric": "f1",
        },
        expected_sample_count=32,
        sample_validator=_assert_qasper_freeform_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "qnli": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.qnli(batch_size=24, stream=True, max_rows=128),
        expected_name="qnli",
        baseline={
            "acc,ll": 0.4609375,
            "acc,ll_avg": 0.4609375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
        suite_factory=lambda: evalution.benchmarks.qqp(batch_size=24, stream=True, max_rows=128),
        expected_name="qqp",
        baseline={
            "acc,ll": 0.6328125,
            "acc,ll_avg": 0.6328125,
            "f1,ll_boolean": 0.4835164835164836,
            "f1,ll_avg_boolean": 0.4835164835164836,
        },
        expected_metrics=frozenset(
            {
                "acc,ll",
                "acc,ll_avg",
                "f1,ll_boolean",
                "f1,ll_avg_boolean",
            }
        ),
        expected_metadata={
            "stream": True,
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
    "race": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.race(batch_size=24, max_rows=128),
        expected_name="race",
        baseline={
            "acc,ll": 0.4609375,
            "acc,ll_avg": 0.4375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "EleutherAI/race",
            "dataset_name": "high",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Article: ",
            metadata_validator=_metadata_has_choice_labels(exact_count=4),
        ),
    ),
    "record": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.record(batch_size=24, stream=True, max_rows=128),
        expected_name="record",
        baseline={
            "acc,ll": 0.171875,
            "acc,ll_avg": 0.0859375,
            "em": 0.234375,
            "f1": 0.234375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg", "em", "f1"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "super_glue",
            "dataset_name": "record",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
            "primary_metric": "f1",
            "order": "native",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="record query: ",
            prompt_substrings=(" entities: ", " passage: "),
            metadata_validator=_metadata_has_record_fields,
        ),
    ),
    "rte": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.rte(batch_size=24, stream=True, max_rows=128),
        expected_name="rte",
        baseline={
            "acc,ll": 0.625,
            "acc,ll_avg": 0.625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
        suite_factory=lambda: evalution.benchmarks.sciq(batch_size=24, stream=True, max_rows=128),
        expected_name="sciq",
        baseline={
            "acc,ll": 0.9296875,
            "acc,ll_avg": 0.8984375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "allenai/sciq",
            "dataset_name": None,
            "split": "test",
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
    "scrolls_qasper": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.scrolls_qasper(batch_size=4, max_rows=32),
        expected_name="scrolls_qasper",
        baseline={
            "em": 0.03125,
            "f1": 0.22782614469615695,
        },
        expected_metrics=frozenset({"em", "f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "tau/scrolls",
            "dataset_name": "qasper",
            "split": "validation",
            "order": "native",
            "generation_submission_mode": "continuous_refill",
            "variant": "qasper",
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_scrolls_qa_sample(
            sample,
            index,
            variant="qasper",
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "siqa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.siqa(batch_size=24, max_rows=128),
        expected_name="siqa",
        baseline={
            "acc,ll": 0.3984375,
            "acc,ll_avg": 0.453125,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "allenai/social_i_qa",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Q: ",
            prompt_suffix="\nA:",
        ),
    ),
    "simple_cooccurrence_bias": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.simple_cooccurrence_bias(
            batch_size=24,
            max_rows=128,
        ),
        expected_name="simple_cooccurrence_bias",
        baseline={
            "likelihood_diff": -1.1651831635765997,
            "pct_male_preferred": 0.8671875,
        },
        expected_metrics=frozenset({"likelihood_diff", "pct_male_preferred"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "oskarvanderwal/simple-cooccurrence-bias",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "grouped_choice_loglikelihood_bias",
            "primary_metric": "pct_male_preferred",
            "choice_texts": ["female", "woman", "male", "man"],
        },
        expected_sample_count=128,
        sample_validator=_assert_simple_cooccurrence_bias_sample,
    ),
    "swag": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.swag(batch_size=24, stream=True, max_rows=128),
        expected_name="swag",
        baseline={
            "acc,ll": 0.4921875,
            "acc,ll_avg": 0.6484375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "swag",
            "dataset_name": "regular",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_has_choice_labels(exact_count=4),
        ),
    ),
    "swde": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.swde(
            batch_size=24,
            max_rows=128,
            max_new_tokens=48,
        ),
        expected_name="swde",
        baseline={"contains": 0.9296875},
        expected_metrics=frozenset({"contains"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "hazyresearch/based-swde-v2",
            "dataset_name": "default",
            "split": "validation",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_contains_match",
            "primary_metric": "contains",
            "prompt_variant": "webpage_attribute_completion",
        },
        expected_sample_count=128,
        sample_validator=_assert_generated_contains_sample,
    ),
    "sst2": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.sst2(batch_size=24, stream=True, max_rows=128),
        expected_name="sst2",
        baseline={
            "acc,ll": 0.5,
            "acc,ll_avg": 0.5,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
    "squad_completion": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.squad_completion(
            batch_size=24,
            max_rows=128,
            max_new_tokens=48,
        ),
        expected_name="squad_completion",
        baseline={"contains": 0.421875},
        expected_metrics=frozenset({"contains"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "hazyresearch/based-squad",
            "dataset_name": "default",
            "split": "validation",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_contains_match",
            "primary_metric": "contains",
            "prompt_variant": "truncated_context_completion",
        },
        expected_sample_count=128,
        sample_validator=_assert_generated_contains_sample,
    ),
    "squadv2": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.squadv2(batch_size=16, max_rows=32),
        expected_name="squadv2",
        baseline={
            "em": 0.375,
            "f1": 0.3819444444444444,
        },
        expected_metrics=frozenset({"em", "f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "squad_v2",
            "dataset_name": "squad_v2",
            "split": "validation",
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
            "no_answer_token": "unanswerable",
        },
        expected_sample_count=32,
        sample_validator=_assert_squadv2_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "triviaqa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.triviaqa(batch_size=16, max_rows=32),
        expected_name="triviaqa",
        baseline={
            "em": 0.1875,
            "f1": 0.25358089581501343,
        },
        expected_metrics=frozenset({"em", "f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "trivia_qa",
            "dataset_name": "rc.nocontext",
            "split": "validation",
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
        },
        expected_sample_count=32,
        sample_validator=_assert_triviaqa_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "niah_single_1": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.niah_single_1(
            batch_size=4,
            max_rows=32,
        ),
        expected_name="niah_single_1",
        baseline={"contains_fraction": 1.0},
        expected_metrics=frozenset({"contains_fraction"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "NVIDIA/RULER",
            "dataset_name": "niah_single_1",
            "split": "test",
            "order": "native",
            "generation_submission_mode": "continuous_refill",
            "variant": "niah_single_1",
            "max_length": 4096,
            "scoring_mode": "generated_contains_fraction",
            "primary_metric": "contains_fraction",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_ruler_sample(
            sample,
            index,
            variant="niah_single_1",
            max_length=4096,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "nq_open": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.nq_open(batch_size=16, max_rows=32),
        expected_name="nq_open",
        baseline={
            "em": 0.15625,
            "f1": 0.2424139492753623,
        },
        expected_metrics=frozenset({"em", "f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "nq_open",
            "dataset_name": "nq_open",
            "split": "validation",
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
        },
        expected_sample_count=32,
        sample_validator=_assert_nq_open_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "noticia": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.noticia(
            batch_size=8,
            max_rows=32,
            max_new_tokens=64,
        ),
        expected_name="noticia",
        baseline=_select_llama3_2_gpu_baseline(
            default={
                "rouge1": 0.06413572119903121,
                "average_len": 24.96875,
            },
            rtx4090={
                "rouge1": 0.060665674326899545,
                "average_len": 26.0,
            },
            a100={
                "rouge1": 0.06413572119903121,
                "average_len": 24.96875,
            },
        ),
        expected_metrics=frozenset({"rouge1", "average_len"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "Iker/NoticIA",
            "dataset_name": None,
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_clickbait_truth_summary",
            "primary_metric": "rouge1",
            "prompt_variant": "headline_body_to_truth_summary",
        },
        expected_sample_count=32,
        sample_validator=_assert_noticia_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xlsum_es": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xlsum_es(
            batch_size=4,
            max_new_tokens=128,
            max_rows=32,
        ),
        expected_name="xlsum_es",
        baseline={
            "rouge1": 0.12894381902528843,
            "rouge2": 0.031865469410523234,
            "rougeLsum": 0.08805611967197531,
        },
        expected_metrics=frozenset({"rouge1", "rouge2", "rougeLsum"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "csebuetnlp/xlsum",
            "dataset_name": "spanish",
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_summary_rouge",
            "primary_metric": "rougeLsum",
            "archive_filename": "data/spanish_XLSum_v2.0.tar.bz2",
            "archive_sha256": "70499154fe1d1c8df3b4667921d2c8c7b508da5473aa9387c4330b3b22288360",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_generated_summary_sample(
            sample,
            index,
            prompt_prefix="Texto: ",
            prompt_suffix="\n\nResumen:",
            metadata_validator=_metadata_has_xlsum_es_fields,
            allow_empty_prediction=True,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "coqa": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.coqa(batch_size=16, max_rows=32),
        expected_name="coqa",
        baseline={
            "em": 0.3125,
            "f1": 0.46814123376623373,
        },
        expected_metrics=frozenset({"em", "f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "coqa",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
            "prompt_mode": "gold_history_conversation",
        },
        expected_sample_count=32,
        sample_validator=_assert_coqa_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "drop": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.drop(batch_size=16, max_rows=32),
        expected_name="drop",
        baseline={
            "em": 0.21875,
            "f1": 0.3130140692640693,
        },
        expected_metrics=frozenset({"em", "f1"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "drop",
            "dataset_name": None,
            "split": "validation",
            "scoring_mode": "generated_qa_exact_match_f1",
            "primary_metric": "f1",
        },
        expected_sample_count=32,
        sample_validator=_assert_drop_sample,
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "fld": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.fld(batch_size=24, max_rows=128),
        expected_name="fld",
        baseline={"em": 0.0},
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "hitachi-nlp/FLD.v2",
            "dataset_name": "default",
            "split": "test",
            "generation_submission_mode": "continuous_refill",
            "scoring_mode": "generated_exact_match",
            "primary_metric": "em",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_generated_exact_match_sample(
            sample,
            index,
            prompt_prefix="Based on the provided facts ($context$), either prove or disprove the hypothesis or state that it is unknown. ",
            metadata_validator=_metadata_has_fld_fields,
            allow_empty_prediction=True,
        ),
    ),
    "french_bench_arc_challenge": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.french_bench_arc_challenge(
            batch_size=24,
            stream=True,
            max_rows=128,
        ),
        expected_name="french_bench_arc_challenge",
        baseline={
            "acc,ll": 0.21875,
            "acc,ll_avg": 0.34375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "manu/french_bench_arc_challenge",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Question: ",
            prompt_suffix="\nRéponse:",
            metadata_validator=_metadata_has_french_bench_arc_challenge_fields,
        ),
    ),
    "wic": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.wic(batch_size=24, stream=True, max_rows=128),
        expected_name="wic",
        baseline={
            "acc,ll": 0.421875,
            "acc,ll_avg": 0.421875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
    "webqs": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.webqs(batch_size=24, stream=True, max_rows=128),
        expected_name="webqs",
        baseline={
            "em": 0.1328125,
        },
        expected_metrics=frozenset({"em"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "web_questions",
            "dataset_name": None,
            "split": "test",
            "scoring_mode": "accepted_alias_greedy_exact_match",
            "primary_metric": "em",
        },
        expected_sample_count=128,
        sample_validator=_assert_webqs_sample,
    ),
    "wmdp_bio": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.wmdp_bio(batch_size=24, max_rows=128),
        expected_name="wmdp_bio",
        baseline={
            "acc,ll": 0.3359375,
            "acc,ll_avg": 0.34375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "walledai/WMDP",
            "dataset_name": None,
            "split": "bio",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Question: ",
            prompt_substrings=("\nA. ", "\nB. ", "\nC. ", "\nD. ", "\nAnswer:"),
            metadata_validator=_metadata_field_in("subset", {"bio"}),
        ),
    ),
    "wmdp_chem": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.wmdp_chem(batch_size=24, max_rows=128),
        expected_name="wmdp_chem",
        baseline={
            "acc,ll": 0.3125,
            "acc,ll_avg": 0.3515625,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "walledai/WMDP",
            "dataset_name": None,
            "split": "chem",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Question: ",
            prompt_substrings=("\nA. ", "\nB. ", "\nC. ", "\nD. ", "\nAnswer:"),
            metadata_validator=_metadata_field_in("subset", {"chem"}),
        ),
    ),
    "wmdp_cyber": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.wmdp_cyber(batch_size=24, max_rows=128),
        expected_name="wmdp_cyber",
        baseline={
            "acc,ll": 0.390625,
            "acc,ll_avg": 0.3046875,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "walledai/WMDP",
            "dataset_name": None,
            "split": "cyber",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            prompt_prefix="Question: ",
            prompt_substrings=("\nA. ", "\nB. ", "\nC. ", "\nD. ", "\nAnswer:"),
            metadata_validator=_metadata_field_in("subset", {"cyber"}),
        ),
    ),
    "xwinograd_en": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xwinograd_en(batch_size=16, stream=True, max_rows=32),
        expected_name="xwinograd_en",
        baseline={"acc,ll": 0.75, "acc,ll_avg": 0.75},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "Muennighoff/xwinograd",
            "dataset_name": "en",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "partial_evaluation_blank_replacement",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xwinograd_sample(sample, index, language="en"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xwinograd_fr": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xwinograd_fr(batch_size=16, stream=True, max_rows=32),
        expected_name="xwinograd_fr",
        baseline={"acc,ll": 0.71875, "acc,ll_avg": 0.71875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "Muennighoff/xwinograd",
            "dataset_name": "fr",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "partial_evaluation_blank_replacement",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xwinograd_sample(sample, index, language="fr"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xwinograd_jp": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xwinograd_jp(batch_size=16, stream=True, max_rows=32),
        expected_name="xwinograd_jp",
        baseline={"acc,ll": 0.59375, "acc,ll_avg": 0.59375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "Muennighoff/xwinograd",
            "dataset_name": "jp",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "partial_evaluation_blank_replacement",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xwinograd_sample(sample, index, language="jp"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xwinograd_pt": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xwinograd_pt(batch_size=16, stream=True, max_rows=32),
        expected_name="xwinograd_pt",
        baseline={"acc,ll": 0.59375, "acc,ll_avg": 0.59375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "Muennighoff/xwinograd",
            "dataset_name": "pt",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "partial_evaluation_blank_replacement",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xwinograd_sample(sample, index, language="pt"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xwinograd_ru": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xwinograd_ru(batch_size=16, stream=True, max_rows=32),
        expected_name="xwinograd_ru",
        baseline={"acc,ll": 0.65625, "acc,ll_avg": 0.65625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "Muennighoff/xwinograd",
            "dataset_name": "ru",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "partial_evaluation_blank_replacement",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xwinograd_sample(sample, index, language="ru"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xwinograd_zh": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xwinograd_zh(batch_size=16, stream=True, max_rows=32),
        expected_name="xwinograd_zh",
        baseline={"acc,ll": 0.6875, "acc,ll_avg": 0.6875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "Muennighoff/xwinograd",
            "dataset_name": "zh",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "partial_evaluation_blank_replacement",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xwinograd_sample(sample, index, language="zh"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xstorycloze_ar": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xstorycloze_ar(batch_size=24, stream=True, max_rows=32),
        expected_name="xstorycloze_ar",
        baseline={"acc,ll": 0.59375, "acc,ll_avg": 0.625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "juletxara/xstory_cloze",
            "dataset_name": "ar",
            "split": "eval",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xstorycloze_sample(sample, index, language="ar"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xstorycloze_en": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xstorycloze_en(batch_size=24, stream=True, max_rows=32),
        expected_name="xstorycloze_en",
        baseline={"acc,ll": 0.65625, "acc,ll_avg": 0.6875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "juletxara/xstory_cloze",
            "dataset_name": "en",
            "split": "eval",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xstorycloze_sample(sample, index, language="en"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xstorycloze_es": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xstorycloze_es(batch_size=24, stream=True, max_rows=32),
        expected_name="xstorycloze_es",
        baseline={"acc,ll": 0.59375, "acc,ll_avg": 0.59375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "juletxara/xstory_cloze",
            "dataset_name": "es",
            "split": "eval",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xstorycloze_sample(sample, index, language="es"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xstorycloze_eu": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xstorycloze_eu(batch_size=24, stream=True, max_rows=32),
        expected_name="xstorycloze_eu",
        baseline={"acc,ll": 0.4375, "acc,ll_avg": 0.4375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "juletxara/xstory_cloze",
            "dataset_name": "eu",
            "split": "eval",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xstorycloze_sample(sample, index, language="eu"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xstorycloze_hi": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xstorycloze_hi(batch_size=24, stream=True, max_rows=32),
        expected_name="xstorycloze_hi",
        baseline={"acc,ll": 0.5, "acc,ll_avg": 0.59375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "juletxara/xstory_cloze",
            "dataset_name": "hi",
            "split": "eval",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xstorycloze_sample(sample, index, language="hi"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xstorycloze_id": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xstorycloze_id(batch_size=24, stream=True, max_rows=32),
        expected_name="xstorycloze_id",
        baseline={"acc,ll": 0.59375, "acc,ll_avg": 0.59375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "juletxara/xstory_cloze",
            "dataset_name": "id",
            "split": "eval",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xstorycloze_sample(sample, index, language="id"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xstorycloze_my": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xstorycloze_my(batch_size=24, stream=True, max_rows=32),
        expected_name="xstorycloze_my",
        baseline={"acc,ll": 0.5, "acc,ll_avg": 0.46875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "juletxara/xstory_cloze",
            "dataset_name": "my",
            "split": "eval",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xstorycloze_sample(sample, index, language="my"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xstorycloze_ru": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xstorycloze_ru(batch_size=24, stream=True, max_rows=32),
        expected_name="xstorycloze_ru",
        baseline={"acc,ll": 0.625, "acc,ll_avg": 0.6875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "juletxara/xstory_cloze",
            "dataset_name": "ru",
            "split": "eval",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xstorycloze_sample(sample, index, language="ru"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xstorycloze_sw": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xstorycloze_sw(batch_size=24, stream=True, max_rows=32),
        expected_name="xstorycloze_sw",
        baseline={"acc,ll": 0.5625, "acc,ll_avg": 0.75},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "juletxara/xstory_cloze",
            "dataset_name": "sw",
            "split": "eval",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xstorycloze_sample(sample, index, language="sw"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xstorycloze_te": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xstorycloze_te(batch_size=24, stream=True, max_rows=32),
        expected_name="xstorycloze_te",
        baseline={"acc,ll": 0.59375, "acc,ll_avg": 0.59375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "juletxara/xstory_cloze",
            "dataset_name": "te",
            "split": "eval",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xstorycloze_sample(sample, index, language="te"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "xstorycloze_zh": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.xstorycloze_zh(batch_size=24, stream=True, max_rows=32),
        expected_name="xstorycloze_zh",
        baseline={"acc,ll": 0.5, "acc,ll_avg": 0.53125},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "juletxara/xstory_cloze",
            "dataset_name": "zh",
            "split": "eval",
            "scoring_mode": "multiple_choice_loglikelihood",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_xstorycloze_sample(sample, index, language="zh"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "winogender_all": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.winogender_all(batch_size=24, stream=True, max_rows=32),
        expected_name="winogender_all",
        baseline={"acc,ll": 0.5625, "acc,ll_avg": 0.5625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "oskarvanderwal/winogender",
            "dataset_name": "all",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "pronoun_reference_prompt",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_winogender_sample(sample, index, variant="all", gender=None),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "winogender_female": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.winogender_female(batch_size=24, stream=True, max_rows=32),
        expected_name="winogender_female",
        baseline={"acc,ll": 0.5625, "acc,ll_avg": 0.5625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "oskarvanderwal/winogender",
            "dataset_name": "all",
            "split": "test",
            "gender_filter": "female",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "pronoun_reference_prompt",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_winogender_sample(sample, index, variant="all", gender="female"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "winogender_gotcha": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.winogender_gotcha(batch_size=24, stream=True, max_rows=32),
        expected_name="winogender_gotcha",
        baseline={"acc,ll": 0.5625, "acc,ll_avg": 0.5625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "oskarvanderwal/winogender",
            "dataset_name": "gotcha",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "pronoun_reference_prompt",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_winogender_sample(sample, index, variant="gotcha", gender=None),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "winogender_gotcha_female": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.winogender_gotcha_female(batch_size=24, stream=True, max_rows=32),
        expected_name="winogender_gotcha_female",
        baseline={"acc,ll": 0.5, "acc,ll_avg": 0.46875},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "oskarvanderwal/winogender",
            "dataset_name": "gotcha",
            "split": "test",
            "gender_filter": "female",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "pronoun_reference_prompt",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_winogender_sample(sample, index, variant="gotcha", gender="female"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "winogender_gotcha_male": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.winogender_gotcha_male(batch_size=24, stream=True, max_rows=32),
        expected_name="winogender_gotcha_male",
        baseline={"acc,ll": 0.5625, "acc,ll_avg": 0.5625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "oskarvanderwal/winogender",
            "dataset_name": "gotcha",
            "split": "test",
            "gender_filter": "male",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "pronoun_reference_prompt",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_winogender_sample(sample, index, variant="gotcha", gender="male"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "winogender_male": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.winogender_male(batch_size=24, stream=True, max_rows=32),
        expected_name="winogender_male",
        baseline={"acc,ll": 0.59375, "acc,ll_avg": 0.59375},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "oskarvanderwal/winogender",
            "dataset_name": "all",
            "split": "test",
            "gender_filter": "male",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "pronoun_reference_prompt",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_winogender_sample(sample, index, variant="all", gender="male"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "winogender_neutral": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.winogender_neutral(batch_size=24, stream=True, max_rows=32),
        expected_name="winogender_neutral",
        baseline={"acc,ll": 0.65625, "acc,ll_avg": 0.65625},
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "oskarvanderwal/winogender",
            "dataset_name": "all",
            "split": "test",
            "gender_filter": "neutral",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "pronoun_reference_prompt",
        },
        expected_sample_count=32,
        sample_validator=lambda sample, index: _assert_winogender_sample(sample, index, variant="all", gender="neutral"),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_32,
    ),
    "c4": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.c4(batch_size=1, max_rows=32),
        expected_name="c4",
        baseline={
            "word_perplexity": 44.857430232245065,
            "byte_perplexity": 1.914216519142986,
            "bits_per_byte": 0.9367540238818579,
        },
        expected_metrics=frozenset({"word_perplexity", "byte_perplexity", "bits_per_byte"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "allenai/c4",
            "dataset_name": "en",
            "split": "validation",
            "scoring_mode": "rolling_loglikelihood_perplexity",
            "primary_metric": "word_perplexity",
        },
        expected_sample_count=32,
        sample_validator=_assert_c4_sample,
        abs_tolerance=0.05,
    ),
    "wikitext": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.wikitext(batch_size=1, max_rows=62),
        expected_name="wikitext",
        baseline={
            "word_perplexity": 16.73750168679846,
            "byte_perplexity": 1.6936991338088203,
            "bits_per_byte": 0.7601776192318436,
        },
        expected_metrics=frozenset({"word_perplexity", "byte_perplexity", "bits_per_byte"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "EleutherAI/wikitext_document_level",
            "dataset_name": "wikitext-2-raw-v1",
            "split": "test",
            "scoring_mode": "rolling_loglikelihood_perplexity",
            "primary_metric": "word_perplexity",
        },
        expected_sample_count=62,
        sample_validator=_assert_wikitext_sample,
    ),
    "wsc273": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.wsc273(batch_size=24, max_rows=128),
        expected_name="wsc273",
        baseline={
            "acc,ll": 0.7734375,
            "acc,ll_avg": 0.7734375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": False,
            "dataset_path": "winograd_wsc",
            "dataset_name": "wsc273",
            "split": "test",
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "partial_evaluation",
        },
        expected_sample_count=128,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            metadata_validator=_metadata_has_choice_labels(exact_count=2),
        ),
    ),
    "wsc": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.wsc(batch_size=24, stream=True, max_rows=128),
        expected_name="wsc",
        baseline={
            "acc,ll": 0.36538461538461536,
            "acc,ll_avg": 0.36538461538461536,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
            "dataset_path": "super_glue",
            "dataset_name": "wsc.fixed",
            "split": "validation",
            "scoring_mode": "multiple_choice_loglikelihood",
            "order": "native",
        },
        expected_sample_count=104,
        sample_validator=lambda sample, index: _assert_multiple_choice_loglikelihood_sample(
            sample,
            index,
            target_values={"yes", "no"},
            prediction_values={"yes", "no"},
            prompt_prefix="Passage: ",
            prompt_substrings=("does the pronoun", "\nAnswer:"),
            metadata_validator=_metadata_has_wsc_fields,
        ),
        abs_tolerance=SCORE_BASELINE_ABS_TOLERANCE_104,
    ),
    "wnli": SuiteSpec(
        suite_factory=lambda: evalution.benchmarks.wnli(batch_size=24, stream=True, max_rows=71),
        expected_name="wnli",
        baseline={
            "acc,ll": 0.5070422535211268,
            "acc,ll_avg": 0.5070422535211268,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
        suite_factory=lambda: evalution.benchmarks.winogrande(batch_size=24, stream=True, max_rows=128),
        expected_name="winogrande",
        baseline={
            "acc,ll": 0.5859375,
            "acc,ll_avg": 0.5859375,
        },
        expected_metrics=frozenset({"acc,ll", "acc,ll_avg"}),
        expected_metadata={
            "stream": True,
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
            stream=True,
            max_rows=32,
        ),
        expected_name="mmlu_pro",
        baseline={"em,choice_label": 0.125},
        expected_metrics=frozenset({"em,choice_label"}),
        expected_metadata={
            "stream": True,
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
            stream=True,
            max_rows=32,
        ),
        expected_name="mmlu_pro_stem",
        baseline={"em,choice_label": 0.34375},
        expected_metrics=frozenset({"em,choice_label"}),
        expected_metadata={
            "stream": True,
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

for _task_name, _baseline in {
    "arithmetic_1dc": 0.2421875,
    "arithmetic_2da": 1.0,
    "arithmetic_2dm": 0.625,
    "arithmetic_2ds": 0.609375,
    "arithmetic_3da": 0.78125,
    "arithmetic_3ds": 0.640625,
    "arithmetic_4da": 0.4453125,
    "arithmetic_4ds": 0.1953125,
    "arithmetic_5da": 0.203125,
    "arithmetic_5ds": 0.3203125,
}.items():
    SUITE_SPECS[_task_name] = _arithmetic_suite_spec(_task_name, _baseline)

for _task_name, _subset, _baseline in (
    ("bbh_boolean_expressions", "boolean_expressions", 0.5),
    ("bbh_causal_judgement", "causal_judgement", 0.15625),
    ("bbh_date_understanding", "date_understanding", 0.375),
    ("bbh_disambiguation_qa", "disambiguation_qa", 0.09375),
    ("bbh_dyck_languages", "dyck_languages", 0.0),
    ("bbh_formal_fallacies", "formal_fallacies", 0.03125),
    ("bbh_geometric_shapes", "geometric_shapes", 0.0),
    ("bbh_hyperbaton", "hyperbaton", 0.40625),
    ("bbh_logical_deduction_five_objects", "logical_deduction_five_objects", 0.0),
    ("bbh_logical_deduction_seven_objects", "logical_deduction_seven_objects", 0.15625),
    ("bbh_logical_deduction_three_objects", "logical_deduction_three_objects", 0.03125),
    ("bbh_movie_recommendation", "movie_recommendation", 0.0),
    ("bbh_multistep_arithmetic_two", "multistep_arithmetic_two", 0.0),
    ("bbh_navigate", "navigate", 0.375),
    ("bbh_object_counting", "object_counting", 0.375),
    ("bbh_penguins_in_a_table", "penguins_in_a_table", 0.09375),
    ("bbh_reasoning_about_colored_objects", "reasoning_about_colored_objects", 0.1875),
    ("bbh_ruin_names", "ruin_names", 0.28125),
    ("bbh_salient_translation_error_detection", "salient_translation_error_detection", 0.0),
    ("bbh_snarks", "snarks", 0.5625),
    ("bbh_sports_understanding", "sports_understanding", 0.0),
    ("bbh_temporal_sequences", "temporal_sequences", 0.125),
    ("bbh_tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_five_objects", 0.0),
    ("bbh_tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_seven_objects", 0.0),
    ("bbh_tracking_shuffled_objects_three_objects", "tracking_shuffled_objects_three_objects", 0.03125),
    ("bbh_web_of_lies", "web_of_lies", 0.46875),
    ("bbh_word_sorting", "word_sorting", 0.0),
):
    SUITE_SPECS[_task_name] = _bbh_suite_spec(_task_name, _subset, _baseline)

for _task_name, _subset, _baseline in (
    ("arabicmmlu_all", "All", {"acc,ll": 0.2890625, "acc,ll_avg": 0.2890625}),
    (
        "arabicmmlu_islamic_studies",
        "Islamic Studies",
        {"acc,ll": 0.2890625, "acc,ll_avg": 0.2890625},
    ),
    (
        "arabicmmlu_computer_science_high_school",
        "Computer Science (High School)",
        {"acc,ll": 0.3515625, "acc,ll_avg": 0.3515625},
    ),
    ("arabicmmlu_driving_test", "Driving Test", {"acc,ll": 0.453125, "acc,ll_avg": 0.453125}),
):
    SUITE_SPECS[_task_name] = _arabicmmlu_suite_spec(
        task_name=_task_name,
        subset=_subset,
        baseline=_baseline,
    )

for _task_name, _qa_split, _baseline in (
    ("babilong_qa1", "qa1", 0.0625),
    ("babilong_qa2", "qa2", 0.0),
    ("babilong_qa3", "qa3", 0.0),
    ("babilong_qa4", "qa4", 0.0),
    ("babilong_qa5", "qa5", 0.40625),
    ("babilong_qa6", "qa6", 0.03125),
    ("babilong_qa7", "qa7", 0.0),
    ("babilong_qa8", "qa8", 0.15625),
    ("babilong_qa9", "qa9", 0.40625),
    ("babilong_qa10", "qa10", 0.15625),
    ("babilong_qa11", "qa11", 0.0),
    ("babilong_qa12", "qa12", 0.0),
    ("babilong_qa13", "qa13", 0.0),
    ("babilong_qa14", "qa14", 0.03125),
    ("babilong_qa15", "qa15", 0.0),
    ("babilong_qa16", "qa16", 0.53125),
    ("babilong_qa17", "qa17", 0.25),
    ("babilong_qa18", "qa18", 0.0),
    ("babilong_qa19", "qa19", 0.0),
    ("babilong_qa20", "qa20", 0.0),
):
    SUITE_SPECS[_task_name] = _babilong_suite_spec(_task_name, _qa_split, _baseline)

for _task_name, _language in (
    ("paws_x_de", "de"),
    ("paws_x_en", "en"),
    ("paws_x_es", "es"),
    ("paws_x_fr", "fr"),
    ("paws_x_ja", "ja"),
    ("paws_x_ko", "ko"),
    ("paws_x_zh", "zh"),
):
    SUITE_SPECS[_task_name] = _paws_x_suite_spec(
        _task_name,
        language=_language,
        baseline={
            "paws_x_de": {
                "acc,ll": 0.4296875,
                "acc,ll_avg": 0.4296875,
                "f1,ll_boolean": 0.6010928961748634,
                "f1,ll_avg_boolean": 0.6010928961748634,
            },
            "paws_x_en": {
                "acc,ll": 0.3671875,
                "acc,ll_avg": 0.3671875,
                "f1,ll_boolean": 0.4671052631578947,
                "f1,ll_avg_boolean": 0.4671052631578947,
            },
            "paws_x_es": {
                "acc,ll": 0.4375,
                "acc,ll_avg": 0.4375,
                "f1,ll_boolean": 0.5135135135135135,
                "f1,ll_avg_boolean": 0.5135135135135135,
            },
            "paws_x_fr": {
                "acc,ll": 0.5703125,
                "acc,ll_avg": 0.5703125,
                "f1,ll_boolean": 0.3373493975903614,
                "f1,ll_avg_boolean": 0.3373493975903614,
            },
            "paws_x_ja": {
                "acc,ll": 0.4375,
                "acc,ll_avg": 0.4375,
                "f1,ll_boolean": 0.6086956521739131,
                "f1,ll_avg_boolean": 0.6086956521739131,
            },
            "paws_x_ko": {
                "acc,ll": 0.43359375,
                "acc,ll_avg": 0.43359375,
                "f1,ll_boolean": 0.6048114048114048,
                "f1,ll_avg_boolean": 0.6048114048114048,
            },
            "paws_x_zh": {
                "acc,ll": 0.453125,
                "acc,ll_avg": 0.453125,
                "f1,ll_boolean": 0.6195652173913043,
                "f1,ll_avg_boolean": 0.6195652173913043,
            },
        }[_task_name],
    )

for _task_name, _language in (
    ("xcopa_et", "et"),
    ("xcopa_ht", "ht"),
    ("xcopa_id", "id"),
    ("xcopa_it", "it"),
    ("xcopa_qu", "qu"),
    ("xcopa_sw", "sw"),
    ("xcopa_ta", "ta"),
    ("xcopa_th", "th"),
    ("xcopa_tr", "tr"),
    ("xcopa_vi", "vi"),
    ("xcopa_zh", "zh"),
):
    SUITE_SPECS[_task_name] = _xcopa_suite_spec(
        _task_name,
        language=_language,
        baseline={
            "xcopa_et": {
                "acc,ll": 0.515,
                "acc,ll_avg": 0.515,
            },
            "xcopa_ht": {
                "acc,ll": 0.48,
                "acc,ll_avg": 0.48,
            },
            "xcopa_id": {
                "acc,ll": 0.65,
                "acc,ll_avg": 0.65,
            },
            "xcopa_it": {
                "acc,ll": 0.675,
                "acc,ll_avg": 0.675,
            },
            "xcopa_qu": {
                "acc,ll": 0.54,
                "acc,ll_avg": 0.54,
            },
            "xcopa_sw": {
                "acc,ll": 0.6,
                "acc,ll_avg": 0.6,
            },
            "xcopa_ta": {
                "acc,ll": 0.5,
                "acc,ll_avg": 0.5,
            },
            "xcopa_th": {
                "acc,ll": 0.685,
                "acc,ll_avg": 0.685,
            },
            "xcopa_tr": {
                "acc,ll": 0.645,
                "acc,ll_avg": 0.645,
            },
            "xcopa_vi": {
                "acc,ll": 0.65,
                "acc,ll_avg": 0.65,
            },
            "xcopa_zh": {
                "acc,ll": 0.595,
                "acc,ll_avg": 0.595,
            },
        }[_task_name],
    )

for _subset, _baseline in {
    "adjunct_island": {
        "acc,ll": 0.8125,
        "acc,ll_avg": 0.8125,
    },
    "anaphor_gender_agreement": {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    },
    "animate_subject_passive": {
        "acc,ll": 0.84375,
        "acc,ll_avg": 0.875,
    },
    "animate_subject_trans": {
        "acc,ll": 0.9375,
        "acc,ll_avg": 0.75,
    },
    "complex_NP_island": {
        "acc,ll": 0.78125,
        "acc,ll_avg": 0.78125,
    },
    "determiner_noun_agreement_1": {
        "acc,ll": 0.96875,
        "acc,ll_avg": 0.96875,
    },
    "matrix_question_npi_licensor_present": {
        "acc,ll": 0.40625,
        "acc,ll_avg": 0.28125,
    },
    "npi_present_1": {
        "acc,ll": 0.59375,
        "acc,ll_avg": 0.59375,
    },
}.items():
    SUITE_SPECS[f"blimp_{_subset.lower()}"] = _blimp_suite_spec(
        subset=_subset,
        baseline=_baseline,
    )

for _subset, _baseline, _sample_count in (
    (
        "accountant",
        {
            "acc,ll": 0.3125,
            "acc,ll_avg": 0.3125,
        },
        32,
    ),
    (
        "computer_network",
        {
            "acc,ll": 0.2631578947368421,
            "acc,ll_avg": 0.2631578947368421,
        },
        19,
    ),
    (
        "high_school_physics",
        {
            "acc,ll": 0.47368421052631576,
            "acc,ll_avg": 0.47368421052631576,
        },
        19,
    ),
    (
        "law",
        {
            "acc,ll": 0.3333333333333333,
            "acc,ll_avg": 0.3333333333333333,
        },
        24,
    ),
):
    SUITE_SPECS[f"ceval_{_subset}"] = _ceval_suite_spec(
        subset=_subset,
        baseline=_baseline,
        expected_sample_count=_sample_count,
    )

for _language, _baseline in {
    "amh": {"acc,ll": 0.34375, "acc,ll_avg": 0.34375},
    "eng": {"acc,ll": 0.34375, "acc,ll_avg": 0.34375},
    "fra": {"acc,ll": 0.34375, "acc,ll_avg": 0.34375},
    "swa": {"acc,ll": 0.34375, "acc,ll_avg": 0.34375},
}.items():
    SUITE_SPECS[f"afrixnli_{_language}"] = _afrixnli_suite_spec(
        language=_language,
        baseline=_baseline,
    )

for _language, _baseline in {
    "ar": {"acc,ll": 0.3125, "acc,ll_avg": 0.3125},
    "en": {"acc,ll": 0.3125, "acc,ll_avg": 0.3125},
    "fr": {"acc,ll": 0.3125, "acc,ll_avg": 0.3125},
    "sw": {"acc,ll": 0.3125, "acc,ll_avg": 0.3125},
}.items():
    SUITE_SPECS[f"xnli_{_language}"] = _xnli_suite_spec(
        language=_language,
        baseline=_baseline,
    )

for _language, _baseline in {
    "ar": {"em": 0.1875, "f1": 0.339294733044733},
    "en": {"em": 0.15625, "f1": 0.3847293331668331},
    "es": _select_llama3_2_gpu_baseline(
        default={"em": 0.125, "f1": 0.5209415584415584},
        rtx4090={"em": 0.125, "f1": 0.5209415584415584},
        a100={"em": 0.09375, "f1": 0.42406655844155844},
    ),
    "zh": {"em": 0.375, "f1": 0.45312499999999994},
}.items():
    SUITE_SPECS[f"xquad_{_language}"] = _xquad_suite_spec(
        language=_language,
        baseline=_baseline,
    )

for _variant, _baseline in {
    "mc1": 0.3125,
    "mc2": 0.5797437857629697,
}.items():
    SUITE_SPECS[f"truthfulqa_{_variant}"] = _truthfulqa_suite_spec(
        variant=_variant,
        baseline=_baseline,
    )

for _task_name, _category, _baseline in (
    ("bbq_age", "Age", {"acc,ll": 0.34375, "acc,ll_avg": 0.34375}),
    (
        "bbq_disability_status",
        "Disability_status",
        {"acc,ll": 0.4765625, "acc,ll_avg": 0.4765625},
    ),
    (
        "bbq_gender_identity",
        "Gender_identity",
        {"acc,ll": 0.4921875, "acc,ll_avg": 0.4921875},
    ),
    (
        "bbq_nationality",
        "Nationality",
        {"acc,ll": 0.4140625, "acc,ll_avg": 0.4140625},
    ),
):
    SUITE_SPECS[_task_name] = _bbq_suite_spec(
        _task_name,
        category=_category,
        baseline=_baseline,
    )

for _task_name, _subset, _baseline in (
    (
        "inverse_scaling_hindsight_neglect",
        "hindsight-neglect",
        {"acc,ll": 0.421875, "acc,ll_avg": 0.421875},
    ),
    (
        "inverse_scaling_into_the_unknown",
        "into-the-unknown",
        {"acc,ll": 0.296875, "acc,ll_avg": 0.296875},
    ),
    (
        "inverse_scaling_resisting_correction",
        "resisting-correction",
        {"acc,ll": 0.796875, "acc,ll_avg": 0.828125},
    ),
    (
        "inverse_scaling_sig_figs",
        "sig-figs",
        {"acc,ll": 0.0625, "acc,ll_avg": 0.0625},
    ),
):
    SUITE_SPECS[_task_name] = _inverse_scaling_suite_spec(
        task_name=_task_name,
        subset=_subset,
        baseline=_baseline,
    )

for _language, _baseline in {
    "amh_Ethi": {"acc,ll": 0.28125, "acc,ll_avg": 0.28125},
    "eng_Latn": {"acc,ll": 0.53125, "acc,ll_avg": 0.53125},
    "fra_Latn": {"acc,ll": 0.5625, "acc,ll_avg": 0.5625},
    "por_Latn": {"acc,ll": 0.5625, "acc,ll_avg": 0.5625},
    "spa_Latn": {"acc,ll": 0.46875, "acc,ll_avg": 0.46875},
    "swh_Latn": {"acc,ll": 0.25, "acc,ll_avg": 0.25},
}.items():
    SUITE_SPECS[f"belebele_{_language}"] = _belebele_suite_spec(
        language=_language,
        baseline=_baseline,
    )

for _subset, _baseline in {
    "boolqa": {"acc,ll": 0.71875, "acc,ll_avg": 0.71875},
    "commonsenseqa": {"acc,ll": 0.34375, "acc,ll_avg": 0.34375},
    "mmlu": {"acc,ll": 0.34375, "acc,ll_avg": 0.34375},
    "openbookqa": {"acc,ll": 0.28125, "acc,ll_avg": 0.28125},
    "piqa": {"acc,ll": 0.546875, "acc,ll_avg": 0.546875},
}.items():
    SUITE_SPECS[f"bangla_{_subset}"] = _bangla_suite_spec(
        subset=_subset,
        baseline=_baseline,
    )

for _language, _dataset_path, _dataset_name, _baseline in (
    ("da", "LumiOpen/arc_challenge_mt", "da", {"acc,exam": 0.1875}),
    ("fi", "LumiOpen/arc_challenge_mt", "fi", {"acc,exam": 0.28125}),
    ("is", "mideind/icelandic-arc-challenge", None, {"acc,exam": 0.25}),
    ("pt", "LumiOpen/arc_challenge_mt", "pt", {"acc,exam": 0.21875}),
):
    SUITE_SPECS[f"arc_mt_{_language}"] = _arc_mt_suite_spec(
        language=_language,
        dataset_path=_dataset_path,
        dataset_name=_dataset_name,
        baseline=_baseline,
    )

for _task_name, _baseline in {
    "crows_pairs_english": {
        "pct_stereotype": 0.5625,
        "ll_diff": 3.1171875,
    },
    "crows_pairs_english_age": {
        "pct_stereotype": 0.75,
        "ll_diff": 3.5078125,
    },
    "crows_pairs_english_autre": {
        "pct_stereotype": 0.7272727272727273,
        "ll_diff": 6.613636363636363,
    },
    "crows_pairs_english_disability": {
        "pct_stereotype": 0.625,
        "ll_diff": 8.796875,
    },
    "crows_pairs_english_gender": {
        "pct_stereotype": 0.5,
        "ll_diff": 3.34375,
    },
    "crows_pairs_english_nationality": {
        "pct_stereotype": 0.625,
        "ll_diff": 3.9140625,
    },
    "crows_pairs_english_physical_appearance": {
        "pct_stereotype": 0.5625,
        "ll_diff": 4.3125,
    },
    "crows_pairs_english_race_color": {
        "pct_stereotype": 0.40625,
        "ll_diff": 3.1484375,
    },
    "crows_pairs_english_religion": {
        "pct_stereotype": 0.6875,
        "ll_diff": 3.3828125,
    },
    "crows_pairs_english_sexual_orientation": {
        "pct_stereotype": 0.6875,
        "ll_diff": 5.125,
    },
    "crows_pairs_english_socioeconomic": {
        "pct_stereotype": 0.78125,
        "ll_diff": 4.1875,
    },
    "crows_pairs_french": {
        "pct_stereotype": 0.46875,
        "ll_diff": 6.6640625,
    },
    "crows_pairs_french_age": {
        "pct_stereotype": 0.4375,
        "ll_diff": 3.7421875,
    },
    "crows_pairs_french_autre": {
        "pct_stereotype": 0.46153846153846156,
        "ll_diff": 5.288461538461538,
    },
    "crows_pairs_french_disability": {
        "pct_stereotype": 0.53125,
        "ll_diff": 7.4609375,
    },
    "crows_pairs_french_gender": {
        "pct_stereotype": 0.46875,
        "ll_diff": 4.4375,
    },
    "crows_pairs_french_nationality": {
        "pct_stereotype": 0.40625,
        "ll_diff": 5.046875,
    },
    "crows_pairs_french_physical_appearance": {
        "pct_stereotype": 0.4375,
        "ll_diff": 3.828125,
    },
    "crows_pairs_french_race_color": {
        "pct_stereotype": 0.3125,
        "ll_diff": 3.9296875,
    },
    "crows_pairs_french_religion": {
        "pct_stereotype": 0.46875,
        "ll_diff": 3.84375,
    },
    "crows_pairs_french_sexual_orientation": {
        "pct_stereotype": 0.6875,
        "ll_diff": 6.2734375,
    },
    "crows_pairs_french_socioeconomic": {
        "pct_stereotype": 0.5625,
        "ll_diff": 5.72265625,
    },
}.items():
    SUITE_SPECS[_task_name] = _crows_pairs_suite_spec(
        _task_name,
        baseline=_baseline,
        expected_sample_count={
            "crows_pairs_english_autre": 11,
            "crows_pairs_french_autre": 13,
        }.get(_task_name, 32),
    )


def run_suite_spec(
    capsys: pytest.CaptureFixture[str],
    suite_key: str,
) -> tuple[Any, Any]:
    """Run one registered suite spec and enforce its serialized and metric baselines."""

    spec = SUITE_SPECS[suite_key]
    result, test_result = run_llama3_2_suite(capsys, spec.suite_factory())
    _assert_suite_matches_spec(test_result, spec)
    assert_single_test_serialization(result, test_result)
    return result, test_result


def run_suite_specs(
    capsys: pytest.CaptureFixture[str],
    suite_keys: tuple[str, ...] | list[str],
) -> tuple[Any, list[Any]]:
    """Run suite specs."""
    suite_keys = list(suite_keys)
    specs = [SUITE_SPECS[suite_key] for suite_key in suite_keys]
    result, test_results = run_llama3_2_suites(
        capsys,
        [spec.suite_factory() for spec in specs],
    )
    for test_result, spec in zip(test_results, specs, strict=True):
        _assert_suite_matches_spec(test_result, spec)
    serialized = result.to_dict()
    assert len(serialized["tests"]) == len(test_results)
    for serialized_test, test_result in zip(serialized["tests"], test_results, strict=True):
        assert serialized_test["name"] == test_result.name
        assert len(serialized_test["samples"]) == len(test_result.samples)
        if test_result.samples:
            assert serialized_test["samples"][0]["prediction"] is not None
    return result, test_results


def run_compare_suite_spec(
    capsys: pytest.CaptureFixture[str],
    suite_key: str,
) -> tuple[Any, Any]:
    """Run compare suite spec. Keep the nested traversal explicit so ordering and metadata stay aligned."""
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


def _assert_suite_matches_spec(test_result: Any, spec: SuiteSpec) -> None:
    """Assert suite matches spec for the surrounding tests."""
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
