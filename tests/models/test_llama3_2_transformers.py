from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import evalution

LLAMA3_2_1B_INSTRUCT = Path("/monster/data/model/Llama-3.2-1B-Instruct")
# Allow up to two samples of score movement at 128 rows because continuous batching drifts slightly run to run.
SCORE_BASELINE_ABS_TOLERANCE = 2 / 128
GSM8K_BASELINE = {
    "exact_match,strict-match": 0.328125,
    "exact_match,flexible-extract": 0.3671875,
}
GSM8K_PLATINUM_BASELINE = {
    "exact_match,strict-match": 0.3515625,
    "exact_match,flexible-extract": 0.390625,
}
ARC_CHALLENGE_BASELINE = {
    "exact_match,choice-label": 0.4765625,
    "exact_match,choice-text": 0.4765625,
}
ARC_EASY_BASELINE = {
    "accuracy,loglikelihood": 0.6640625,
    "accuracy,loglikelihood_norm": 0.6484375,
}
HELLASWAG_BASELINE = {
    "accuracy,loglikelihood": 0.4375,
    "accuracy,loglikelihood_norm": 0.5390625,
}
PIQA_BASELINE = {
    "accuracy,loglikelihood": 0.71875,
    "accuracy,loglikelihood_norm": 0.7890625,
}
QNLI_BASELINE = {
    "accuracy,loglikelihood": 0.4609375,
    "accuracy,loglikelihood_norm": 0.4609375,
}
BOOLQ_BASELINE = {
    "accuracy,loglikelihood": 0.6796875,
    "accuracy,loglikelihood_norm": 0.6796875,
}
CB_BASELINE = {
    "accuracy,loglikelihood": 0.5714285714285714,
    "accuracy,loglikelihood_norm": 0.5714285714285714,
    "f1,loglikelihood_macro": 0.39345839345839345,
    "f1,loglikelihood_norm_macro": 0.39345839345839345,
}
COPA_BASELINE = {
    "accuracy,loglikelihood": 0.74,
    "accuracy,loglikelihood_norm": 0.68,
}
WINOGRANDE_BASELINE = {
    "accuracy,loglikelihood": 0.5625,
    "accuracy,loglikelihood_norm": 0.5703125,
}
OPENBOOKQA_BASELINE = {
    "accuracy,loglikelihood": 0.25,
    "accuracy,loglikelihood_norm": 0.328125,
}
MMLU_BASELINE = {
    "accuracy,loglikelihood": 0.3671875,
    "accuracy,loglikelihood_norm": 0.3671875,
}
MRPC_BASELINE = {
    "accuracy,loglikelihood": 0.6953125,
    "accuracy,loglikelihood_norm": 0.6953125,
    "f1,loglikelihood_yes": 0.8186046511627907,
    "f1,loglikelihood_norm_yes": 0.8186046511627907,
}
RTE_BASELINE = {
    "accuracy,loglikelihood": 0.625,
    "accuracy,loglikelihood_norm": 0.625,
}
SST2_BASELINE = {
    "accuracy,loglikelihood": 0.5390625,
    "accuracy,loglikelihood_norm": 0.5390625,
}
WIC_BASELINE = {
    "accuracy,loglikelihood": 0.5,
    "accuracy,loglikelihood_norm": 0.5,
}
WNLI_BASELINE = {
    "accuracy,loglikelihood": 0.4225352112676056,
    "accuracy,loglikelihood_norm": 0.4225352112676056,
}

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# Compare score maps against the stored regression baseline while allowing small continuous-batching drift.
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
                evalution.boolq(
                    batch_size=24,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.cb(
                    batch_size=24,
                    streaming=True,
                    max_rows=56,
                )
            )
            .run(
                evalution.copa(
                    batch_size=24,
                    streaming=True,
                    max_rows=100,
                )
            )
            .run(
                evalution.arc_easy(
                    batch_size=24,
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
            .run(
                evalution.mmlu(
                    subject="all",
                    num_fewshot=5,
                    batch_size=24,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.mrpc(
                    batch_size=24,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.openbookqa(
                    batch_size=24,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.piqa(
                    batch_size=24,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.qnli(
                    batch_size=24,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.rte(
                    batch_size=24,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.sst2(
                    batch_size=24,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.wic(
                    batch_size=24,
                    streaming=True,
                    max_rows=128,
                )
            )
            .run(
                evalution.wnli(
                    batch_size=24,
                    streaming=True,
                    max_rows=71,
                )
            )
            .run(
                evalution.winogrande(
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
    assert len(result.tests) == 18

    tests_by_name = {test_result.name: test_result for test_result in result.tests}
    assert set(tests_by_name) == {
        "gsm8k_cot",
        "gsm8k_platinum_cot",
        "boolq",
        "cb",
        "copa",
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "mmlu",
        "mrpc",
        "openbookqa",
        "piqa",
        "qnli",
        "rte",
        "sst2",
        "wic",
        "wnli",
        "winogrande",
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
    assert_metrics_match_baseline(gsm8k_result.metrics, GSM8K_BASELINE)

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
    assert_metrics_match_baseline(test_result.metrics, GSM8K_PLATINUM_BASELINE)
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

    boolq_result = tests_by_name["boolq"]
    assert boolq_result.metadata["streaming"] is True
    assert boolq_result.metadata["dataset_path"] == "super_glue"
    assert boolq_result.metadata["dataset_name"] == "boolq"
    assert boolq_result.metadata["split"] == "validation"
    assert boolq_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(boolq_result.samples) == 128
    assert set(boolq_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    boolq_raw_score = boolq_result.metrics["accuracy,loglikelihood"]
    boolq_norm_score = boolq_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(boolq_result.metrics, BOOLQ_BASELINE)
    assert isinstance(boolq_raw_score, float)
    assert isinstance(boolq_norm_score, float)

    for index, sample in enumerate(boolq_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target in {"yes", "no"}
        assert sample.prediction in {"yes", "no"}
        assert "\nQuestion: " in sample.prompt
        assert "\nAnswer:" in sample.prompt
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

    cb_result = tests_by_name["cb"]
    assert cb_result.metadata["streaming"] is True
    assert cb_result.metadata["dataset_path"] == "super_glue"
    assert cb_result.metadata["dataset_name"] == "cb"
    assert cb_result.metadata["split"] == "validation"
    assert cb_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(cb_result.samples) == 56
    assert set(cb_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
        "f1,loglikelihood_macro",
        "f1,loglikelihood_norm_macro",
    }
    cb_raw_score = cb_result.metrics["accuracy,loglikelihood"]
    cb_norm_score = cb_result.metrics["accuracy,loglikelihood_norm"]
    cb_macro_f1 = cb_result.metrics["f1,loglikelihood_macro"]
    cb_norm_macro_f1 = cb_result.metrics["f1,loglikelihood_norm_macro"]
    assert_metrics_match_baseline(cb_result.metrics, CB_BASELINE)
    assert isinstance(cb_raw_score, float)
    assert isinstance(cb_norm_score, float)
    assert isinstance(cb_macro_f1, float)
    assert isinstance(cb_norm_macro_f1, float)

    for index, sample in enumerate(cb_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target in {"True", "False", "Neither"}
        assert sample.prediction in {"True", "False", "Neither"}
        assert "\nQuestion: " in sample.prompt
        assert "True, False, or Neither?\nAnswer:" in sample.prompt
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

    copa_result = tests_by_name["copa"]
    assert copa_result.metadata["streaming"] is True
    assert copa_result.metadata["dataset_path"] == "super_glue"
    assert copa_result.metadata["dataset_name"] == "copa"
    assert copa_result.metadata["split"] == "validation"
    assert copa_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(copa_result.samples) == 100
    assert set(copa_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    copa_raw_score = copa_result.metrics["accuracy,loglikelihood"]
    copa_norm_score = copa_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(copa_result.metrics, COPA_BASELINE)
    assert isinstance(copa_raw_score, float)
    assert isinstance(copa_norm_score, float)

    for index, sample in enumerate(copa_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target
        assert sample.prediction
        assert sample.metadata["question"] in {"cause", "effect"}
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

    arc_easy_result = tests_by_name["arc_easy"]
    assert arc_easy_result.metadata["streaming"] is True
    assert arc_easy_result.metadata["dataset_path"] == "allenai/ai2_arc"
    assert arc_easy_result.metadata["dataset_name"] == "ARC-Easy"
    assert arc_easy_result.metadata["split"] == "validation"
    assert arc_easy_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(arc_easy_result.samples) == 128
    assert set(arc_easy_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    arc_easy_raw_score = arc_easy_result.metrics["accuracy,loglikelihood"]
    arc_easy_norm_score = arc_easy_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(arc_easy_result.metrics, ARC_EASY_BASELINE)
    assert isinstance(arc_easy_raw_score, float)
    assert isinstance(arc_easy_norm_score, float)

    for index, sample in enumerate(arc_easy_result.samples):
        assert sample.index == index
        assert sample.prompt.startswith("Question: ")
        assert sample.prompt.endswith("\nAnswer:")
        assert sample.target
        assert sample.prediction
        assert len(sample.metadata["choice_labels"]) >= 2
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
    assert_metrics_match_baseline(arc_result.metrics, ARC_CHALLENGE_BASELINE)
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
    assert_metrics_match_baseline(hellaswag_result.metrics, HELLASWAG_BASELINE)
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

    mmlu_result = tests_by_name["mmlu"]
    assert mmlu_result.metadata["streaming"] is True
    assert mmlu_result.metadata["dataset_path"] == "cais/mmlu"
    assert mmlu_result.metadata["dataset_name"] == "all"
    assert mmlu_result.metadata["split"] == "validation"
    assert mmlu_result.metadata["fewshot_split"] == "dev"
    assert mmlu_result.metadata["num_fewshot"] == 5
    assert mmlu_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(mmlu_result.samples) == 128
    assert set(mmlu_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    mmlu_raw_score = mmlu_result.metrics["accuracy,loglikelihood"]
    mmlu_norm_score = mmlu_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(mmlu_result.metrics, MMLU_BASELINE)
    assert isinstance(mmlu_raw_score, float)
    assert isinstance(mmlu_norm_score, float)

    for index, sample in enumerate(mmlu_result.samples):
        assert sample.index == index
        assert sample.prompt.startswith("The following are multiple choice questions")
        assert sample.target in {"A", "B", "C", "D"}
        assert sample.prediction in {"A", "B", "C", "D"}
        assert sample.metadata["subject"]
        assert len(sample.metadata["choice_texts"]) == 4
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

    mrpc_result = tests_by_name["mrpc"]
    assert mrpc_result.metadata["streaming"] is True
    assert mrpc_result.metadata["dataset_path"] == "nyu-mll/glue"
    assert mrpc_result.metadata["dataset_name"] == "mrpc"
    assert mrpc_result.metadata["split"] == "validation"
    assert mrpc_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(mrpc_result.samples) == 128
    assert set(mrpc_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
        "f1,loglikelihood_yes",
        "f1,loglikelihood_norm_yes",
    }
    mrpc_raw_score = mrpc_result.metrics["accuracy,loglikelihood"]
    mrpc_norm_score = mrpc_result.metrics["accuracy,loglikelihood_norm"]
    mrpc_f1_score = mrpc_result.metrics["f1,loglikelihood_yes"]
    mrpc_norm_f1_score = mrpc_result.metrics["f1,loglikelihood_norm_yes"]
    assert_metrics_match_baseline(mrpc_result.metrics, MRPC_BASELINE)
    assert isinstance(mrpc_raw_score, float)
    assert isinstance(mrpc_norm_score, float)
    assert isinstance(mrpc_f1_score, float)
    assert isinstance(mrpc_norm_f1_score, float)

    for index, sample in enumerate(mrpc_result.samples):
        assert sample.index == index
        assert sample.prompt.startswith("Sentence 1: ")
        assert "\nSentence 2: " in sample.prompt
        assert "Do both sentences mean the same thing?" in sample.prompt
        assert sample.target in {"no", "yes"}
        assert sample.prediction in {"no", "yes"}
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

    openbookqa_result = tests_by_name["openbookqa"]
    assert openbookqa_result.metadata["streaming"] is True
    assert openbookqa_result.metadata["dataset_path"] == "allenai/openbookqa"
    assert openbookqa_result.metadata["dataset_name"] == "main"
    assert openbookqa_result.metadata["split"] == "validation"
    assert openbookqa_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(openbookqa_result.samples) == 128
    assert set(openbookqa_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    openbookqa_raw_score = openbookqa_result.metrics["accuracy,loglikelihood"]
    openbookqa_norm_score = openbookqa_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(openbookqa_result.metrics, OPENBOOKQA_BASELINE)
    assert isinstance(openbookqa_raw_score, float)
    assert isinstance(openbookqa_norm_score, float)

    for index, sample in enumerate(openbookqa_result.samples):
        assert sample.index == index
        assert sample.prompt.startswith("Question: ")
        assert sample.prompt.endswith("\nAnswer:")
        assert sample.target
        assert sample.prediction
        assert len(sample.metadata["choice_labels"]) == 4
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

    piqa_result = tests_by_name["piqa"]
    assert piqa_result.metadata["streaming"] is True
    assert piqa_result.metadata["dataset_path"] == "baber/piqa"
    assert piqa_result.metadata["dataset_name"] is None
    assert piqa_result.metadata["split"] == "validation"
    assert piqa_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(piqa_result.samples) == 128
    assert set(piqa_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    piqa_raw_score = piqa_result.metrics["accuracy,loglikelihood"]
    piqa_norm_score = piqa_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(piqa_result.metrics, PIQA_BASELINE)
    assert isinstance(piqa_raw_score, float)
    assert isinstance(piqa_norm_score, float)

    for index, sample in enumerate(piqa_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target
        assert sample.prediction
        assert sample.prompt.startswith("Question: ")
        assert "\nAnswer:" in sample.prompt
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

    qnli_result = tests_by_name["qnli"]
    assert qnli_result.metadata["streaming"] is True
    assert qnli_result.metadata["dataset_path"] == "nyu-mll/glue"
    assert qnli_result.metadata["dataset_name"] == "qnli"
    assert qnli_result.metadata["split"] == "validation"
    assert qnli_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(qnli_result.samples) == 128
    assert set(qnli_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    qnli_raw_score = qnli_result.metrics["accuracy,loglikelihood"]
    qnli_norm_score = qnli_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(qnli_result.metrics, QNLI_BASELINE)
    assert isinstance(qnli_raw_score, float)
    assert isinstance(qnli_norm_score, float)

    for index, sample in enumerate(qnli_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target in {"yes", "no"}
        assert sample.prediction in {"yes", "no"}
        assert "Question: Does this response answer the question?\nAnswer:" in sample.prompt
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

    rte_result = tests_by_name["rte"]
    assert rte_result.metadata["streaming"] is True
    assert rte_result.metadata["dataset_path"] == "super_glue"
    assert rte_result.metadata["dataset_name"] == "rte"
    assert rte_result.metadata["split"] == "validation"
    assert rte_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(rte_result.samples) == 128
    assert set(rte_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    rte_raw_score = rte_result.metrics["accuracy,loglikelihood"]
    rte_norm_score = rte_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(rte_result.metrics, RTE_BASELINE)
    assert isinstance(rte_raw_score, float)
    assert isinstance(rte_norm_score, float)

    for index, sample in enumerate(rte_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target in {"True", "False"}
        assert sample.prediction in {"True", "False"}
        assert "\nQuestion: " in sample.prompt
        assert " True or False?\nAnswer:" in sample.prompt
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

    sst2_result = tests_by_name["sst2"]
    assert sst2_result.metadata["streaming"] is True
    assert sst2_result.metadata["dataset_path"] == "nyu-mll/glue"
    assert sst2_result.metadata["dataset_name"] == "sst2"
    assert sst2_result.metadata["split"] == "validation"
    assert sst2_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(sst2_result.samples) == 128
    assert set(sst2_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    sst2_raw_score = sst2_result.metrics["accuracy,loglikelihood"]
    sst2_norm_score = sst2_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(sst2_result.metrics, SST2_BASELINE)
    assert isinstance(sst2_raw_score, float)
    assert isinstance(sst2_norm_score, float)

    for index, sample in enumerate(sst2_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target in {"negative", "positive"}
        assert sample.prediction in {"negative", "positive"}
        assert "\nQuestion: Is this sentence positive or negative?\nAnswer:" in sample.prompt
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

    wic_result = tests_by_name["wic"]
    assert wic_result.metadata["streaming"] is True
    assert wic_result.metadata["dataset_path"] == "super_glue"
    assert wic_result.metadata["dataset_name"] == "wic"
    assert wic_result.metadata["split"] == "validation"
    assert wic_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(wic_result.samples) == 128
    assert set(wic_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    wic_raw_score = wic_result.metrics["accuracy,loglikelihood"]
    wic_norm_score = wic_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(wic_result.metrics, WIC_BASELINE)
    assert isinstance(wic_raw_score, float)
    assert isinstance(wic_norm_score, float)

    for index, sample in enumerate(wic_result.samples):
        assert sample.index == index
        assert sample.prompt.startswith("Sentence 1: ")
        assert "\nSentence 2: " in sample.prompt
        assert "used in the same way" in sample.prompt
        assert sample.target in {"yes", "no"}
        assert sample.prediction in {"yes", "no"}
        assert sample.metadata["word"]
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

    wnli_result = tests_by_name["wnli"]
    assert wnli_result.metadata["streaming"] is True
    assert wnli_result.metadata["dataset_path"] == "nyu-mll/glue"
    assert wnli_result.metadata["dataset_name"] == "wnli"
    assert wnli_result.metadata["split"] == "validation"
    assert wnli_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(wnli_result.samples) == 71
    assert set(wnli_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    wnli_raw_score = wnli_result.metrics["accuracy,loglikelihood"]
    wnli_norm_score = wnli_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(wnli_result.metrics, WNLI_BASELINE)
    assert isinstance(wnli_raw_score, float)
    assert isinstance(wnli_norm_score, float)

    for index, sample in enumerate(wnli_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target in {"False", "True"}
        assert sample.prediction in {"False", "True"}
        assert "\nQuestion: " in sample.prompt
        assert " True or False?\nAnswer:" in sample.prompt
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

    winogrande_result = tests_by_name["winogrande"]
    assert winogrande_result.metadata["streaming"] is True
    assert winogrande_result.metadata["dataset_path"] == "winogrande"
    assert winogrande_result.metadata["dataset_name"] == "winogrande_xl"
    assert winogrande_result.metadata["split"] == "validation"
    assert winogrande_result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(winogrande_result.samples) == 128
    assert set(winogrande_result.metrics) == {
        "accuracy,loglikelihood",
        "accuracy,loglikelihood_norm",
    }
    winogrande_raw_score = winogrande_result.metrics["accuracy,loglikelihood"]
    winogrande_norm_score = winogrande_result.metrics["accuracy,loglikelihood_norm"]
    assert_metrics_match_baseline(winogrande_result.metrics, WINOGRANDE_BASELINE)
    assert isinstance(winogrande_raw_score, float)
    assert isinstance(winogrande_norm_score, float)

    for index, sample in enumerate(winogrande_result.samples):
        assert sample.index == index
        assert sample.prompt
        assert sample.target
        assert sample.prediction
        assert " _ " in sample.metadata["sentence"]
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
    assert len(serialized_tests["cb"]["samples"]) == len(cb_result.samples)
    assert serialized_tests["cb"]["samples"][0]["prediction"]
    assert len(serialized_tests["copa"]["samples"]) == len(copa_result.samples)
    assert serialized_tests["copa"]["samples"][0]["prediction"]
    assert len(serialized_tests["arc_challenge"]["samples"]) == len(arc_result.samples)
    assert serialized_tests["arc_challenge"]["samples"][0]["prediction"]
    assert len(serialized_tests["arc_easy"]["samples"]) == len(arc_easy_result.samples)
    assert serialized_tests["arc_easy"]["samples"][0]["prediction"]
    assert len(serialized_tests["hellaswag"]["samples"]) == len(hellaswag_result.samples)
    assert serialized_tests["hellaswag"]["samples"][0]["prediction"]
    assert len(serialized_tests["mmlu"]["samples"]) == len(mmlu_result.samples)
    assert serialized_tests["mmlu"]["samples"][0]["prediction"]
    assert len(serialized_tests["mrpc"]["samples"]) == len(mrpc_result.samples)
    assert serialized_tests["mrpc"]["samples"][0]["prediction"]
    assert len(serialized_tests["openbookqa"]["samples"]) == len(openbookqa_result.samples)
    assert serialized_tests["openbookqa"]["samples"][0]["prediction"]
    assert len(serialized_tests["qnli"]["samples"]) == len(qnli_result.samples)
    assert serialized_tests["qnli"]["samples"][0]["prediction"]
    assert len(serialized_tests["rte"]["samples"]) == len(rte_result.samples)
    assert serialized_tests["rte"]["samples"][0]["prediction"]
    assert len(serialized_tests["sst2"]["samples"]) == len(sst2_result.samples)
    assert serialized_tests["sst2"]["samples"][0]["prediction"]
    assert len(serialized_tests["wic"]["samples"]) == len(wic_result.samples)
    assert serialized_tests["wic"]["samples"][0]["prediction"]
    assert len(serialized_tests["wnli"]["samples"]) == len(wnli_result.samples)
    assert serialized_tests["wnli"]["samples"][0]["prediction"]
