# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

mastermind_module = importlib.import_module("evalution.benchmarks.mastermind")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 4
        assert requests[0].context.endswith("\n\nThe secret code is:")
        assert [request.continuation for request in requests] == [
            " orange, orange",
            " green, white",
            " orange, brown",
            " white, orange",
        ]
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=2),
            LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=2),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=2),
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=2),
        ]


@pytest.mark.parametrize(
    ("factory_name", "expected_name", "expected_path", "expected_shape", "expected_difficulty"),
    [
        ("mastermind_24_easy", "mastermind_24_easy", "flair/mastermind_24_mcq_random", "24", "easy"),
        ("mastermind_24_hard", "mastermind_24_hard", "flair/mastermind_24_mcq_close", "24", "hard"),
        ("mastermind_35_easy", "mastermind_35_easy", "flair/mastermind_35_mcq_random", "35", "easy"),
        ("mastermind_35_hard", "mastermind_35_hard", "flair/mastermind_35_mcq_close", "35", "hard"),
        ("mastermind_46_easy", "mastermind_46_easy", "flair/mastermind_46_mcq_random", "46", "easy"),
        ("mastermind_46_hard", "mastermind_46_hard", "flair/mastermind_46_mcq_close", "46", "hard"),
    ],
)
def test_mastermind_variant_scores_multiple_choice(
    monkeypatch,
    factory_name,
    expected_name,
    expected_path,
    expected_shape,
    expected_difficulty,
) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": 10153,
                "instruction": (
                    "Your goal is to find the secret two-color code. "
                    "The following colors are possible: brown, green, white, orange."
                ),
                "options": {
                    "label": ["A", "B", "C", "D"],
                    "text": ["orange, orange", "green, white", "orange, brown", "white, orange"],
                },
                "answerKey": "C",
            }
        ]
    )
    monkeypatch.setattr(mastermind_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = getattr(evalution.benchmarks, factory_name)(max_rows=1, batch_size=4).evaluate(FakeSession())

    assert result.name == expected_name
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == expected_path
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "test"
    assert result.metadata["variant"] == expected_name
    assert result.metadata["code_shape"] == expected_shape
    assert result.metadata["difficulty"] == expected_difficulty

    sample = result.samples[0]
    assert sample.target == "orange, brown"
    assert sample.prediction == "orange, brown"
    assert sample.metadata["variant"] == expected_name
    assert sample.metadata["code_shape"] == expected_shape
    assert sample.metadata["difficulty"] == expected_difficulty
    assert sample.metadata["option_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_texts"] == [
        "orange, orange",
        "green, white",
        "orange, brown",
        "white, orange",
    ]


def test_mastermind_prompt_helper_formats_instruction() -> None:
    assert mastermind_module._mastermind_prompt("Solve it") == "Solve it\n\nThe secret code is:"


def test_mastermind_rejects_invalid_variant_and_dataset_settings() -> None:
    with pytest.raises(ValueError, match="unsupported mastermind variant"):
        evalution.benchmarks.mastermind(variant="bad")
    with pytest.raises(ValueError, match="mastermind dataset_path must match the configured variant"):
        mastermind_module.Mastermind(variant="mastermind_24_easy", dataset_path="bad")
    with pytest.raises(ValueError, match="mastermind does not use a dataset_name"):
        mastermind_module.Mastermind(variant="mastermind_24_easy", dataset_name="foo")
