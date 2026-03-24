# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

arabicmmlu_module = importlib.import_module("evalution.benchmarks.arabicmmlu")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "This is a Islamic Studies question. Select the correct answer!\n\n"
            "Question: كم عدد سور القرآن الكريم؟\n"
            "A. 111\n"
            "B. 112\n"
            "C. 113\n"
            "D. 114\n\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
        ]


def test_arabicmmlu_scores_multiple_choice_rows(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "ID": 0,
                "Source": "source",
                "Country": None,
                "Group": "Humanities",
                "Subject": "Islamic Studies",
                "Level": None,
                "Question": "كم عدد سور القرآن الكريم؟",
                "Context": None,
                "Answer Key": "D",
                "Option 1": "111",
                "Option 2": "112",
                "Option 3": "113",
                "Option 4": "114",
                "Option 5": None,
                "is_few_shot": 0,
            }
        ]
    )
    monkeypatch.setattr(arabicmmlu_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.arabicmmlu(subset="Islamic Studies", max_rows=1, batch_size=8).evaluate(
        FakeSession()
    )

    assert result.name == "arabicmmlu_islamic_studies"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "MBZUAI/ArabicMMLU"
    assert result.metadata["dataset_name"] == "Islamic Studies"
    assert result.metadata["split"] == "test"
    sample = result.samples[0]
    assert sample.target == "D"
    assert sample.prediction == "D"
    assert sample.metadata["subset"] == "Islamic Studies"
    assert sample.metadata["subject"] == "Islamic Studies"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["raw_choices"] == ["111", "112", "113", "114"]


def test_arabicmmlu_slugify_subset_name() -> None:
    assert arabicmmlu_module._slugify_subset_name("Islamic Studies") == "islamic_studies"
    assert (
        arabicmmlu_module._slugify_subset_name("Computer Science (High School)")
        == "computer_science_high_school"
    )


def test_arabicmmlu_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="unsupported arabicmmlu subset"):
        evalution.benchmarks.arabicmmlu(subset="unknown")
