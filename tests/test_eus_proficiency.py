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

eus_proficiency_module = importlib.import_module("evalution.benchmarks.eus_proficiency")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "Galdera: Bi seme-alaba ditu, ..... ederragoak.\n"
            "A: zenbat eta\n"
            "B: haiek baino\n"
            "C: nola edo hala\n"
            "D: zein baino zein\n"
            "Erantzuna:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
        ]


def test_eus_proficiency_scores_four_way_multiple_choice(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": 0,
                "question": "Bi seme-alaba ditu, ..... ederragoak.",
                "candidates": [
                    "zenbat eta",
                    "haiek baino",
                    "nola edo hala",
                    "zein baino zein",
                ],
                "answer": 3,
            }
        ]
    )
    monkeypatch.setattr(eus_proficiency_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.eus_proficiency(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "eus_proficiency"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "HiTZ/EusProficiency"
    assert result.metadata["dataset_name"] == "default"
    assert result.metadata["split"] == "test"
    sample = result.samples[0]
    assert sample.target == "D"
    assert sample.prediction == "D"
    assert sample.metadata["id"] == 0
    assert sample.metadata["question"] == "Bi seme-alaba ditu, ..... ederragoak."
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["raw_choices"] == [
        "zenbat eta",
        "haiek baino",
        "nola edo hala",
        "zein baino zein",
    ]


def test_eus_proficiency_prompt_matches_upstream_shape() -> None:
    doc = {
        "question": "Nor da?",
        "candidates": ["Lehen aukera", "Bigarren aukera", "Hirugarren aukera", "Laugarren aukera"],
    }
    assert eus_proficiency_module._eus_proficiency_prompt(doc) == (
        "Galdera: Nor da?\n"
        "A: Lehen aukera\n"
        "B: Bigarren aukera\n"
        "C: Hirugarren aukera\n"
        "D: Laugarren aukera\n"
        "Erantzuna:"
    )


def test_eus_proficiency_rejects_non_four_way_rows() -> None:
    with pytest.raises(ValueError, match="exactly four candidates"):
        eus_proficiency_module._eus_proficiency_prompt(
            {"question": "Nor da?", "candidates": ["A", "B", "C"]}
        )
