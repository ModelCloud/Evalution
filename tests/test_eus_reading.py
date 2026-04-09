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

eus_reading_module = importlib.import_module("evalution.benchmarks.eus_reading")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "Pasartea: Ura baliabide urria da.\n\n"
            "Galdera: Zer dio testuak urari buruz?\n"
            "A: Mugagabea dela\n"
            "B: Baliabide urria dela\n"
            "C: Ez dela beharrezkoa\n"
            "D: Kutsadurarik ez duela\n"
            "Erantzuna:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
        ]


def test_eus_reading_scores_context_question_multiple_choice(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": 0,
                "context": "Ura baliabide urria da.",
                "question": "Zer dio testuak urari buruz?",
                "candidates": [
                    "Mugagabea dela",
                    "Baliabide urria dela",
                    "Ez dela beharrezkoa",
                    "Kutsadurarik ez duela",
                ],
                "answer": 1,
            }
        ]
    )
    monkeypatch.setattr(eus_reading_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.eus_reading(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "eus_reading"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "HiTZ/EusReading"
    assert result.metadata["dataset_name"] == "default"
    assert result.metadata["split"] == "test"
    sample = result.samples[0]
    assert sample.target == "B"
    assert sample.prediction == "B"
    assert sample.metadata["id"] == 0
    assert sample.metadata["context"] == "Ura baliabide urria da."
    assert sample.metadata["question"] == "Zer dio testuak urari buruz?"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["raw_choices"] == [
        "Mugagabea dela",
        "Baliabide urria dela",
        "Ez dela beharrezkoa",
        "Kutsadurarik ez duela",
    ]


def test_eus_reading_prompt_matches_upstream_shape() -> None:
    doc = {
        "context": "Pasarte laburra.",
        "question": "Nor da?",
        "candidates": ["Lehen aukera", "Bigarren aukera", "Hirugarren aukera"],
    }
    assert eus_reading_module._eus_reading_prompt(doc) == (
        "Pasartea: Pasarte laburra.\n\n"
        "Galdera: Nor da?\n"
        "A: Lehen aukera\n"
        "B: Bigarren aukera\n"
        "C: Hirugarren aukera\n"
        "Erantzuna:"
    )


def test_eus_reading_rejects_invalid_candidate_count() -> None:
    with pytest.raises(ValueError, match="at least two candidates"):
        eus_reading_module._eus_reading_prompt(
            {"context": "x", "question": "Nor da?", "candidates": ["Bakarra"]}
        )

    with pytest.raises(ValueError, match="at most four candidates"):
        eus_reading_module._eus_reading_prompt(
            {
                "context": "x",
                "question": "Nor da?",
                "candidates": ["A", "B", "C", "D", "E"],
            }
        )
