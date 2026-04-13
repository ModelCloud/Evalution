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

# Keep shared test fixtures and expectations explicit at module scope.
eus_trivia_module = importlib.import_module("evalution.benchmarks.eus_trivia")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "Galdera: Nola bota behar dira honakoak ontzi horietara?\n"
            "A: Apurturik\n"
            "B: Denak lotuta\n"
            "C: Tapoia kendu gabe\n"
            "D: Tapoirik gabe\n"
            "Erantzuna:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
        ]


def test_eus_trivia_scores_label_choice_benchmark(monkeypatch) -> None:
    """Verify eus trivia scores label choice benchmark. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "id": 0,
                "category": "Giza eta Natur Zientziak",
                "difficulty": "zaila",
                "question": "Nola bota behar dira honakoak ontzi horietara?",
                "candidates": [
                    "Apurturik",
                    "Denak lotuta",
                    "Tapoia kendu gabe",
                    "Tapoirik gabe",
                ],
                "answer": 3,
            }
        ]
    )
    monkeypatch.setattr(eus_trivia_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.eus_trivia(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "eus_trivia"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "HiTZ/EusTrivia"
    assert result.metadata["dataset_name"] == "default"
    assert result.metadata["split"] == "test"
    sample = result.samples[0]
    assert sample.target == "D"
    assert sample.prediction == "D"
    assert sample.metadata["id"] == 0
    assert sample.metadata["category"] == "Giza eta Natur Zientziak"
    assert sample.metadata["difficulty"] == "zaila"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["raw_choices"] == [
        "Apurturik",
        "Denak lotuta",
        "Tapoia kendu gabe",
        "Tapoirik gabe",
    ]


def test_eus_trivia_prompt_matches_upstream_shape() -> None:
    """Verify eus trivia prompt matches upstream shape."""
    doc = {
        "question": "Nor da?",
        "candidates": [
            "Lehen aukera",
            "Bigarren aukera",
            "Hirugarren aukera",
            "Laugarren aukera",
        ],
    }
    assert eus_trivia_module._eus_trivia_prompt(doc) == (
        "Galdera: Nor da?\n"
        "A: Lehen aukera\n"
        "B: Bigarren aukera\n"
        "C: Hirugarren aukera\n"
        "D: Laugarren aukera\n"
        "Erantzuna:"
    )


def test_eus_trivia_rejects_invalid_candidate_count() -> None:
    """Verify eus trivia rejects invalid candidate count."""
    with pytest.raises(ValueError, match="at least two candidates"):
        eus_trivia_module._eus_trivia_prompt({"question": "Nor da?", "candidates": ["Bakarra"]})

    with pytest.raises(ValueError, match="at most four candidates"):
        eus_trivia_module._eus_trivia_prompt(
            {
                "question": "Nor da?",
                "candidates": ["A", "B", "C", "D", "E"],
            }
        )
