# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

race_module = importlib.import_module("evalution.benchmarks.race")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 4
        assert requests[0].context == (
            "Article: Liz kept a notebook of bird sightings.\n\n"
            "Question: Why did Liz bring binoculars?\n"
            "Answer: To see distant birds clearly.\n"
            "What did Liz record after lunch?"
        )
        assert requests[0].continuation == " A robin near the pond."
        assert requests[3].continuation == " A broken bicycle chain."
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=6),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=5),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=5),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=5),
        ]


def test_race_scores_flattened_question_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "article_index": 0,
                "problem_index": 1,
                "article": "Liz kept a notebook of bird sightings.",
                "question": "What did Liz record after lunch?",
                "answer": "A",
                "options": [
                    "A robin near the pond.",
                    "A train schedule change.",
                    "A store closing early.",
                    "A broken bicycle chain.",
                ],
                "previous_problems": [
                    {
                        "question": "Why did Liz bring binoculars?",
                        "answer": "B",
                        "options": [
                            "To fix a bicycle.",
                            "To see distant birds clearly.",
                            "To read a map at night.",
                            "To measure rainfall.",
                        ],
                    }
                ],
            }
        ]
    )
    monkeypatch.setattr(race_module, "_load_race_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.race(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "race"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "EleutherAI/race"
    assert result.metadata["dataset_name"] == "high"
    assert result.metadata["split"] == "test"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.target == "A robin near the pond."
    assert sample.prediction == "A robin near the pond."
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["previous_problem_count"] == 1


def test_race_loader_flattens_article_level_rows(monkeypatch) -> None:
    article_dataset = Dataset.from_list(
        [
            {
                "article": "Liz kept a notebook of bird sightings.",
                "problems": str(
                    [
                        {
                            "question": "Why did Liz bring binoculars?",
                            "answer": "B",
                            "options": [
                                "To fix a bicycle.",
                                "To see distant birds clearly.",
                                "To read a map at night.",
                                "To measure rainfall.",
                            ],
                        },
                        {
                            "question": "What did Liz record after lunch?",
                            "answer": "A",
                            "options": [
                                "A robin near the pond.",
                                "A train schedule change.",
                                "A store closing early.",
                                "A broken bicycle chain.",
                            ],
                        },
                    ]
                ),
            }
        ]
    )
    monkeypatch.setattr(race_module, "load_dataset", lambda *args, **kwargs: article_dataset)

    dataset = race_module._load_race_dataset(
        "EleutherAI/race",
        "high",
        split="test",
    )

    assert len(dataset) == 2
    assert dataset[0]["problem_index"] == 0
    assert dataset[1]["problem_index"] == 1
    assert dataset[1]["previous_problems"][0]["answer"] == "B"


def test_race_prompt_matches_upstream_cumulative_format() -> None:
    prompt = race_module._race_prompt(
        "Liz kept a notebook of bird sightings.",
        [
            {
                "question": "Why did Liz bring binoculars?",
                "answer": "B",
                "options": [
                    "To fix a bicycle.",
                    "To see distant birds clearly.",
                    "To read a map at night.",
                    "To measure rainfall.",
                ],
            }
        ],
        "What did Liz record after lunch?",
    )

    assert prompt == (
        "Article: Liz kept a notebook of bird sightings.\n\n"
        "Question: Why did Liz bring binoculars?\n"
        "Answer: To see distant birds clearly.\n"
        "What did Liz record after lunch?"
    )
