# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

mutual_module = importlib.import_module("evalution.benchmarks.mutual")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "Dialogue: f : it's so cold today would you mind my closing the window ? m : of course not .\n"
            "Reply options:\n"
            "A. i will ask others to mend the window .\n"
            "B. it's so hot here so i will leave the windows open .\n"
            "C. thank you for closing the window for me .\n"
            "D. thank you ! i will shut the window now .\n"
            "Answer:"
        )
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=9),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=9),
            LoglikelihoodOutput(logprob=-0.8, is_greedy=False, token_count=9),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=9),
        ]


def test_mutual_scores_dialogue_response_selection(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "answers": "D",
                "options": [
                    "i will ask others to mend the window .",
                    "it's so hot here so i will leave the windows open .",
                    "thank you for closing the window for me .",
                    "thank you ! i will shut the window now .",
                ],
                "article": "f : it's so cold today would you mind my closing the window ? m : of course not .",
                "id": "train_5283",
            }
        ]
    )
    monkeypatch.setattr(mutual_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.mutual(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "mutual"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "tasksource/mutual"
    assert result.metadata["split"] == "validation"
    sample = result.samples[0]
    assert sample.target == "thank you ! i will shut the window now ."
    assert sample.prediction == "thank you ! i will shut the window now ."
    assert sample.metadata["dialogue_id"] == "train_5283"
