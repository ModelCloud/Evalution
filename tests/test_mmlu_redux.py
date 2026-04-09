# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

mmlu_redux_module = importlib.import_module("evalution.benchmarks.mmlu_redux")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "2 + 2 = ?\n"
            "A. 1\n"
            "B. 4\n"
            "C. 5\n"
            "D. 9\n"
            "Please respond with the correct letter (A, B, C or D) without any additional comments, "
            "only the correct letter:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=1),
        ]


def test_mmlu_redux_scores_subject_shards_as_letter_choices(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "2 + 2 = ?",
                "choices": ["1", "4", "5", "9"],
                "answer": "B",
                "subject": "abstract_algebra",
            }
        ]
    )
    monkeypatch.setattr(
        mmlu_redux_module,
        "_load_mmlu_redux_subject_dataset",
        lambda *args, **kwargs: dataset,
    )

    result = evalution.benchmarks.mmlu_redux(
        subsets="stem.abstract_algebra",
        max_rows=1,
        batch_size=8,
    ).evaluate(FakeSession())

    assert result.name == "mmlu_redux_stem_abstract_algebra"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata == {
        "dataset_path": "fxmarty/mmlu-redux-2.0-ok",
        "dataset_name": None,
        "split": "test",
        "order": "native",
        "stream": False,
        "scoring_mode": "multiple_choice_loglikelihood",
        "subsets": ["stem.abstract_algebra"],
        "subset_paths": [["stem", "abstract_algebra"]],
        "subset_kinds": ["leaf"],
        "selection_mode": "single",
    }
    sample = result.samples[0]
    assert sample.target == "B"
    assert sample.prediction == "B"
    assert sample.metadata["subject"] == "abstract_algebra"
    assert sample.metadata["subset"] == "stem.abstract_algebra"
    assert sample.metadata["subset_path"] == ["stem", "abstract_algebra"]
    assert sample.metadata["choice_texts"] == ["1", "4", "5", "9"]


def test_mmlu_redux_exposes_resolved_all_subjects() -> None:
    assert "abstract_algebra" in mmlu_redux_module._MMLU_REDUX_ALL_SUBJECTS
