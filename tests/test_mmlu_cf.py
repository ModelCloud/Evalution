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

mmlu_cf_module = importlib.import_module("evalution.benchmarks.mmlu_cf")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "There is a single choice question (with answers). Answer the question by replying A, B, C or D.\n\n"
            "Cells use DNA primarily to\n"
            "A. store genetic information\n"
            "B. generate electricity\n"
            "C. dissolve minerals\n"
            "D. stop diffusion\n"
            "Answer: A\n\n"
            "Water freezes at what temperature on the Celsius scale?\n"
            "A. 50\n"
            "B. 0\n"
            "C. 25\n"
            "D. 100\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
        ]


def test_mmlu_cf_scores_subject_with_dev_fewshots(monkeypatch) -> None:
    validation = Dataset.from_list(
        [
            {
                "Question": "Water freezes at what temperature on the Celsius scale?",
                "A": "50",
                "B": "0",
                "C": "25",
                "D": "100",
                "Answer": "B",
            }
        ]
    )
    dev = Dataset.from_list(
        [
            {
                "Question": "Cells use DNA primarily to",
                "A": "store genetic information",
                "B": "generate electricity",
                "C": "dissolve minerals",
                "D": "stop diffusion",
                "Answer": "A",
            }
        ]
    )

    def fake_load_dataset(path, name=None, *, split=None, **kwargs):
        del path, name, kwargs
        if split == "val":
            return validation
        if split == "dev":
            return dev
        raise AssertionError(f"unexpected split: {split}")

    monkeypatch.setattr(mmlu_cf_module, "_load_mmlu_cf_dataset", fake_load_dataset)

    result = evalution.benchmarks.mmlu_cf_biology(
        num_fewshot=1,
        max_rows=1,
        batch_size=8,
    ).evaluate(FakeSession())

    assert result.name == "mmlu_cf_biology"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata == {
        "dataset_path": "microsoft/MMLU-CF",
        "dataset_name": "biology",
        "split": "val",
        "fewshot_split": "dev",
        "num_fewshot": 1,
        "stream": False,
        "scoring_mode": "multiple_choice_loglikelihood",
    }
    sample = result.samples[0]
    assert sample.target == "B"
    assert sample.prediction == "B"
    assert sample.metadata["subject"] == "biology"
    assert sample.metadata["question"] == "Water freezes at what temperature on the Celsius scale?"
    assert sample.metadata["choice_texts"] == ["50", "0", "25", "100"]
    assert sample.metadata["fewshot_count"] == 1


def test_mmlu_cf_rejects_unknown_subject() -> None:
    with pytest.raises(ValueError, match="unsupported MMLU-CF subject"):
        evalution.benchmarks.mmlu_cf(subject="unknown_subject")
