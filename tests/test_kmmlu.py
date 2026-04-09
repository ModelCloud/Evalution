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

kmmlu_module = importlib.import_module("evalution.benchmarks.kmmlu")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "복식부기의 차변과 대변은 무엇을 기록하는가?\n"
            "A. 자산과 부채\n"
            "B. 거래의 양면\n"
            "C. 세금과 이익\n"
            "D. 자본과 현금\n"
            "정답：B\n\n"
            "회계의 기본 목적은 무엇인가?\n"
            "A. 거래 정보를 체계적으로 보고하는 것\n"
            "B. 세율을 올리는 것\n"
            "C. 재고를 숨기는 것\n"
            "D. 차입을 늘리는 것\n"
            "정답："
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
        ]


def test_kmmlu_scores_subject_with_dev_fewshots(monkeypatch) -> None:
    test = Dataset.from_list(
        [
            {
                "question": "회계의 기본 목적은 무엇인가?",
                "answer": "1",
                "A": "거래 정보를 체계적으로 보고하는 것",
                "B": "세율을 올리는 것",
                "C": "재고를 숨기는 것",
                "D": "차입을 늘리는 것",
                "Category": "accounting",
                "Human Accuracy": "0.83",
            }
        ]
    )
    dev = Dataset.from_list(
        [
            {
                "question": "복식부기의 차변과 대변은 무엇을 기록하는가?",
                "answer": "2",
                "A": "자산과 부채",
                "B": "거래의 양면",
                "C": "세금과 이익",
                "D": "자본과 현금",
                "Category": "accounting",
                "Human Accuracy": "0.91",
            }
        ]
    )

    def fake_load_dataset(path, name=None, *, split=None, **kwargs):
        del path, name, kwargs
        if split == "test":
            return test
        if split == "dev":
            return dev
        raise AssertionError(f"unexpected split: {split}")

    monkeypatch.setattr(kmmlu_module, "_load_kmmlu_dataset", fake_load_dataset)

    result = evalution.benchmarks.kmmlu_accounting(
        num_fewshot=1,
        max_rows=1,
        batch_size=8,
    ).evaluate(FakeSession())

    assert result.name == "kmmlu_accounting"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata == {
        "dataset_path": "HAERAE-HUB/KMMLU",
        "dataset_name": "Accounting",
        "split": "test",
        "fewshot_split": "dev",
        "num_fewshot": 1,
        "stream": False,
        "scoring_mode": "multiple_choice_loglikelihood",
    }
    sample = result.samples[0]
    assert sample.target == "A"
    assert sample.prediction == "A"
    assert sample.metadata["subset"] == "accounting"
    assert sample.metadata["category"] == "accounting"
    assert sample.metadata["question"] == "회계의 기본 목적은 무엇인가?"
    assert sample.metadata["human_accuracy"] == pytest.approx(0.83)


def test_kmmlu_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="unsupported KMMLU subset"):
        evalution.benchmarks.kmmlu(subset="unknown_subject")
