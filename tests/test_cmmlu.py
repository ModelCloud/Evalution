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

cmmlu_module = importlib.import_module("evalution.benchmarks.cmmlu")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "以下是单项选择题，请直接给出正确答案的选项。\n\n"
            "二加二等于几？\n"
            "A. 1\n"
            "B. 2\n"
            "C. 3\n"
            "D. 4\n"
            "答案：D\n\n"
            "中国四大发明之一是？\n"
            "A. 指南针\n"
            "B. 火箭\n"
            "C. 电视\n"
            "D. 电脑\n"
            "答案："
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.6, is_greedy=False, token_count=1),
        ]


def test_cmmlu_scores_subject_with_dev_fewshots(monkeypatch) -> None:
    test = Dataset.from_list(
        [
            {
                "Question": "中国四大发明之一是？",
                "A": "指南针",
                "B": "火箭",
                "C": "电视",
                "D": "电脑",
                "Answer": "A",
            }
        ]
    )
    dev = Dataset.from_list(
        [
            {
                "Question": "二加二等于几？",
                "A": "1",
                "B": "2",
                "C": "3",
                "D": "4",
                "Answer": "D",
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

    monkeypatch.setattr(cmmlu_module, "_load_cmmlu_dataset", fake_load_dataset)

    result = evalution.benchmarks.cmmlu_agronomy(
        num_fewshot=1,
        max_rows=1,
        batch_size=8,
    ).evaluate(FakeSession())

    assert result.name == "cmmlu_agronomy"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata == {
        "dataset_path": "haonan-li/cmmlu",
        "dataset_name": "agronomy",
        "split": "test",
        "fewshot_split": "dev",
        "num_fewshot": 1,
        "stream": False,
        "scoring_mode": "multiple_choice_loglikelihood",
    }
    sample = result.samples[0]
    assert sample.target == "A"
    assert sample.prediction == "A"
    assert sample.metadata["subset"] == "agronomy"
    assert sample.metadata["question"] == "中国四大发明之一是？"
    assert sample.metadata["choice_texts"] == ["指南针", "火箭", "电视", "电脑"]
    assert sample.metadata["fewshot_count"] == 1


def test_cmmlu_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="unsupported CMMLU subset"):
        evalution.benchmarks.cmmlu(subset="unknown_subject")
