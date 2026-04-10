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

longbench2_module = importlib.import_module("evalution.benchmarks.longbench2")


class MultipleChoiceSession:
    def __init__(self, expected_context: str, expected_continuations: list[str]) -> None:
        self.expected_context = expected_context
        self.expected_continuations = expected_continuations

    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        request_items = list(requests)
        assert [request.context for request in request_items] == [self.expected_context] * len(request_items)
        assert [request.continuation for request in request_items] == self.expected_continuations
        return [
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.5, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-4.0, is_greedy=False, token_count=1),
        ][: len(request_items)]


def test_longbench2_scores_label_ranked_rows(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "domain": "Single-Document QA",
                "difficulty": "hard",
                "length": "short",
                "question": "Which option is correct?",
                "choices": ["wrong one", "right one", "wrong two", "wrong three"],
                "answer": 1,
                "context": "A very long document body.",
            }
        ]
    )
    monkeypatch.setattr(longbench2_module, "load_dataset", lambda *args, **kwargs: dataset)

    prompt = (
        "Please read the following text and answer the question below.\n\n"
        "<text>\n"
        "A very long document body.\n"
        "</text>\n\n"
        "What is the correct answer to this question: Which option is correct?\n"
        "Choices:\n"
        "(A) wrong one\n"
        "(B) right one\n"
        "(C) wrong two\n"
        "(D) wrong three\n\n"
        "Answer:"
    )

    result = evalution.benchmarks.longbench2_academic_single(max_rows=1, batch_size=8).evaluate(
        MultipleChoiceSession(
            expected_context=prompt,
            expected_continuations=[" A", " B", " C", " D"],
        )
    )

    assert result.name == "longbench2_academic_single"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata == {
        "dataset_path": "recursal/longbench-v2",
        "dataset_name": "academic_single",
        "split": "train",
        "order": "native",
        "stream": False,
        "scoring_mode": "multiple_choice_loglikelihood",
    }
    sample = result.samples[0]
    assert sample.target == "B"
    assert sample.metadata["choice_texts"] == ["wrong one", "right one", "wrong two", "wrong three"]
    assert sample.metadata["domain"] == "Single-Document QA"


def test_longbench2_dispatcher_accepts_task_and_dataset_aliases() -> None:
    suite = evalution.benchmarks.longbench2(subset="academic_multi")
    assert suite.dataset_name == "academic_multi"
    assert suite.task_name() == "longbench2_academic_multi"

    task_suite = evalution.benchmarks.longbench2(subset="longbench2_translate")
    assert task_suite.dataset_name == "new_language_translation"
    assert task_suite.task_name() == "longbench2_translate"

    with pytest.raises(ValueError, match="unsupported longbench2 subset"):
        evalution.benchmarks.longbench2(subset="unknown")
