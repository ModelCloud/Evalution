# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput, LoglikelihoodOutput

gsm8k_module = importlib.import_module("evalution.benchmarks.gsm8k")
sciq_module = importlib.import_module("evalution.benchmarks.sciq")


class RecordingGenerationSession:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, requests, *, batch_size=None):
        del batch_size
        self.prompts.extend(request.prompt or "" for request in requests)
        return [
            GenerationOutput(prompt=request.prompt or "", text=response)
            for request, response in zip(requests, self.responses, strict=True)
        ]

    def close(self) -> None:
        return None


class RecordingLoglikelihoodSession:
    def __init__(self) -> None:
        self.contexts: list[str] = []

    def loglikelihood(self, requests, *, batch_size=None):
        del batch_size
        self.contexts.extend(request.context for request in requests)
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1)
            for _request in requests
        ]


def test_base_suite_length_order_uses_generation_prompt_length(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "Short?",
                "answer": "1\n#### 1",
            },
            {
                "question": "This is a much longer arithmetic word problem prompt used to force ordering.",
                "answer": "2\n#### 2",
            },
        ]
    )
    monkeypatch.setattr(gsm8k_module, "load_dataset", lambda *args, **kwargs: dataset)

    session = RecordingGenerationSession(["The answer is 1.", "The answer is 2."])
    result = evalution.benchmarks.gsm8k(
        variant="base",
        max_rows=2,
        batch_size=2,
        order="length|desc",
    ).evaluate(session)

    assert session.prompts[0].startswith(
        "Question: This is a much longer arithmetic word problem prompt used to force ordering."
    )
    assert session.prompts[1].startswith("Question: Short?")
    assert result.metadata["order"] == "length|desc"


def test_multiple_choice_length_order_uses_prompt_and_choice_length(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "Short?",
                "distractor1": "A",
                "distractor2": "B",
                "distractor3": "C",
                "correct_answer": "D",
                "support": "Tiny support.",
            },
            {
                "question": "Which explanation is the longest one in this synthetic dataset?",
                "distractor1": "Short one",
                "distractor2": "Another short one",
                "distractor3": "A medium sized distractor",
                "correct_answer": "The longest answer option in this sample by a clear margin",
                "support": "This support passage is intentionally much longer than the first sample.",
            },
        ]
    )
    monkeypatch.setattr(sciq_module, "load_dataset", lambda *args, **kwargs: dataset)

    session = RecordingLoglikelihoodSession()
    result = evalution.benchmarks.sciq(
        max_rows=2,
        batch_size=8,
        order="length|asc",
    ).evaluate(session)

    assert session.contexts[0].startswith("Tiny support.\nQuestion: Short?\nAnswer:")
    assert session.contexts[4].startswith(
        "This support passage is intentionally much longer than the first sample.\n"
        "Question: Which explanation is the longest one in this synthetic dataset?\n"
        "Answer:"
    )
    assert result.metadata["order"] == "length|asc"


def test_shuffle_order_with_seed_is_deterministic(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "Question 0?",
                "answer": "0\n#### 0",
            },
            {
                "question": "Question 1?",
                "answer": "1\n#### 1",
            },
            {
                "question": "Question 2?",
                "answer": "2\n#### 2",
            },
        ]
    )
    monkeypatch.setattr(gsm8k_module, "load_dataset", lambda *args, **kwargs: dataset)

    left_session = RecordingGenerationSession(["0", "1", "2"])
    right_session = RecordingGenerationSession(["0", "1", "2"])
    left_result = evalution.benchmarks.gsm8k(
        variant="base",
        max_rows=3,
        batch_size=3,
        order="shuffle|7",
    ).evaluate(left_session)
    right_result = evalution.benchmarks.gsm8k(
        variant="base",
        max_rows=3,
        batch_size=3,
        order="shuffle|7",
    ).evaluate(right_session)

    assert left_session.prompts == right_session.prompts
    assert left_result.metadata["order"] == "shuffle|7"
    assert right_result.metadata["order"] == "shuffle|7"
