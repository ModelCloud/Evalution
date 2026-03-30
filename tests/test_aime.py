# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

aime_module = importlib.import_module("evalution.benchmarks.aime")


class FakeGenerationSession:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        assert batch_size in {1, 3}
        self.requests.extend(requests)
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]


def test_aime_scores_boxed_math_exact_match(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "ID": "1983-1",
                "Year": 1983,
                "Problem Number": 1,
                "Question": "Find the value of x.",
                "Answer": "42",
                "Part": None,
            }
        ]
    )
    monkeypatch.setattr(aime_module, "load_dataset", lambda *args, **kwargs: dataset)

    session = FakeGenerationSession(["Work... \\boxed{42}"])
    result = evalution.benchmarks.aime(max_rows=1, batch_size=3).evaluate(session)

    assert result.name == "aime"
    assert result.metrics == {"em": 1.0}
    assert result.metadata == {
        "dataset_path": "gneubig/aime-1983-2024",
        "dataset_name": None,
        "split": "train",
        "order": "native",
        "stream": False,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_math_exact_match",
        "primary_metric": "em",
    }
    assert session.requests[0].prompt == "Question: Find the value of x.\nAnswer:"
    assert session.requests[0].stop == ["Question:", "</s>", "<|im_end|>", "<|eot_id|>"]

    sample = result.samples[0]
    assert sample.prompt == "Question: Find the value of x.\nAnswer:"
    assert sample.target == "42"
    assert sample.prediction == "Work... \\boxed{42}"
    assert sample.extracted == {
        "prediction-stripped": "Work... \\boxed{42}",
        "answer-extract": "42",
        "prediction-normalized": "42",
        "target-normalized": "42",
    }
    assert sample.scores == {"em": 1.0}
    assert sample.metadata == {
        "problem_id": "1983-1",
        "year": 1983,
        "problem_number": 1,
        "part": None,
    }


def test_aime24_uses_problem_field_and_stringifies_integer_answers(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "ID": "2024-I-1",
                "Problem": "Compute the final value.",
                "Solution": "Reasoning omitted.",
                "Answer": 33,
            }
        ]
    )
    monkeypatch.setattr(aime_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.aime24(max_rows=1, batch_size=3).evaluate(
        FakeGenerationSession(["Final answer: $33$"])
    )

    assert result.name == "aime24"
    assert result.metrics == {"em": 1.0}
    sample = result.samples[0]
    assert sample.prompt == "Question: Compute the final value.\nAnswer:"
    assert sample.target == "33"
    assert sample.metadata == {
        "problem_id": "2024-I-1",
        "solution": "Reasoning omitted.",
    }


def test_aime25_factory_uses_test_split_and_lowercase_fields() -> None:
    suite = evalution.benchmarks.aime25()

    assert suite.dataset_path == "math-ai/aime25"
    assert suite.split == "test"
    assert suite.question_field == "problem"
    assert suite.answer_field == "answer"
    assert suite.id_field == "id"
