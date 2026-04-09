# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

mlqa_module = importlib.import_module("evalution.benchmarks.mlqa")


class FakeSession:
    def generate_continuous(self, requests, *, batch_size=None):
        assert batch_size == 4
        request_items = list(requests)
        assert len(request_items) == 1
        assert request_items[0][1].prompt == (
            "Context: The Panthers defense gave up just 308 points.\n\n"
            "Question: How many points did the Panthers defense surrender?\n\n"
            "Answer:"
        )
        for item_id, request in request_items:
            yield item_id, GenerationOutput(prompt=request.prompt, text="308")


def test_mlqa_scores_language_paired_extractable_answers(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "sample-1",
                "context": "The Panthers defense gave up just 308 points.",
                "question": "How many points did the Panthers defense surrender?",
                "answers": {"text": ["308"], "answer_start": [34]},
            }
        ]
    )
    monkeypatch.setattr(mlqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.mlqa_en_en(max_rows=1, batch_size=4).evaluate(FakeSession())

    assert result.name == "mlqa_en_en"
    assert result.metrics == {"em": 1.0, "f1": 1.0}
    assert result.metadata["dataset_path"] == "facebook/mlqa"
    assert result.metadata["dataset_name"] == "mlqa.en.en"
    assert result.metadata["split"] == "test"
    assert result.metadata["context_language"] == "en"
    assert result.metadata["question_language"] == "en"
    sample = result.samples[0]
    assert sample.target == "308"
    assert sample.prediction == "308"
    assert sample.metadata["context_language"] == "en"
    assert sample.metadata["question_language"] == "en"


def test_mlqa_normalization_and_dataset_name_validation() -> None:
    assert mlqa_module._normalize_mlqa_answer("The, cat!", "en") == "cat"

    with pytest.raises(ValueError, match="unsupported mlqa context language"):
        evalution.benchmarks.mlqa(context_language="xx", question_language="en")

    with pytest.raises(ValueError, match="dataset_name must match"):
        evalution.benchmarks.mlqa(
            context_language="en",
            question_language="es",
            dataset_name="mlqa.en.en",
        )
