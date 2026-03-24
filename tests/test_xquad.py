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

xquad_module = importlib.import_module("evalution.benchmarks.xquad")


class FakeSession:
    def generate_continuous(self, requests, *, batch_size=None):
        assert batch_size == 4
        request_items = list(requests)
        assert len(request_items) == 2
        assert request_items[0][1].prompt == (
            "Context: The Panthers defense gave up just 308 points.\n\n"
            "Question: How many points did the Panthers defense surrender?\n\n"
            "Answer:"
        )
        predictions = ["308", "blue"]
        for (item_id, request), prediction in zip(request_items, predictions, strict=True):
            yield item_id, GenerationOutput(prompt=request.prompt, text=prediction)


def test_xquad_scores_extractable_answers(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "sample-1",
                "context": "The Panthers defense gave up just 308 points.",
                "question": "How many points did the Panthers defense surrender?",
                "answers": {"text": ["308"], "answer_start": [34]},
            },
            {
                "id": "sample-2",
                "context": "The sky is blue on a clear day.",
                "question": "What color is the sky on a clear day?",
                "answers": {"text": ["blue", "the sky is blue"], "answer_start": [11, 0]},
            },
        ]
    )
    monkeypatch.setattr(xquad_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.xquad(language="en", max_rows=2, batch_size=4).evaluate(FakeSession())

    assert result.name == "xquad_en"
    assert result.metrics == {"em": 1.0, "f1": 1.0}
    assert result.metadata["dataset_path"] == "google/xquad"
    assert result.metadata["dataset_name"] == "xquad.en"
    assert result.metadata["split"] == "validation"

    first_sample = result.samples[0]
    assert first_sample.target == "308"
    assert first_sample.prediction == "308"
    assert first_sample.metadata["language"] == "en"


def test_xquad_prompt_helper_formats_qa_prompt() -> None:
    assert (
        xquad_module._xquad_prompt(context="Context text", question="Question text")
        == "Context: Context text\n\nQuestion: Question text\n\nAnswer:"
    )


def test_xquad_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unsupported xquad language"):
        evalution.benchmarks.xquad(language="zzz")


def test_xquad_rejects_dataset_name_mismatch() -> None:
    with pytest.raises(ValueError, match="dataset_name must match"):
        evalution.benchmarks.xquad(language="en", dataset_name="xquad.fr")
