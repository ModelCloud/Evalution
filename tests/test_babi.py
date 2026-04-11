# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

# Keep shared test fixtures and expectations explicit at module scope.
babi_module = importlib.import_module("evalution.benchmarks.babi")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self, responses: list[str]) -> None:
        """Initialize this object."""
        self.responses = responses
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        """Generate generate."""
        assert batch_size == 1
        self.requests.extend(requests)
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]

    def close(self) -> None:
        """Release the resources owned by this object."""
        return None


def test_babi_scores_generated_exact_match(monkeypatch) -> None:
    """Verify babi scores generated exact match. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "passage": "John travelled to the hallway.\nMary journeyed to the bathroom.\n",
                "question": "Where is John?",
                "answer": "hallway",
                "task": 1,
            }
        ]
    )
    monkeypatch.setattr(babi_module, "load_dataset", lambda *args, **kwargs: dataset)

    session = FakeSession([" hallway"])
    result = evalution.benchmarks.babi(max_rows=1, batch_size=4).evaluate(session)

    assert result.name == "babi"
    assert result.metrics == {"em": 1.0}
    assert result.metadata["dataset_path"] == "Muennighoff/babi"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "test"
    assert result.metadata["scoring_mode"] == "generated_exact_match"
    assert result.metadata["primary_metric"] == "em"
    assert len(result.samples) == 1

    request = session.requests[0]
    assert request.prompt == (
        "Passage: John travelled to the hallway.\n"
        "Mary journeyed to the bathroom.\n"
        "Question: Where is John?\n"
        "Answer:"
    )
    assert request.stop == ["\n", "Passage:"]
    assert request.max_new_tokens == 16

    sample = result.samples[0]
    assert sample.prompt == request.prompt
    assert sample.target == " hallway"
    assert sample.prediction == " hallway"
    assert sample.extracted == {
        "prediction-stripped": "hallway",
        "target-stripped": "hallway",
    }
    assert sample.metadata == {"task": 1}


def test_babi_marks_mismatched_generation_incorrect(monkeypatch) -> None:
    """Verify babi marks mismatched generation incorrect."""
    dataset = Dataset.from_list(
        [
            {
                "passage": "John travelled to the hallway.\nMary journeyed to the bathroom.\n",
                "question": "Where is John?",
                "answer": "hallway",
                "task": 1,
            }
        ]
    )
    monkeypatch.setattr(babi_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.babi(max_rows=1).evaluate(FakeSession([" bathroom"]))

    assert result.metrics == {"em": 0.0}
