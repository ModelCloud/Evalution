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
from evalution.scorers.qa_text import best_qa_scores

# Keep shared test fixtures and expectations explicit at module scope.
nq_open_module = importlib.import_module("evalution.benchmarks.nq_open")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self, outputs: list[str]) -> None:
        """Initialize this object."""
        self.outputs = outputs
        self.offset = 0

    def generate(self, requests, *, batch_size=None):
        """Generate generate."""
        assert batch_size in {None, 1}
        request_list = list(requests)
        batch_outputs = self.outputs[self.offset : self.offset + len(request_list)]
        self.offset += len(request_list)
        return [
            GenerationOutput(prompt=request.prompt, text=output)
            for request, output in zip(request_list, batch_outputs, strict=True)
        ]


def test_nq_open_scores_best_alias(monkeypatch) -> None:
    """Verify nq open scores best alias. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "question": "when was the last time anyone was on the moon",
                "answer": ["14 December 1972 UTC", "December 1972"],
            }
        ]
    )
    monkeypatch.setattr(nq_open_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.nq_open(max_rows=1).evaluate(FakeSession(["December 1972"]))

    assert result.name == "nq_open"
    assert result.metadata == {
        "dataset_path": "nq_open",
        "dataset_name": "nq_open",
        "split": "validation",
        "stream": False,
        "order": "native",
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_qa_exact_match_f1",
        "primary_metric": "f1",
    }
    assert result.metrics == {"em": 1.0, "f1": 1.0}

    sample = result.samples[0]
    assert sample.extracted["best_answer"] == "December 1972"
    assert sample.metadata["question"] == "when was the last time anyone was on the moon"
    assert sample.metadata["answer_aliases"] == ["14 December 1972 UTC", "December 1972"]


def test_nq_open_qa_text_scorer_uses_best_alias() -> None:
    """Verify nq open QA text scorer uses best alias. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    exact, f1_score, best_index = best_qa_scores(
        "December 1972",
        ["14 December 1972 UTC", "December 1972"],
    )
    assert exact == 1.0
    assert f1_score == 1.0
    assert best_index == 1
