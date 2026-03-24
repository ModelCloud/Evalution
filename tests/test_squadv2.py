# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput
from evalution.scorers.qa_text import best_qa_scores, canonicalize_no_answer

squadv2_module = importlib.import_module("evalution.benchmarks.squadv2")


class FakeSession:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.offset = 0

    def generate(self, requests, *, batch_size=None):
        assert batch_size in {None, 1}
        request_list = list(requests)
        batch_outputs = self.outputs[self.offset : self.offset + len(request_list)]
        self.offset += len(request_list)
        return [
            GenerationOutput(prompt=request.prompt, text=output)
            for request, output in zip(request_list, batch_outputs, strict=True)
        ]


def test_squadv2_scores_best_alias_and_no_answer(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "1",
                "title": "Normans",
                "context": "Normandy is in France.",
                "question": "In what country is Normandy located?",
                "answers": {"text": ["France", "The France"]},
            },
            {
                "id": "2",
                "title": "Unknown",
                "context": "This passage does not contain the answer.",
                "question": "Who won the race?",
                "answers": {"text": []},
            },
        ]
    )
    monkeypatch.setattr(squadv2_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.squadv2(max_rows=2).evaluate(
        FakeSession([" The France.", ""])
    )

    assert result.name == "squadv2"
    assert result.metadata == {
        "dataset_path": "squad_v2",
        "dataset_name": "squad_v2",
        "split": "validation",
        "stream": False,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_qa_exact_match_f1",
        "primary_metric": "f1",
        "no_answer_token": "unanswerable",
    }
    assert result.metrics == {"em": 1.0, "f1": 1.0}

    first, second = result.samples
    assert first.scores == {"em": 1.0, "f1": 1.0}
    assert first.metadata["answer_texts"] == ["France", "The France"]
    assert first.metadata["has_answer"] is True
    assert second.extracted["prediction-normalized"] == "unanswerable"
    assert second.scores == {"em": 1.0, "f1": 1.0}
    assert second.metadata["has_answer"] is False


def test_squadv2_qa_text_scorer_handles_partial_overlap() -> None:
    assert canonicalize_no_answer("") == "unanswerable"
    exact, f1_score, best_index = best_qa_scores("Normandy, France", ["France", "Paris"])
    assert exact == 0.0
    assert best_index == 0
    assert 0.0 < f1_score < 1.0
