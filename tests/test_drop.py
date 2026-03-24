# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput
from evalution.scorers.qa_text import best_qa_scores

drop_module = importlib.import_module("evalution.benchmarks.drop")


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


def test_drop_scores_best_alias(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "section_id": "sec-1",
                "query_id": "query-1",
                "passage": "The Raiders scored first through Chaz Schilens.",
                "question": "Who scored the first touchdown?",
                "answers_spans": {
                    "spans": ["Chaz Schilens", "Schilens"],
                    "types": ["span", "span"],
                },
            }
        ]
    )
    monkeypatch.setattr(drop_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.drop(max_rows=1).evaluate(FakeSession(["Schilens"]))

    assert result.name == "drop"
    assert result.metadata == {
        "dataset_path": "drop",
        "dataset_name": None,
        "split": "validation",
        "streaming": False,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_qa_exact_match_f1",
        "primary_metric": "f1",
    }
    assert result.metrics == {"em": 1.0, "f1": 1.0}

    sample = result.samples[0]
    assert sample.metadata["answer_spans"] == ["Chaz Schilens", "Schilens"]
    assert sample.metadata["answer_types"] == ["span", "span"]
    assert sample.extracted["best_answer"] == "Schilens"


def test_drop_qa_text_scorer_uses_best_alias() -> None:
    exact, f1_score, best_index = best_qa_scores(
        "Schilens",
        ["Chaz Schilens", "Schilens"],
    )
    assert exact == 1.0
    assert f1_score == 1.0
    assert best_index == 1
