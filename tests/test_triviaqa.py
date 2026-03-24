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

triviaqa_module = importlib.import_module("evalution.benchmarks.triviaqa")


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


def test_triviaqa_scores_best_alias(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "Who was the man behind The Chipmunks?",
                "question_id": "tc_2",
                "question_source": "http://www.triviacountry.com/",
                "entity_pages": {},
                "search_results": {},
                "answer": {
                    "aliases": ["David Seville", "Seville"],
                    "normalized_aliases": ["david seville", "seville"],
                    "matched_wiki_entity_name": "",
                    "normalized_matched_wiki_entity_name": "",
                    "normalized_value": "david seville",
                    "type": "WikipediaEntity",
                    "value": "David Seville",
                },
            }
        ]
    )
    monkeypatch.setattr(triviaqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.triviaqa(max_rows=1).evaluate(
        FakeSession(["David Seville"])
    )

    assert result.name == "triviaqa"
    assert result.metadata == {
        "dataset_path": "trivia_qa",
        "dataset_name": "rc.nocontext",
        "split": "validation",
        "stream": False,
        "order": "native",
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_qa_exact_match_f1",
        "primary_metric": "f1",
    }
    assert result.metrics == {"em": 1.0, "f1": 1.0}

    sample = result.samples[0]
    assert sample.extracted["best_answer"] == "David Seville"
    assert sample.metadata["answer_aliases"] == ["David Seville", "Seville"]
    assert sample.metadata["answer_type"] == "WikipediaEntity"


def test_triviaqa_qa_text_scorer_uses_best_alias() -> None:
    exact, f1_score, best_index = best_qa_scores("seville", ["David Seville", "Seville"])
    assert exact == 1.0
    assert f1_score == 1.0
    assert best_index == 1
