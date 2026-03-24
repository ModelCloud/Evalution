# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

coqa_module = importlib.import_module("evalution.benchmarks.coqa")


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


def test_coqa_flattens_turns_and_builds_gold_history_prompt(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "source": "toy",
                "story": "Alpha lived in Paris with a cat.",
                "questions": ["Where did Alpha live?", "Did Alpha live alone?"],
                "answers": {
                    "input_text": ["in Paris", "no"],
                    "answer_start": [12, -1],
                    "answer_end": [20, -1],
                },
            }
        ]
    )
    monkeypatch.setattr(coqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.coqa(max_rows=2).evaluate(FakeSession(["in Paris", "no"]))

    assert result.name == "coqa"
    assert result.metadata == {
        "dataset_path": "coqa",
        "dataset_name": None,
        "split": "validation",
        "stream": False,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_qa_exact_match_f1",
        "primary_metric": "f1",
        "prompt_mode": "gold_history_conversation",
    }
    assert result.metrics == {"em": 1.0, "f1": 1.0}

    first_sample, second_sample = result.samples
    assert (
        first_sample.prompt
        == "Story: Alpha lived in Paris with a cat.\nQuestion: Where did Alpha live?\nAnswer:"
    )
    assert second_sample.prompt == (
        "Story: Alpha lived in Paris with a cat.\n"
        "Question: Where did Alpha live?\n"
        "Answer: in Paris\n"
        "Question: Did Alpha live alone?\n"
        "Answer:"
    )
    assert second_sample.metadata["history_turns"] == 1
    assert second_sample.metadata["turn_index"] == 2


def test_coqa_prompt_rejects_mismatched_history_lengths() -> None:
    try:
        coqa_module._coqa_prompt(
            story="Story.",
            history_questions=["Q1"],
            history_answers=[],
            question="Q2",
        )
    except ValueError as exc:
        assert "counts must match" in str(exc)
    else:
        raise AssertionError("expected mismatched history lengths to raise")
