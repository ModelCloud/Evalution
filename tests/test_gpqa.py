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

gpqa_module = importlib.import_module("evalution.benchmarks.gpqa")


class FakeMainSession:
    def generate(self, requests, *, batch_size=None):
        assert batch_size == 1
        assert len(requests) == 1
        assert requests[0].prompt == (
            "What is the correct answer to this question: Which structure below is not involved in the proposed therapy?\n"
            "\n"
            "Choices:\n"
            "(A) antisense\n"
            "(B) lariat\n"
            "(C) polyA tail\n"
            "(D) R-loops\n"
            "\n"
            'Format your response as follows: "The correct answer is (insert answer here)"'
        )
        assert requests[0].stop[0] == "\n\n"
        return [
            GenerationOutput(
                prompt=requests[0].prompt or "",
                text="After reasoning, the correct answer is (D).",
            )
        ]

    def close(self) -> None:
        return None


class FakeDiamondSession:
    def generate(self, requests, *, batch_size=None):
        assert batch_size == 1
        assert len(requests) == 1
        assert "(A) 10^-4 eV" in requests[0].prompt
        assert "(B) 10^-11 eV" in requests[0].prompt
        return [GenerationOutput(prompt=requests[0].prompt or "", text="10^-4 eV")]

    def close(self) -> None:
        return None


def test_gpqa_main_uses_seeded_choice_shuffle_and_label_scoring(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "Question": "Which structure below is not involved in the proposed therapy?",
                "Correct Answer": "R-loops",
                "Incorrect Answer 1": "lariat",
                "Incorrect Answer 2": "polyA tail",
                "Incorrect Answer 3": "antisense",
                "Record ID": "rec-main",
                "High-level domain": "Biology",
                "Subdomain": "Molecular Biology",
            }
        ]
    )
    monkeypatch.setattr(gpqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.gpqa_main(max_rows=1, batch_size=2).evaluate(FakeMainSession())

    assert result.name == "gpqa_main"
    assert result.metrics == {"em,choice_label": 1.0}
    assert result.metadata["dataset_path"] == "Idavidrein/gpqa"
    assert result.metadata["dataset_name"] == "gpqa_main"
    assert result.metadata["split"] == "train"
    assert result.metadata["subset"] == "main"
    assert result.metadata["shuffle_seed"] == 0
    assert result.metadata["scoring_mode"] == "generated_choice_label_exact_match"
    sample = result.samples[0]
    assert sample.target == "D"
    assert sample.extracted["choice-label"] == "D"
    assert sample.extracted["choice-text"] == "R-loops"
    assert sample.metadata["choice_texts"] == ["antisense", "lariat", "polyA tail", "R-loops"]
    assert sample.metadata["gold_choice"] == "R-loops"


def test_gpqa_diamond_recovers_answer_from_choice_text(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "Question": "Which one of the following options could be clearly distinguished?",
                "Correct Answer": "10^-4 eV",
                "Incorrect Answer 1": "10^-11 eV",
                "Incorrect Answer 2": "10^-8 eV",
                "Incorrect Answer 3": "10^-9 eV",
                "Record ID": "rec-diamond",
                "High-level domain": "Physics",
                "Subdomain": "Physics (general)",
            }
        ]
    )
    monkeypatch.setattr(gpqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.gpqa_diamond(seed=1, max_rows=1, batch_size=1).evaluate(FakeDiamondSession())

    assert result.name == "gpqa_diamond"
    assert result.metrics == {"em,choice_label": 1.0}
    sample = result.samples[0]
    assert sample.target == "A"
    assert sample.extracted["choice-label"] == "A"
    assert sample.extracted["choice-text"] == "10^-4 eV"
    assert sample.metadata["shuffle_seed"] == 1


def test_gpqa_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="unsupported gpqa subset"):
        evalution.benchmarks.gpqa(subset="unknown_subset")


def test_gpqa_rejects_dataset_name_mismatch() -> None:
    with pytest.raises(ValueError, match="dataset_name must match"):
        evalution.benchmarks.gpqa(subset="main", dataset_name="gpqa_diamond")
