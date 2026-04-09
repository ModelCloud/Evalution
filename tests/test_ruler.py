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

ruler_module = importlib.import_module("evalution.benchmarks.ruler")


class DummyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": text.split()}


class ContinuousSession:
    def __init__(self, prediction: str, expected_prompt: str) -> None:
        self.prediction = prediction
        self.expected_prompt = expected_prompt
        self.tokenizer = DummyTokenizer()

    def generate_continuous(self, requests, *, batch_size=None):
        assert batch_size == 4
        items = list(requests)
        assert len(items) == 1
        assert items[0][1].prompt == self.expected_prompt
        for item_id, request in items:
            yield item_id, GenerationOutput(prompt=request.prompt, text=self.prediction)


def test_ruler_requires_a_tokenizer_for_generation() -> None:
    with pytest.raises(ValueError, match="ruler requires a tokenizer"):
        evalution.benchmarks.ruler()._resolve_generation_tokenizer()


def test_ruler_scores_niah_rows_with_contains_fraction(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "index": 0,
                "input": "Context line.\nQuestion line.",
                "outputs": ["12345"],
                "gen_prefix": "Answer:",
                "length": 14,
                "max_length": 128,
            }
        ]
    )
    monkeypatch.setattr(ruler_module, "_load_ruler_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.niah_single_1(max_rows=1, batch_size=4).evaluate(
        ContinuousSession(
            prediction="Answer: 12345",
            expected_prompt="Context line.\nQuestion line. Answer:",
        )
    )

    assert result.name == "niah_single_1"
    assert result.metrics == {"contains_fraction": 1.0}
    assert result.metadata == {
        "dataset_path": "NVIDIA/RULER",
        "dataset_name": "niah_single_1",
        "split": "test",
        "order": "native",
        "stream": False,
        "generation_submission_mode": "continuous_refill",
        "variant": "niah_single_1",
        "max_length": 4096,
        "scoring_mode": "generated_contains_fraction",
        "primary_metric": "contains_fraction",
    }
    sample = result.samples[0]
    assert sample.metadata["gen_prefix"] == "Answer:"
    assert sample.metadata["max_length"] == 128


def test_ruler_generates_repeat_haystack_rows_with_real_loader() -> None:
    dataset = ruler_module._load_ruler_dataset(
        "NVIDIA/RULER",
        "niah_single_1",
        split="test",
        variant="niah_single_1",
        tokenizer=DummyTokenizer(),
        max_length=128,
        sample_count=1,
    )

    row = dataset[0]
    assert row["max_length"] == 128
    assert row["outputs"][0] in row["input"]
    assert row["gen_prefix"].startswith("The special magic")


def test_ruler_scores_qa_rows_against_multiple_references(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "index": 0,
                "input": "Document 1:\nAlpha.\n\nQuestion: Which city?",
                "outputs": ["Paris, France", "City of Paris"],
                "gen_prefix": "Answer:",
                "length": 64,
                "max_length": 512,
            }
        ]
    )
    monkeypatch.setattr(ruler_module, "_load_ruler_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.ruler_qa_squad(max_rows=1, batch_size=4).evaluate(
        ContinuousSession(
            prediction="The City of Paris is the answer.",
            expected_prompt="Document 1:\nAlpha.\n\nQuestion: Which city? Answer:",
        )
    )

    assert result.name == "ruler_qa_squad"
    assert result.metrics["contains_fraction"] == pytest.approx(0.5)
    assert result.samples[0].extracted["matched_outputs"] == ["City of Paris"]


def test_ruler_factory_aliases_and_prompt_composition() -> None:
    suite = evalution.benchmarks.ruler(variant="ruler_cwe")
    assert suite.dataset_name == "ruler_cwe"
    assert suite.task_name() == "ruler_cwe"
    assert ruler_module._compose_prompt("Prompt body", "Answer:") == "Prompt body Answer:"
    assert ruler_module._compose_prompt("Prompt body Answer:", "Answer:") == "Prompt body Answer:"

    with pytest.raises(ValueError, match="unsupported ruler variant"):
        evalution.benchmarks.ruler(variant="ruler_unknown")
