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

mgsm_module = importlib.import_module("evalution.benchmarks.mgsm")


class FakeSession:
    def __init__(self) -> None:
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        assert batch_size in {1, 8}
        self.requests.extend(requests)
        assert len(requests) == 1
        assert requests[0].prompt == (
            "Question: If Maya has 7 marbles and buys 5 more, how many marbles does she have?\n"
            "Answer:"
        )
        return [
            GenerationOutput(
                prompt=requests[0].prompt,
                text="The answer is 12.",
            )
        ]

    def close(self) -> None:
        return None


def test_mgsm_scores_direct_numeric_generation(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "If Maya has 7 marbles and buys 5 more, how many marbles does she have?",
                "answer_number": 12,
            }
        ]
    )
    monkeypatch.setattr(mgsm_module, "_load_mgsm_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.mgsm_direct_en(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "mgsm_direct_en"
    assert result.metrics == {"acc,num": 1.0}
    assert result.metadata["dataset_path"] == "juletxara/mgsm"
    assert result.metadata["dataset_name"] == "en"
    assert result.metadata["split"] == "test"
    assert result.metadata["variant"] == "base"
    assert result.metadata["language"] == "en"
    sample = result.samples[0]
    assert sample.target == "12"
    assert sample.extracted["numeric-extract"] == "12"
    assert sample.metadata["language"] == "en"
    assert sample.metadata["answer_number"] == "12"


def test_mgsm_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="unsupported MGSM language"):
        evalution.benchmarks.mgsm(language="xx")


def test_mgsm_rejects_non_base_variants() -> None:
    with pytest.raises(ValueError, match="only supports the direct base variant"):
        evalution.benchmarks.mgsm(language="en", variant="cot")
