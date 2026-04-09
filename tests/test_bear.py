# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

bear_module = importlib.import_module("evalution.benchmarks.bear")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        if len(requests) == 2:
            assert requests[0].context == ""
            assert requests[0].continuation == "The head of government of Abu Dhabi is Khalifa bin Zayed Al Nahyan."
            assert requests[1].continuation == "The head of government of Abu Dhabi is Kyriakos Mitsotakis."
            return [
                LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=13),
                LoglikelihoodOutput(logprob=-4.5, is_greedy=False, token_count=11),
            ]
        assert len(requests) == 3
        assert requests[0].context == ""
        assert requests[0].continuation == "The head of government of Benin is Patrice Talon."
        assert requests[1].continuation == "The head of government of Benin is Rumen Radev."
        assert requests[2].continuation == "The head of government of Benin is Giorgia Meloni."
        return [
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=10),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=10),
            LoglikelihoodOutput(logprob=-2.7, is_greedy=False, token_count=10),
        ]


def test_bear_scores_bear_and_bear_big_rows(monkeypatch) -> None:
    bear_dataset = Dataset.from_list(
        [
            {
                "composite_id": "head-of-government-abu-dhabi-0",
                "relation": "head_of_government",
                "item": 0,
                "template_index": 0,
                "template": "The head of government of [X] is [Y].",
                "subject": "Abu Dhabi",
                "answer_options": ["Khalifa bin Zayed Al Nahyan", "Kyriakos Mitsotakis"],
                "correct": 0,
                "text_options": [
                    "The head of government of Abu Dhabi is Khalifa bin Zayed Al Nahyan.",
                    "The head of government of Abu Dhabi is Kyriakos Mitsotakis.",
                ],
            }
        ]
    )
    bear_big_dataset = Dataset.from_list(
        [
            {
                "composite_id": "head-of-government-benin-0",
                "relation": "head_of_government",
                "item": 0,
                "template_index": 0,
                "template": "The head of government of [X] is [Y].",
                "subject": "Benin",
                "answer_options": ["Patrice Talon", "Rumen Radev", "Giorgia Meloni"],
                "correct": 1,
                "text_options": [
                    "The head of government of Benin is Patrice Talon.",
                    "The head of government of Benin is Rumen Radev.",
                    "The head of government of Benin is Giorgia Meloni.",
                ],
            }
        ]
    )

    def fake_load_dataset(dataset_path, dataset_name, *, split, **kwargs):
        assert dataset_path == "lm-pub-quiz/BEAR"
        assert split == "test"
        if dataset_name == "BEAR":
            return bear_dataset
        if dataset_name == "BEAR_big":
            return bear_big_dataset
        raise AssertionError(f"unexpected dataset name: {dataset_name}")

    monkeypatch.setattr(bear_module, "load_dataset", fake_load_dataset)

    bear_result = evalution.benchmarks.bear(max_rows=1, batch_size=8).evaluate(FakeSession())
    bear_big_result = evalution.benchmarks.bear_big(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert bear_result.name == "bear"
    assert bear_result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert bear_result.metadata["dataset_path"] == "lm-pub-quiz/BEAR"
    assert bear_result.metadata["dataset_name"] == "BEAR"
    assert bear_result.metadata["split"] == "test"
    assert bear_result.metadata["prompt_variant"] == "empty_context_full_statement"
    assert bear_result.samples[0].prompt == ""
    assert bear_result.samples[0].target == "The head of government of Abu Dhabi is Khalifa bin Zayed Al Nahyan."
    assert bear_result.samples[0].prediction == "The head of government of Abu Dhabi is Khalifa bin Zayed Al Nahyan."
    assert bear_result.samples[0].metadata["variant"] == "bear"
    assert bear_result.samples[0].metadata["choice_count"] == 2

    assert bear_big_result.name == "bear_big"
    assert bear_big_result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert bear_big_result.metadata["dataset_name"] == "BEAR_big"
    assert bear_big_result.samples[0].prompt == ""
    assert bear_big_result.samples[0].target == "The head of government of Benin is Rumen Radev."
    assert bear_big_result.samples[0].prediction == "The head of government of Benin is Rumen Radev."
    assert bear_big_result.samples[0].metadata["variant"] == "bear_big"
    assert bear_big_result.samples[0].metadata["choice_count"] == 3


def test_bear_rejects_unknown_variant_and_mismatched_dataset_name() -> None:
    with pytest.raises(ValueError, match="unsupported bear variant"):
        bear_module.BEAR(variant="unknown")

    with pytest.raises(ValueError, match="dataset_name must match the configured variant"):
        bear_module.BEAR(variant="bear", dataset_name="BEAR_big")
