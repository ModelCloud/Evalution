# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

wic_module = importlib.import_module("evalution.benchmarks.wic")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 7
        assert len(requests) == 2
        assert requests[0].context == (
            "Sentence 1: An emerging professional class.\n"
            "Sentence 2: Apologizing for losing your temper, even though you were badly provoked, showed real class.\n"
            "Question: Is the word 'class' used in the same way in the two sentences above?\n"
            "Answer:"
        )
        assert requests[0].continuation == " no"
        assert requests[1].continuation == " yes"
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
        ]


def test_wic_scores_word_in_context_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "word": "class",
                "sentence1": "An emerging professional class.",
                "sentence2": "Apologizing for losing your temper, even though you were badly provoked, showed real class.",
                "start1": 25,
                "start2": 85,
                "end1": 30,
                "end2": 90,
                "idx": 0,
                "label": 0,
            }
        ]
    )
    monkeypatch.setattr(wic_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.wic(max_rows=1, batch_size=7).evaluate(FakeSession())

    assert result.name == "wic"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "super_glue"
    assert result.metadata["dataset_name"] == "wic"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.target == "no"
    assert sample.prediction == "no"
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }
    assert sample.metadata["idx"] == 0
    assert sample.metadata["word"] == "class"


def test_wic_helpers_extract_target_word_and_prompt() -> None:
    doc = {
        "sentence1": "A short note.",
        "sentence2": "Another note.",
        "start1": 8,
        "end1": 12,
    }
    assert wic_module._wic_target_word(doc) == "note"
    assert "Is the word 'note' used in the same way" in wic_module._wic_prompt(
        {
            **doc,
            "sentence2": "Another note.",
        }
    )
