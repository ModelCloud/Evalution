# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

wsc_module = importlib.import_module("evalution.benchmarks.wsc")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 9
        assert len(requests) == 2
        assert requests[0].context == (
            'Passage: The trophy doesn\'t fit in the suitcase because *it* is too large.\n'
            'Question: In the passage above, does the pronoun "*it*" refer to "*the trophy*"?\n'
            "Answer:"
        )
        assert requests[0].continuation == " no"
        assert requests[1].continuation == " yes"
        return [
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
        ]


def test_wsc_scores_superglue_wsc_fixed_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "text": "The trophy doesn't fit in the suitcase because it is too large.",
                "span1_text": "the trophy",
                "span2_text": "it",
                "span2_index": 8,
                "idx": 3,
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(wsc_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.wsc(max_rows=1, batch_size=9).evaluate(FakeSession())

    assert result.name == "wsc"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "super_glue"
    assert result.metadata["dataset_name"] == "wsc.fixed"
    assert result.metadata["split"] == "validation"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.target == "yes"
    assert sample.prediction == "yes"
    assert sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }
    assert sample.metadata["idx"] == 3
    assert sample.metadata["noun"] == "the trophy"
    assert sample.metadata["pronoun"] == "it"
    assert sample.metadata["span2_index"] == 8


def test_wsc_prompt_matches_upstream_wsc_formatting() -> None:
    doc = {
        "text": "The fish ate the worm. It was tasty.",
        "span1_text": "the worm",
        "span2_text": "It",
        "span2_index": 5,
    }
    assert wsc_module._wsc_prompt(doc) == (
        "Passage: The fish ate the worm. *It* was tasty.\n"
        'Question: In the passage above, does the pronoun "*It*" refer to "*the worm*"?\n'
        "Answer:"
    )
