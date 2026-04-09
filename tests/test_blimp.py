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

blimp_module = importlib.import_module("evalution.benchmarks.blimp")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == ""
        assert requests[0].continuation == "Who should Derek hug after shocking Richard?"
        assert requests[1].context == ""
        assert requests[1].continuation == "Who should Derek hug Richard after shocking?"
        assert requests[2].context == ""
        assert requests[2].continuation == "The lawyer admired the judge."
        assert requests[3].context == ""
        assert requests[3].continuation == "The lawyer admired judge the."
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=8),
            LoglikelihoodOutput(logprob=-4.0, is_greedy=False, token_count=8),
            LoglikelihoodOutput(logprob=-3.5, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=6),
        ]


def test_blimp_scores_sentence_pairs(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "sentence_good": "Who should Derek hug after shocking Richard?",
                "sentence_bad": "Who should Derek hug Richard after shocking?",
                "field": "syntax",
                "linguistics_term": "island_effects",
                "UID": "adjunct_island",
                "simple_LM_method": True,
                "one_prefix_method": False,
                "two_prefix_method": False,
                "lexically_identical": True,
                "pair_id": 0,
            },
            {
                "sentence_good": "The lawyer admired the judge.",
                "sentence_bad": "The lawyer admired judge the.",
                "field": "syntax",
                "linguistics_term": "word_order",
                "UID": "adjunct_island",
                "simple_LM_method": True,
                "one_prefix_method": False,
                "two_prefix_method": False,
                "lexically_identical": True,
                "pair_id": 1,
            },
        ]
    )
    monkeypatch.setattr(blimp_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.blimp(
        subset="adjunct_island",
        max_rows=2,
        batch_size=8,
    ).evaluate(FakeSession())

    assert result.name == "blimp_adjunct_island"
    assert result.metrics == {
        "acc,ll": 0.5,
        "acc,ll_avg": 0.5,
    }
    assert result.metadata == {
        "dataset_path": "blimp",
        "dataset_name": "adjunct_island",
        "split": "train",
        "order": "native",
        "stream": False,
        "scoring_mode": "multiple_choice_loglikelihood",
        "prompt_variant": "full_sentence_pair",
    }
    assert len(result.samples) == 2

    sample = result.samples[0]
    assert sample.prompt == ""
    assert sample.target == "Who should Derek hug after shocking Richard?"
    assert sample.prediction == "Who should Derek hug after shocking Richard?"
    assert sample.metadata["subset"] == "adjunct_island"
    assert sample.metadata["field"] == "syntax"
    assert sample.metadata["linguistics_term"] == "island_effects"
    assert sample.metadata["uid"] == "adjunct_island"
    assert sample.metadata["simple_lm_method"] is True
    assert sample.metadata["one_prefix_method"] is False
    assert sample.metadata["two_prefix_method"] is False
    assert sample.metadata["lexically_identical"] is True
    assert sample.metadata["pair_id"] == 0


def test_blimp_normalizes_task_name() -> None:
    assert evalution.benchmarks.blimp(subset="complex_NP_island").task_name() == "blimp_complex_np_island"


def test_blimp_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="unsupported blimp subset"):
        evalution.benchmarks.blimp(subset="unknown_subset")
