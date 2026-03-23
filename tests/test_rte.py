# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

rte_module = importlib.import_module("evalution.benchmarks.rte")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 5
        assert len(requests) == 2
        assert requests[0].context == (
            "Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44.\n"
            "Question: Christopher Reeve had an accident. True or False?\n"
            "Answer:"
        )
        assert requests[0].continuation == " True"
        assert requests[1].continuation == " False"
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
        ]


def test_rte_scores_entailment_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "premise": "Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44.",
                "hypothesis": "Christopher Reeve had an accident.",
                "idx": 0,
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(rte_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.rte(max_rows=1, batch_size=5).evaluate(FakeSession())

    assert result.name == "rte"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "super_glue"
    assert result.metadata["dataset_name"] == "rte"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.target == "False"
    assert sample.prediction == "False"
    assert sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }
    assert sample.metadata["idx"] == 0


def test_rte_prompt_helper_formats_true_false_question() -> None:
    assert rte_module._rte_prompt("Premise.", "Hypothesis.") == "Premise.\nQuestion: Hypothesis. True or False?\nAnswer:"
