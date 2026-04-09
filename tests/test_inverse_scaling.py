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

inverse_scaling_module = importlib.import_module("evalution.benchmarks.inverse_scaling")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 2
        assert requests[0].context.endswith("Answer:")
        assert [request.continuation for request in requests] == [" Y", " N"]
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
        ]


def test_inverse_scaling_scores_raw_prompt_choices(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "prompt": "Question: Is the outcome rational? Choose Y or N.\nAnswer:",
                "classes": "[' Y', ' N']",
                "answer_index": 1,
                "round": 1,
                "part": 2,
            }
        ]
    )
    monkeypatch.setattr(inverse_scaling_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.inverse_scaling_hindsight_neglect(
        max_rows=1,
        batch_size=8,
    ).evaluate(FakeSession())

    assert result.name == "inverse_scaling_hindsight_neglect"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "pminervini/inverse-scaling"
    assert result.metadata["dataset_name"] == "hindsight-neglect"
    sample = result.samples[0]
    assert sample.target == "N"
    assert sample.prediction == "N"
    assert sample.metadata["subset"] == "hindsight-neglect"
    assert sample.metadata["round"] == 1
    assert sample.metadata["part"] == 2


def test_inverse_scaling_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="unsupported inverse_scaling subset"):
        evalution.benchmarks.inverse_scaling(subset="unknown")
