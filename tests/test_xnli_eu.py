# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

xnli_eu_module = importlib.import_module("evalution.benchmarks.xnli_eu")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 6
        assert requests[0].context == (
            "Beno, horretan pentsatzen ere ez nintzen ari, baina hain frustratuta nengoen, berriro hitz egin nuen harekin., ezta? Ez dut berriz harekin hitz egin."
        )
        assert [request.continuation for request in requests[:3]] == [
            " Bai, Ez dut berriz harekin hitz egin.",
            " Gainera, Ez dut berriz harekin hitz egin.",
            " Ez, Ez dut berriz harekin hitz egin.",
        ]
        return [
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=8),
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=8),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=8),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=7),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=7),
            LoglikelihoodOutput(logprob=-2.1, is_greedy=False, token_count=7),
        ]


def test_xnli_eu_scores_three_way_multiple_choice(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "premise": "Beno, horretan pentsatzen ere ez nintzen ari, baina hain frustratuta nengoen, berriro hitz egin nuen harekin.",
                "hypothesis": "Ez dut berriz harekin hitz egin.",
                "label": 2,
            },
            {
                "premise": "Zozketa notario baten aurrean egiten da eta zenbaki ekiprobableak esleitzeko programa informatiko baten bidez egiten da.",
                "hypothesis": "Ordenagailu bat eta notario bat behar dira zozketa egiteko.",
                "label": 0,
            },
        ]
    )
    monkeypatch.setattr(xnli_eu_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.xnli_eu(max_rows=2, batch_size=4).evaluate(FakeSession())

    assert result.name == "xnli_eu"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "HiTZ/xnli-eu"
    assert result.metadata["dataset_name"] == "eu"
    assert result.metadata["split"] == "test"

    first_sample = result.samples[0]
    assert first_sample.target == "Ez, Ez dut berriz harekin hitz egin."
    assert first_sample.prediction == "Ez, Ez dut berriz harekin hitz egin."
    assert first_sample.metadata["language"] == "eu"
    assert first_sample.metadata["choice_texts"] == ["Bai", "Gainera", "Ez"]

    second_sample = result.samples[1]
    assert second_sample.target == "Bai, Ordenagailu bat eta notario bat behar dira zozketa egiteko."
    assert second_sample.prediction == "Bai, Ordenagailu bat eta notario bat behar dira zozketa egiteko."


def test_xnli_eu_prompt_helper_formats_nli_prompt() -> None:
    assert (
        xnli_eu_module._xnli_eu_prompt("Premisa", "Hipotesia")
        == "Premisa, ezta? Hipotesia"
    )
