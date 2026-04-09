# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

assin_module = importlib.import_module("evalution.benchmarks.assin")


class FakeSession:
    def __init__(self, expected_continuations: list[str], scores: list[float]) -> None:
        self.expected_continuations = expected_continuations
        self.scores = scores

    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        request_items = list(requests)
        assert [request.context for request in request_items] == [""] * len(request_items)
        assert [request.continuation for request in request_items] == self.expected_continuations
        return [
            LoglikelihoodOutput(logprob=score, is_greedy=index == 0, token_count=1)
            for index, score in enumerate(self.scores)
        ]


def test_assin_entailment_scores_two_choice_rows(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "sentence_pair_id": 7,
                "premise": "No Brasil, 809 instituições participarão da Primavera dos Museus neste ano.",
                "hypothesis": "Começa nesta segunda-feira, em todo o Brasil, a Primavera dos Museus.",
                "relatedness_score": 3.25,
                "entailment_judgment": 0,
            }
        ]
    )
    monkeypatch.setattr(assin_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.assin_entailment(max_rows=1, batch_size=4).evaluate(
        FakeSession(
            expected_continuations=[
                " No Brasil, 809 instituições participarão da Primavera dos Museus neste ano., certo? Também, Começa nesta segunda-feira, em todo o Brasil, a Primavera dos Museus.",
                " No Brasil, 809 instituições participarão da Primavera dos Museus neste ano., certo? Sim, Começa nesta segunda-feira, em todo o Brasil, a Primavera dos Museus.",
            ],
            scores=[-0.1, -2.0],
        )
    )

    assert result.name == "assin_entailment"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata == {
        "dataset_path": "nilc-nlp/assin",
        "dataset_name": None,
        "split": "test",
        "order": "native",
        "stream": False,
        "scoring_mode": "multiple_choice_loglikelihood",
    }
    sample = result.samples[0]
    assert sample.target == (
        "No Brasil, 809 instituições participarão da Primavera dos Museus neste ano., certo? Também, "
        "Começa nesta segunda-feira, em todo o Brasil, a Primavera dos Museus."
    )
    assert sample.metadata["variant"] == "assin_entailment"
    assert sample.metadata["sentence_pair_id"] == "7"
    assert sample.metadata["relatedness_score"] == 3.25


def test_assin_paraphrase_scores_two_choice_rows(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "sentence_pair_id": 8,
                "premise": "Parte dos participantes será anunciada no Programa Xuxa Meneghel.",
                "hypothesis": "Alguns dos novos participantes serão revelados no programa Xuxa Meneghel, que vai ao ar na segunda-feira.",
                "relatedness_score": 4.0,
                "entailment_judgment": 1,
            }
        ]
    )
    monkeypatch.setattr(assin_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.assin_paraphrase(max_rows=1, batch_size=4).evaluate(
        FakeSession(
            expected_continuations=[
                " Parte dos participantes será anunciada no Programa Xuxa Meneghel., certo? Não, Alguns dos novos participantes serão revelados no programa Xuxa Meneghel, que vai ao ar na segunda-feira.",
                " Parte dos participantes será anunciada no Programa Xuxa Meneghel., certo? Sim, Alguns dos novos participantes serão revelados no programa Xuxa Meneghel, que vai ao ar na segunda-feira.",
            ],
            scores=[-3.0, -0.2],
        )
    )

    assert result.name == "assin_paraphrase"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    sample = result.samples[0]
    assert sample.target == (
        "Parte dos participantes será anunciada no Programa Xuxa Meneghel., certo? Sim, "
        "Alguns dos novos participantes serão revelados no programa Xuxa Meneghel, que vai ao ar na segunda-feira."
    )
    assert sample.metadata["variant"] == "assin_paraphrase"


def test_assin_dispatcher_and_validation() -> None:
    suite = evalution.benchmarks.assin(variant="assin_paraphrase")
    assert suite.task_name() == "assin_paraphrase"
    assert suite.dataset_path == "nilc-nlp/assin"

    with pytest.raises(ValueError, match="unsupported assin variant"):
        evalution.benchmarks.assin(variant="unknown")

    with pytest.raises(ValueError, match="does not use a dataset_name"):
        assin_module.ASSIN(variant="assin_entailment", dataset_name="assin")
