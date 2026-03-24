# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

alghafa_module = importlib.import_module("evalution.benchmarks.alghafa")


class CopaArabicSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 2
        assert requests[0].context == "السؤال: كان جسدي يلقي بظلّه على العشب لان\nالجواب:"
        assert requests[0].continuation == " العشب قد قطع"
        assert requests[1].continuation == " الشمس اشرقت"
        return [
            LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=3),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=2),
        ]


class PiqaArabicSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 2
        assert requests[0].context == "السؤال: كيف أقوم بتجهيز قفص خنزير غينيا لشاغليه الجدد؟\nالجواب:"
        assert requests[0].continuation.startswith(" قم بتوفير قفص للخنزير الغيني")
        assert requests[1].continuation.startswith(" املأ الحوض بالماء الساخن")
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=24),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=11),
        ]


def test_copa_ar_scores_binary_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "query": "كان جسدي يلقي بظلّه على العشب لان",
                "sol1": "العشب قد قطع",
                "sol2": "الشمس اشرقت",
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(alghafa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.copa_ar(max_rows=1, batch_size=6).evaluate(CopaArabicSession())

    assert result.name == "copa_ar"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "Hennara/copa_ar"
    assert result.metadata["split"] == "test"

    sample = result.samples[0]
    assert sample.prompt == "السؤال: كان جسدي يلقي بظلّه على العشب لان\nالجواب:"
    assert sample.target == "الشمس اشرقت"
    assert sample.prediction == "الشمس اشرقت"
    assert sample.metadata["source_benchmark"] == "copa"
    assert sample.metadata["query"] == "كان جسدي يلقي بظلّه على العشب لان"


def test_piqa_ar_scores_binary_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "goal": "كيف أقوم بتجهيز قفص خنزير غينيا لشاغليه الجدد؟",
                "sol1": (
                    "قم بتوفير قفص للخنزير الغيني مملوء ببضعة بوصات من الفراش المصنوع "
                    "من شرائط الورق الممزقة، وستحتاج أيضًا إلى تزويده بزجاجة ماء وطبق طعام."
                ),
                "sol2": "املأ الحوض بالماء الساخن ثم ضعه في ضوء الشمس.",
                "label": 0,
            }
        ]
    )
    monkeypatch.setattr(alghafa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.piqa_ar(max_rows=1, batch_size=6).evaluate(PiqaArabicSession())

    assert result.name == "piqa_ar"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "Hennara/pica_ar"
    assert result.metadata["split"] == "test"

    sample = result.samples[0]
    assert sample.prompt == "السؤال: كيف أقوم بتجهيز قفص خنزير غينيا لشاغليه الجدد؟\nالجواب:"
    assert sample.target.startswith("قم بتوفير قفص للخنزير الغيني")
    assert sample.prediction.startswith("قم بتوفير قفص للخنزير الغيني")
    assert sample.metadata["source_benchmark"] == "piqa"
    assert sample.metadata["goal"] == "كيف أقوم بتجهيز قفص خنزير غينيا لشاغليه الجدد؟"


def test_alghafa_prompt_helper() -> None:
    assert alghafa_module._arabic_question_answer_prompt("ما السبب؟") == "السؤال: ما السبب؟\nالجواب:"
