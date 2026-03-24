# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

icelandic_winogrande_module = importlib.import_module("evalution.benchmarks.icelandic_winogrande")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 2
        assert requests[0].context == "Valmundur var að læra að hjóla og Bernharður var með honum af því að Valmundur"
        assert requests[0].continuation == " hafði enga reynslu af því að hjóla."
        assert requests[1].context == "Valmundur var að læra að hjóla og Bernharður var með honum af því að Bernharður"
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=7),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=7),
        ]


def test_icelandic_winogrande_scores_partial_evaluation_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "qID": "302OLP89DZ5MCAWZNC1Z2EMR54LACJ-1",
                "sentence": "Valmundur var að læra að hjóla og Bernharður var með honum af því að _ hafði enga reynslu af því að hjóla.",
                "option1": "Valmundur",
                "option2": "Bernharður",
                "answer": "1",
            }
        ]
    )
    monkeypatch.setattr(icelandic_winogrande_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.icelandic_winogrande(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "icelandic_winogrande"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "mideind/icelandic-winogrande"
    assert result.metadata["split"] == "train"
    assert result.metadata["prompt_variant"] == "partial_evaluation_blank_replacement"

    sample = result.samples[0]
    assert sample.prompt == "Valmundur var að læra að hjóla og Bernharður var með honum af því að _ hafði enga reynslu af því að hjóla."
    assert sample.target == "Valmundur var að læra að hjóla og Bernharður var með honum af því að Valmundur hafði enga reynslu af því að hjóla."
    assert sample.prediction == sample.target
    assert sample.metadata["qID"] == "302OLP89DZ5MCAWZNC1Z2EMR54LACJ-1"
    assert sample.metadata["choice_labels"] == ["A", "B"]
    assert sample.metadata["choice_texts"] == ["Valmundur", "Bernharður"]
    assert sample.metadata["answer_label"] == "1"


def test_blank_choice_contexts_and_suffix_handles_spaced_and_unspaced_blanks() -> None:
    assert icelandic_winogrande_module._blank_choice_contexts_and_suffix(
        "Valmundur var að læra að hjóla og Bernharður var með honum af því að _ hafði enga reynslu af því að hjóla.",
        "Valmundur",
        "Bernharður",
    ) == (
        [
            "Valmundur var að læra að hjóla og Bernharður var með honum af því að Valmundur",
            "Valmundur var að læra að hjóla og Bernharður var með honum af því að Bernharður",
        ],
        " hafði enga reynslu af því að hjóla.",
    )
    assert icelandic_winogrande_module._blank_choice_contexts_and_suffix(
        "Báturinn sökk því _ var með gat.",
        "hann",
        "hanninn",
    ) == (
        [
            "Báturinn sökk því hann",
            "Báturinn sökk því hanninn",
        ],
        " var með gat.",
    )
