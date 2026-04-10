# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
import sacrebleu
from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

phrases_es_module = importlib.import_module("evalution.benchmarks.phrases_es")


def test_phrases_es_scores_translation_metrics(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": 45901,
                "es": "No creo que ellos sean nuestra culpa.",
                "va": "No crec que ells siguen culpa nostra.",
            },
            {
                "id": 17113,
                "es": "En estos compresores la capacidad se ve afectada por la presión de trabajo.",
                "va": "En estos compressors, la capacitat es veu afectada per la pressió de treball.",
            },
        ]
    )
    monkeypatch.setattr(phrases_es_module, "load_dataset", lambda *args, **kwargs: dataset)

    class FakeSession:
        def generate(self, requests, *, batch_size=None):
            assert batch_size == 2
            assert len(requests) == 2
            assert requests[0].prompt == (
                "Oració en espanyol: No creo que ellos sean nuestra culpa.\n\n"
                "Oració en valencià:"
            )
            assert requests[0].stop == ["\n"]
            return [
                GenerationOutput(
                    prompt=requests[0].prompt,
                    text="No crec que ells siguen culpa nostra.",
                ),
                GenerationOutput(
                    prompt=requests[1].prompt,
                    text="En estos compressors, la capacitat està afectada pel treball.",
                ),
            ]

    result = evalution.benchmarks.phrases_es(direction="es-va", max_rows=2, batch_size=2).evaluate(
        FakeSession()
    )

    references = [row["va"] for row in dataset]
    predictions = [
        "No crec que ells siguen culpa nostra.",
        "En estos compressors, la capacitat està afectada pel treball.",
    ]
    assert result.name == "phrases_es_va"
    assert result.metrics == {
        "bleu": pytest.approx(sacrebleu.corpus_bleu(predictions, [references]).score),
        "chrf": pytest.approx(sacrebleu.corpus_chrf(predictions, [references]).score),
        "ter": pytest.approx(sacrebleu.corpus_ter(predictions, [references]).score),
    }
    assert result.metadata["dataset_path"] == "gplsi/ES-VA_translation_test"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["direction"] == "es-va"
    assert result.metadata["upstream_task"] == "phrases_es-va"
    assert result.samples[0].metadata["source_language"] == "es"
    assert result.samples[0].metadata["target_language"] == "va"


def test_phrases_es_dispatcher_and_validation() -> None:
    suite = evalution.benchmarks.phrases_es(direction="va-es", max_rows=1)
    alias_suite = evalution.benchmarks.phrases_va_es(max_rows=1)

    assert suite.task_name() == "phrases_va_es"
    assert suite.direction == "va-es"
    assert alias_suite.task_name() == "phrases_va_es"
    assert alias_suite.direction == "va-es"

    with pytest.raises(ValueError, match="unsupported phrases_es direction"):
        evalution.benchmarks.phrases_es(direction="es-fr")
