# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
import sacrebleu

import evalution
from evalution.engines.base import GenerationOutput

flores_es_module = importlib.import_module("evalution.benchmarks.flores_es")


def test_flores_es_scores_translation_metrics(monkeypatch) -> None:
    docs = [
        {
            "id": 1,
            "URL": "https://example.com/1",
            "domain": "news",
            "topic": "medicine",
            "has_image": False,
            "has_hyperlink": True,
            "sentence_eng_Latn": "The team reported a careful improvement in outcomes after the pilot study.",
            "sentence_spa_Latn": "El equipo informó de una mejora cuidadosa en los resultados tras el estudio piloto.",
        },
        {
            "id": 2,
            "URL": "https://example.com/2",
            "domain": "news",
            "topic": "travel",
            "has_image": True,
            "has_hyperlink": False,
            "sentence_eng_Latn": "Researchers expect the final report to include a broader international sample.",
            "sentence_spa_Latn": "Los investigadores esperan que el informe final incluya una muestra internacional más amplia.",
        },
    ]
    monkeypatch.setattr(flores_es_module, "load_flores200_pair", lambda *args, **kwargs: list(docs))

    class FakeSession:
        def generate(self, requests, *, batch_size=None):
            assert batch_size == 2
            assert len(requests) == 2
            assert requests[0].prompt == (
                "English sentence: The team reported a careful improvement in outcomes after the pilot study.\n"
                "Spanish sentence:"
            )
            assert requests[0].stop == ["\n"]
            return [
                GenerationOutput(
                    prompt=requests[0].prompt,
                    text="El equipo informó de una mejora cuidadosa en los resultados tras el estudio piloto.",
                ),
                GenerationOutput(
                    prompt=requests[1].prompt,
                    text="Los investigadores esperan que el informe final presente una muestra global menor.",
                ),
            ]

    result = evalution.benchmarks.flores_es(direction="en-es", max_rows=2, batch_size=2).evaluate(FakeSession())

    references = [doc["sentence_spa_Latn"] for doc in docs]
    predictions = [
        "El equipo informó de una mejora cuidadosa en los resultados tras el estudio piloto.",
        "Los investigadores esperan que el informe final presente una muestra global menor.",
    ]
    assert result.name == "flores_es_en_es"
    assert result.metrics == {
        "bleu": pytest.approx(sacrebleu.corpus_bleu(predictions, [references]).score),
        "chrf": pytest.approx(sacrebleu.corpus_chrf(predictions, [references]).score),
        "ter": pytest.approx(sacrebleu.corpus_ter(predictions, [references]).score),
    }
    assert result.metadata["dataset_path"] == "facebook/flores"
    assert result.metadata["dataset_name"] == "all"
    assert result.metadata["direction"] == "en-es"
    assert result.metadata["upstream_task"] == "spanish_bench_flores_en-es"
    assert result.samples[0].metadata["source_language"] == "en"
    assert result.samples[0].metadata["target_language"] == "es"


def test_flores_es_dispatcher_and_validation() -> None:
    suite = evalution.benchmarks.flores_es(direction="es-pt", max_rows=1)
    alias_suite = evalution.benchmarks.flores_es_es_pt(max_rows=1)

    assert suite.task_name() == "flores_es_es_pt"
    assert suite.direction == "es-pt"
    assert alias_suite.task_name() == "flores_es_es_pt"
    assert alias_suite.direction == "es-pt"

    with pytest.raises(ValueError, match="unsupported flores_es direction"):
        evalution.benchmarks.flores_es(direction="ru-es")
