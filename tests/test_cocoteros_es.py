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
from evalution.scorers.summary_rouge import summary_rouge_scores

cocoteros_module = importlib.import_module("evalution.benchmarks.cocoteros_es")


def test_cocoteros_es_scores_corpus_bleu_and_mean_rouge1(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "text": "Yo me cepillo los dientes después de cada comida.",
                "keywords": "['cepillo', 'diente', 'comida']",
                "context": "La higiene bucodental es clave para prevenir enfermedades dentales.",
            },
            {
                "text": "El perro duerme tranquilo junto a la chimenea.",
                "keywords": "['perro', 'chimenea', 'duerme']",
                "context": "La casa estaba en silencio durante la tormenta.",
            },
        ]
    )
    monkeypatch.setattr(cocoteros_module, "load_dataset", lambda *args, **kwargs: dataset)

    class FakeSession:
        def generate(self, requests, *, batch_size=None):
            assert batch_size == 2
            assert len(requests) == 2
            assert requests[0].prompt == (
                "Genera una frase corta con estas palabras: ['cepillo', 'diente', 'comida']. "
                "El contexto es: La higiene bucodental es clave para prevenir enfermedades dentales. \n\n"
                "Respuesta:"
            )
            assert requests[0].stop == ["\n"]
            return [
                GenerationOutput(
                    prompt=requests[0].prompt,
                    text="Yo me cepillo los dientes después de cada comida.",
                ),
                GenerationOutput(
                    prompt=requests[1].prompt,
                    text="El perro descansa junto al fuego.",
                ),
            ]

    result = evalution.benchmarks.cocoteros_es(max_rows=2, batch_size=2).evaluate(FakeSession())

    predictions = [
        "Yo me cepillo los dientes después de cada comida.",
        "El perro descansa junto al fuego.",
    ]
    references = [
        "Yo me cepillo los dientes después de cada comida.",
        "El perro duerme tranquilo junto a la chimenea.",
    ]
    assert result.name == "cocoteros_es"
    assert result.metrics["bleu"] == pytest.approx(sacrebleu.corpus_bleu(predictions, [references]).score)
    assert result.metrics["rouge1"] == pytest.approx(
        (
            summary_rouge_scores(predictions[0], references[0])["rouge1"]
            + summary_rouge_scores(predictions[1], references[1])["rouge1"]
        )
        / 2
    )
    assert result.metadata["dataset_path"] == "gplsi/cocoteros"
    assert result.metadata["split"] == "test"
    assert result.samples[0].metadata["keywords"] == "['cepillo', 'diente', 'comida']"


def test_cocoteros_prompt_formats_generation_instruction() -> None:
    assert cocoteros_module._cocoteros_prompt("['sol']", "Hace calor.") == (
        "Genera una frase corta con estas palabras: ['sol']. El contexto es: Hace calor. \n\n"
        "Respuesta:"
    )
