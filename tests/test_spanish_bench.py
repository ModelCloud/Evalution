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

# Keep shared test fixtures and expectations explicit at module scope.
spanish_bench_module = importlib.import_module("evalution.benchmarks.spanish_bench")


def test_copa_es_scores_accuracy(monkeypatch) -> None:
    """Verify COPA es scores accuracy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "id": "0",
                "premise": "El hombre abrió el grifo.",
                "choice1": "El retrete se llenó de agua.",
                "choice2": "El agua fluyó del grifo.",
                "question": "effect",
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(spanish_bench_module, "load_dataset", lambda *args, **kwargs: dataset)

    class FakeSession:
        """Provide the fake session helper used by the surrounding tests."""
        def loglikelihood(self, requests, *, batch_size=None):
            """Implement loglikelihood for fake session."""
            assert batch_size == 6
            assert len(requests) == 2
            assert requests[0].context == "El hombre abrió el grifo y por lo tanto"
            assert requests[0].continuation == " el retrete se llenó de agua."
            assert requests[1].continuation == " el agua fluyó del grifo."
            return [
                LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=6),
                LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=6),
            ]

    result = evalution.benchmarks.copa_es(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "copa_es"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "BSC-LT/COPA-es"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "test"
    sample = result.samples[0]
    assert sample.prompt == "El hombre abrió el grifo y por lo tanto"
    assert sample.target == "el agua fluyó del grifo."
    assert sample.prediction == "el agua fluyó del grifo."
    assert sample.metadata["id"] == "0"
    assert sample.metadata["question"] == "effect"


def test_escola_scores_accuracy_and_mcc(monkeypatch) -> None:
    """Verify escola scores accuracy and MCC. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "ID": "EsCoLA_3263",
                "Source": "GDE23",
                "Label": 1,
                "Sentence": "Ana y Juan son diferentes.",
                "Category": 2,
                "Split": "dev",
            },
            {
                "ID": "EsCoLA_3264",
                "Source": "GDE23",
                "Label": 0,
                "Sentence": "Ana es dejó.",
                "Category": 2,
                "Split": "dev",
            },
            {
                "ID": "EsCoLA_3265",
                "Source": "GDE23",
                "Label": 1,
                "Sentence": "Juan llegó temprano.",
                "Category": 2,
                "Split": "dev",
            },
        ]
    )
    monkeypatch.setattr(spanish_bench_module, "load_dataset", lambda *args, **kwargs: dataset)

    class FakeSession:
        """Provide the fake session helper used by the surrounding tests."""
        def loglikelihood(self, requests, *, batch_size=None):
            """Implement loglikelihood for fake session."""
            assert batch_size == 4
            assert len(requests) == 6
            assert requests[0].context == (
                "Ana y Juan son diferentes.\n"
                "Pregunta: ¿Tiene sentido esta frase?\n"
                "Respuesta:"
            )
            assert requests[0].continuation == " no"
            assert requests[1].continuation == " sí"
            return [
                LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
                LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
                LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
                LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
                LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
                LoglikelihoodOutput(logprob=-0.2, is_greedy=False, token_count=1),
            ]

    result = evalution.benchmarks.escola(max_rows=3, batch_size=4).evaluate(FakeSession())

    assert result.name == "escola"
    assert result.metrics["acc,ll"] == pytest.approx(2 / 3)
    assert result.metrics["acc,ll_avg"] == pytest.approx(2 / 3)
    assert result.metrics["mcc,ll"] == pytest.approx(0.5)
    assert result.metrics["mcc,ll_avg"] == pytest.approx(0.5)
    assert result.metadata["dataset_path"] == "nbel/EsCoLA"
    assert result.metadata["split"] == "validation"
    assert result.samples[0].target == "sí"
    assert result.samples[0].prediction == "sí"
    assert result.samples[0].metadata["id"] == "EsCoLA_3263"
    assert result.samples[2].prediction == "no"


def test_openbookqa_es_scores_accuracy(monkeypatch) -> None:
    """Verify openbookqa es scores accuracy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "id": "8-376",
                "question_stem": (
                    "Los tiburones anguila y los rapes viven muy por debajo de la "
                    "superficie del océano, y por eso se les conoce como"
                ),
                "choices": {
                    "text": [
                        "fauna abisal.",
                        "peces.",
                        "peces de mar adentro.",
                        "fauna de alta mar.",
                    ],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": "A",
            }
        ]
    )
    monkeypatch.setattr(spanish_bench_module, "load_dataset", lambda *args, **kwargs: dataset)

    class FakeSession:
        """Provide the fake session helper used by the surrounding tests."""
        def loglikelihood(self, requests, *, batch_size=None):
            """Implement loglikelihood for fake session."""
            assert batch_size == 7
            assert len(requests) == 4
            assert requests[0].context == (
                "Los tiburones anguila y los rapes viven muy por debajo de la "
                "superficie del océano, y por eso se les conoce como"
            )
            assert requests[0].continuation == " fauna abisal."
            assert requests[3].continuation == " fauna de alta mar."
            return [
                LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=2),
                LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
                LoglikelihoodOutput(logprob=-1.9, is_greedy=False, token_count=4),
                LoglikelihoodOutput(logprob=-2.1, is_greedy=False, token_count=4),
            ]

    result = evalution.benchmarks.openbookqa_es(max_rows=1, batch_size=7).evaluate(FakeSession())

    assert result.name == "openbookqa_es"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "BSC-LT/openbookqa-es"
    assert result.metadata["split"] == "test"
    assert result.samples[0].prompt.startswith("Los tiburones anguila")
    assert result.samples[0].target == "fauna abisal."
    assert result.samples[0].metadata["choice_labels"] == ["A", "B", "C", "D"]


def test_paws_es_scores_full_sentence_choices_without_leading_space(monkeypatch) -> None:
    """Verify paws es scores full sentence choices without leading space. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "id": 10,
                "sentence1": "La excepción fue entre fines de 2005 y 2009.",
                "sentence2": "La excepción se dio entre fines del 2005 y 2009.",
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(spanish_bench_module, "load_dataset", lambda *args, **kwargs: dataset)

    class FakeSession:
        """Provide the fake session helper used by the surrounding tests."""
        def loglikelihood(self, requests, *, batch_size=None):
            """Implement loglikelihood for fake session."""
            assert batch_size == 5
            assert len(requests) == 2
            assert requests[0].context == ""
            assert requests[0].continuation == (
                "La excepción fue entre fines de 2005 y 2009, ¿verdad? No, "
                "la excepción se dio entre fines del 2005 y 2009."
            )
            assert requests[1].continuation == (
                "La excepción fue entre fines de 2005 y 2009, ¿verdad? Sí, "
                "la excepción se dio entre fines del 2005 y 2009."
            )
            return [
                LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=14),
                LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=14),
            ]

    result = evalution.benchmarks.paws_es_spanish_bench(max_rows=1, batch_size=5).evaluate(FakeSession())

    assert result.name == "paws_es_spanish_bench"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "paws-x"
    assert result.metadata["dataset_name"] == "es"
    assert result.metadata["split"] == "test"
    assert result.samples[0].prompt == ""
    assert result.samples[0].target.endswith("Sí, la excepción se dio entre fines del 2005 y 2009.")


def test_wnli_es_scores_true_false_multiple_choice_accuracy(monkeypatch) -> None:
    """Verify WNLI es scores true false multiple choice accuracy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "index": 0,
                "sentence1": "El desagüe se ha atascado con pelo. Hay que limpiarlo.",
                "sentence2": "Hay que limpiar el pelo.",
                "label": 0,
            },
            {
                "index": 1,
                "sentence1": "Jane llamó a la puerta de Susan, pero no respondió.",
                "sentence2": "Susan no respondió.",
                "label": 1,
            },
        ]
    )
    monkeypatch.setattr(spanish_bench_module, "load_wnli_es_dataset", lambda *args, **kwargs: dataset)

    class FakeSession:
        """Provide the fake session helper used by the surrounding tests."""
        def loglikelihood(self, requests, *, batch_size=None):
            """Implement loglikelihood for fake session."""
            assert batch_size == 4
            assert len(requests) == 4
            assert requests[0].context == (
                "El desagüe se ha atascado con pelo. Hay que limpiarlo.\n"
                "Pregunta: Hay que limpiar el pelo. ¿Verdadero o Falso?\n"
                "Respuesta:"
            )
            assert requests[0].continuation == " Falso"
            assert requests[1].continuation == " Verdadero"
            return [
                LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
                LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
                LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
                LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            ]

    result = evalution.benchmarks.wnli_es(max_rows=2, batch_size=4).evaluate(FakeSession())

    assert result.name == "wnli_es"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "PlanTL-GOB-ES/wnli-es"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "validation"
    assert result.samples[0].target == "Falso"
    assert result.samples[0].prediction == "Falso"
    assert result.samples[0].metadata["idx"] == 0
    assert result.samples[1].target == "Verdadero"


def test_xnli_es_scores_full_sentence_choices_without_leading_space(monkeypatch) -> None:
    """Verify XNLI es scores full sentence choices without leading space. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "premise": "Y él dijo: Mamá, estoy en casa.",
                "hypothesis": "Llamó a su madre tan pronto como el autobús escolar lo dejó.",
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(spanish_bench_module, "load_dataset", lambda *args, **kwargs: dataset)

    class FakeSession:
        """Provide the fake session helper used by the surrounding tests."""
        def loglikelihood(self, requests, *, batch_size=None):
            """Implement loglikelihood for fake session."""
            assert batch_size == 4
            assert len(requests) == 3
            assert requests[0].context == ""
            assert requests[0].continuation == (
                "Y él dijo: Mamá, estoy en casa, ¿correcto? Sí, "
                "llamó a su madre tan pronto como el autobús escolar lo dejó."
            )
            assert requests[1].continuation == (
                "Y él dijo: Mamá, estoy en casa, ¿correcto? Así que, "
                "llamó a su madre tan pronto como el autobús escolar lo dejó."
            )
            assert requests[2].continuation == (
                "Y él dijo: Mamá, estoy en casa, ¿correcto? No, "
                "llamó a su madre tan pronto como el autobús escolar lo dejó."
            )
            return [
                LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=15),
                LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=15),
                LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=15),
            ]

    result = evalution.benchmarks.xnli_es_spanish_bench(max_rows=1, batch_size=4).evaluate(FakeSession())

    assert result.name == "xnli_es_spanish_bench"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "xnli"
    assert result.metadata["dataset_name"] == "es"
    assert result.metadata["split"] == "validation"
    assert result.samples[0].prompt == ""
    assert "Así que" in result.samples[0].target


def test_spanish_bench_dispatch_and_validation() -> None:
    """Verify spanish bench dispatch and validation."""
    suite = evalution.benchmarks.spanish_bench(task="openbookqa_es", max_rows=1)
    assert suite.task_name() == "openbookqa_es"
    assert suite.dataset_path == "BSC-LT/openbookqa-es"
    assert suite.dataset_name is None
    assert suite.split == "test"

    wnli_suite = evalution.benchmarks.spanish_bench(task="wnli_es", max_rows=1)
    assert wnli_suite.task_name() == "wnli_es"
    assert wnli_suite.dataset_path == "PlanTL-GOB-ES/wnli-es"
    assert wnli_suite.dataset_name is None
    assert wnli_suite.split == "validation"

    with pytest.raises(ValueError, match="unsupported spanish_bench task"):
        evalution.benchmarks.spanish_bench(task="cocoteros_es")

    with pytest.raises(ValueError, match="dataset_name must match"):
        evalution.benchmarks.spanish_bench(task="paws_es_spanish_bench", dataset_name="en")

    with pytest.raises(ValueError, match="split must match"):
        evalution.benchmarks.spanish_bench(task="copa_es", split="validation")
