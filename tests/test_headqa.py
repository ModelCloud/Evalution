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

# Keep shared test fixtures and expectations explicit at module scope.
headqa_module = importlib.import_module("evalution.benchmarks.headqa")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self, *, prompt: str, continuations: list[str]) -> None:
        """Initialize this object."""
        self.prompt = prompt
        self.continuations = continuations

    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 4
        for request, continuation in zip(requests, self.continuations, strict=True):
            assert request.context == self.prompt
            assert request.continuation == continuation
        return [
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.6, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.9, is_greedy=False, token_count=1),
        ]


@pytest.mark.parametrize(
    ("factory_name", "dataset_name", "doc", "expected_prompt", "expected_choices"),
    [
        (
            "headqa_en",
            "en",
            {
                "name": "Cuaderno_2016_1_B",
                "year": "2016",
                "category": "biology",
                "qid": 1,
                "qtext": "Form extracellular fibers with high tensile strength:",
                "ra": 2,
                "answers": [
                    {"aid": 1, "atext": "Fibronectin"},
                    {"aid": 2, "atext": "Collagen"},
                    {"aid": 3, "atext": "Integrins"},
                    {"aid": 4, "atext": "Proteoglycans"},
                ],
            },
            "Question: Form extracellular fibers with high tensile strength:\nAnswer:",
            ["Fibronectin", "Collagen", "Integrins", "Proteoglycans"],
        ),
        (
            "headqa_es",
            "es",
            {
                "name": "Cuaderno_2016_1_B",
                "year": "2016",
                "category": "biology",
                "qid": 1,
                "qtext": "Forma fibras extracelulares con gran resistencia a la tensi\u00f3n:",
                "ra": 2,
                "answers": [
                    {"aid": 1, "atext": "Fibronectina."},
                    {"aid": 2, "atext": "Col\u00e1geno."},
                    {"aid": 3, "atext": "Integrinas."},
                    {"aid": 4, "atext": "Proteoglucanos."},
                ],
            },
            "Question: Forma fibras extracelulares con gran resistencia a la tensi\u00f3n:\nAnswer:",
            ["Fibronectina.", "Col\u00e1geno.", "Integrinas.", "Proteoglucanos."],
        ),
    ],
)
def test_headqa_variants_score_four_way_multiple_choice(monkeypatch, factory_name, dataset_name, doc, expected_prompt, expected_choices) -> None:
    """Verify headqa variants score four way multiple choice. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list([doc])
    monkeypatch.setattr(headqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = getattr(evalution.benchmarks, factory_name)(max_rows=1, batch_size=6).evaluate(
        FakeSession(
            prompt=expected_prompt,
            continuations=[f" {choice}" for choice in expected_choices],
        )
    )

    assert result.name == factory_name
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "EleutherAI/headqa"
    assert result.metadata["dataset_name"] == dataset_name
    assert result.metadata["split"] == "test"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == expected_prompt
    assert sample.target == expected_choices[1]
    assert sample.prediction == expected_choices[1]
    assert sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }
    assert sample.metadata["name"] == "Cuaderno_2016_1_B"
    assert sample.metadata["year"] == "2016"
    assert sample.metadata["category"] == "biology"
    assert sample.metadata["qid"] == 1
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_ids"] == ["1", "2", "3", "4"]
    assert sample.metadata["choice_texts"] == expected_choices


def test_headqa_en_can_emit_label_permutation_metric(monkeypatch) -> None:
    """Verify headqa en can emit label permutation metric. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "name": "Cuaderno_2016_1_B",
                "year": "2016",
                "category": "biology",
                "qid": 1,
                "qtext": "Form extracellular fibers with high tensile strength:",
                "ra": 2,
                "answers": [
                    {"aid": 1, "atext": "Fibronectin"},
                    {"aid": 2, "atext": "Collagen"},
                    {"aid": 3, "atext": "Integrins"},
                    {"aid": 4, "atext": "Proteoglycans"},
                ],
            }
        ]
    )
    monkeypatch.setattr(headqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    class LabelPermutationSession:
        """Define the label permutation session helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.calls = 0

        def loglikelihood(self, requests, *, batch_size=None):
            """Implement loglikelihood for label permutation session."""
            assert batch_size == 6
            self.calls += 1
            if self.calls == 1:
                assert len(requests) == 4
                return [
                    LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
                    LoglikelihoodOutput(logprob=-1.6, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.9, is_greedy=False, token_count=1),
                ]

            assert len(requests) == 24
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. Collagen" in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.4,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.benchmarks.headqa_en(
        max_rows=1,
        batch_size=6,
        label_permutations=0.25,
    ).evaluate(LabelPermutationSession())

    assert result.metrics == {
        "acc,ll": 0.0,
        "acc,ll_avg": 0.0,
        "acc,label_perm:0.25": 1.0,
    }
    assert result.metadata["label_permutations"] == 0.25
    assert result.metadata["label_permutation_metric"] == "acc,label_perm:0.25"
    assert result.samples[0].extracted["predicted_index_label_perm:0.25"] == "1"
    assert result.samples[0].metadata["label_permutation_count"] == 6
