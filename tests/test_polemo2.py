# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

polemo2_module = importlib.import_module("evalution.benchmarks.polemo2")


class FakeSession:
    def generate(self, requests, *, batch_size=None):
        assert batch_size == 2
        assert len(requests) == 2
        assert requests[0].prompt == (
            'Opinia: "Leczyla mnie zle."\n'
            "Określ sentyment podanej opinii. Możliwe odpowiedzi:\n"
            "A - Neutralny\n"
            "B - Negatywny\n"
            "C - Pozytywny\n"
            "D - Niejednoznaczny\n"
            "Prawidłowa odpowiedź:"
        )
        assert requests[0].stop == [".", ","]
        return [
            GenerationOutput(prompt=requests[0].prompt, text="B"),
            GenerationOutput(prompt=requests[1].prompt, text="C"),
        ]


def test_polemo2_in_scores_generated_choice_labels(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {"sentence": "Leczyla mnie zle.", "target": "__label__meta_minus_m"},
            {"sentence": "To jest swietne.", "target": "__label__meta_plus_m"},
        ]
    )
    monkeypatch.setattr(polemo2_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.polemo2_in(max_rows=2, batch_size=2).evaluate(FakeSession())

    assert result.name == "polemo2_in"
    assert result.metrics == {"f1": 1.0}
    assert result.metadata["dataset_path"] == "allegro/klej-polemo2-in"
    assert result.metadata["split"] == "test"
    assert result.metadata["primary_metric"] == "f1"
    assert result.samples[0].target == "B"
    assert result.samples[0].prediction == "B"
    assert result.samples[0].metadata["variant"] == "polemo2_in"


def test_polemo2_prompt_and_prediction_normalization() -> None:
    assert polemo2_module._polemo2_prompt("Opinia") == (
        'Opinia: "Opinia"\n'
        "Określ sentyment podanej opinii. Możliwe odpowiedzi:\n"
        "A - Neutralny\n"
        "B - Negatywny\n"
        "C - Pozytywny\n"
        "D - Niejednoznaczny\n"
        "Prawidłowa odpowiedź:"
    )
    assert polemo2_module._normalize_polemo2_prediction("Odpowiedz: C") == "C"
    assert polemo2_module._normalize_polemo2_prediction("brak") == ""


def test_polemo2_rejects_invalid_variant_and_dataset_settings() -> None:
    with pytest.raises(ValueError, match="unsupported polemo2 variant"):
        evalution.benchmarks.polemo2(variant="bad")
    with pytest.raises(ValueError, match="dataset_path must match"):
        polemo2_module.Polemo2(variant="polemo2_in", dataset_path="bad")
