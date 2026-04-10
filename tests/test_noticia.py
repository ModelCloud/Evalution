# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

noticia_module = importlib.import_module("evalution.benchmarks.noticia")


class FakeSession:
    def generate(self, requests, *, batch_size):
        assert batch_size == 1
        assert len(requests) == 1
        assert requests[0].prompt == noticia_module._noticia_prompt(
            "Le compra un abrigo a su abuela de 97 años y la reacción de esta es una fantasía",
            "La abuela rechaza el abrigo porque le parece de vieja.",
        )
        assert requests[0].stop == ["\n\n", "\n"]
        return [
            GenerationOutput(
                prompt=requests[0].prompt,
                text="Es de vieja!!!",
            )
        ]


def test_noticia_scores_rouge1_and_average_length(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "web_url": "https://example.com/noticia",
                "web_headline": "Le compra un abrigo a su abuela de 97 años y la reacción de esta es una fantasía",
                "summary": "Es de vieja.",
                "web_text": "La abuela rechaza el abrigo porque le parece de vieja.",
            }
        ]
    )
    monkeypatch.setattr(noticia_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.noticia(max_rows=1, batch_size=1).evaluate(FakeSession())

    assert result.name == "noticia"
    assert result.metrics == {
        "rouge1": 1.0,
        "average_len": 3.0,
    }
    assert result.metadata == {
        "dataset_path": "Iker/NoticIA",
        "dataset_name": None,
        "split": "test",
        "order": "native",
        "stream": False,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_clickbait_truth_summary",
        "primary_metric": "rouge1",
        "prompt_variant": "headline_body_to_truth_summary",
    }

    sample = result.samples[0]
    assert sample.target == "Es de vieja."
    assert sample.prediction == "Es de vieja!!!"
    assert sample.extracted == {
        "prediction-clean": "es de vieja",
        "reference-clean": "es de vieja",
    }
    assert sample.metadata == {
        "web_url": "https://example.com/noticia",
        "web_headline": "Le compra un abrigo a su abuela de 97 años y la reacción de esta es una fantasía",
        "web_text_chars": len("La abuela rechaza el abrigo porque le parece de vieja."),
    }
