# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

gsm_plus_module = importlib.import_module("evalution.benchmarks.gsm_plus")


class FakeGenerationSession:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        assert batch_size == 2
        self.requests.extend(requests)
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]


def _dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "question": "What is 2 + 2?",
                "solution": "We add 2 and 2.\n#### 4",
                "answer": "4",
                "perturbation_type": "none",
                "seed_question": "What is 1 + 1?",
                "seed_answer": "2",
            },
            {
                "question": "What is 5 - 2?",
                "solution": "We subtract 2 from 5.\n#### 3",
                "answer": "3",
                "perturbation_type": "none",
                "seed_question": "What is 5 - 1?",
                "seed_answer": "4",
            },
        ]
    )


def test_gsm_plus_scores_regex_extracted_exact_match(monkeypatch) -> None:
    monkeypatch.setattr(gsm_plus_module, "load_dataset", lambda *args, **kwargs: _dataset())

    result = evalution.benchmarks.gsm_plus(max_rows=2, batch_size=2).evaluate(
        FakeGenerationSession(["Reasoning\n#### 4", "The answer is 3"])
    )

    assert result.name == "gsm_plus"
    assert result.metrics == {"em,strict": 0.5, "em,flex": 1.0}
    assert result.metadata["dataset_path"] == "qintongli/GSM-Plus"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "test"
    assert result.metadata["num_fewshot"] == 5
    assert result.metadata["scoring_mode"] == "generated_regex_extract_exact_match"
    assert result.metadata["primary_metric"] == "em,strict"

    sample = result.samples[0]
    assert sample.prompt.startswith("Question: ")
    assert sample.prompt.endswith("\nAnswer:")
    assert sample.target == "We add 2 and 2.\n#### 4"
    assert sample.extracted == {
        "strict-match": "4",
        "flexible-extract": "4",
        "target-stripped": "4",
    }
    assert sample.scores == {"em,strict": 1.0, "em,flex": 1.0}
    assert sample.metadata["perturbation_type"] == "none"
    assert sample.metadata["seed_solution"] == ""


def test_gsm_plus_mini_uses_testmini_split() -> None:
    suite = evalution.benchmarks.gsm_plus_mini()

    assert suite.dataset_path == "qintongli/GSM-Plus"
    assert suite.split == "testmini"


def test_gsm_plus_flexible_extract_uses_last_numeric_match() -> None:
    extracted = gsm_plus_module._extract_flexible_match(
        "20 - 11 = 9, then 9 * 3 = 27. The answer is 27."
    )

    assert extracted == "27."
    assert gsm_plus_module._normalize_gsm_plus_exact_match(extracted) == "27"
