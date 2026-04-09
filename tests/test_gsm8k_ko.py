# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

gsm8k_ko_module = importlib.import_module("evalution.benchmarks.gsm8k_ko")


class FakeSession:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def generate(self, requests, *, batch_size=None):
        assert batch_size in {1, 4}
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]

    def close(self) -> None:
        return None


def test_gsm8k_ko_scores_numeric_primary(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "40 더하기 2는 얼마입니까?",
                "answer": "40 + 2 = 42\n#### 42",
                "question_en": "What is 40 plus 2?",
                "answer_en": "40 + 2 = 42\n#### 42",
            }
        ]
    )
    monkeypatch.setattr(gsm8k_ko_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_ko(max_rows=1, batch_size=4)
    result = suite.evaluate(FakeSession(["최종 답은 42입니다."]))

    assert result.name == "gsm8k_ko"
    assert result.metrics == {"acc,num": 1.0}
    assert result.metadata["dataset_path"] == "kuotient/gsm8k-ko"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "test"
    assert result.metadata["variant"] == "base"
    assert result.metadata["scoring_mode"] == "numeric_format_insensitive"
    assert result.samples[0].prompt == "Question: 40 더하기 2는 얼마입니까?\nAnswer:"
    assert result.samples[0].extracted["numeric-extract"] == "42"
