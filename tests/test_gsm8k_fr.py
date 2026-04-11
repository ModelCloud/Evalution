# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

# Keep shared test fixtures and expectations explicit at module scope.
gsm8k_fr_module = importlib.import_module("evalution.benchmarks.gsm8k_fr")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self, responses: list[str]) -> None:
        """Initialize this object."""
        self.responses = responses

    def generate(self, requests, *, batch_size=None):
        """Generate generate."""
        assert batch_size in {1, 4}
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]

    def close(self) -> None:
        """Release the resources owned by this object."""
        return None


def test_gsm8k_fr_scores_numeric_primary(monkeypatch) -> None:
    """Verify GSM8K fr scores numeric primary. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "question": "Combien font 40 plus 2 ?",
                "answer": "40 + 2 = 42\n#### 42",
            }
        ]
    )
    monkeypatch.setattr(gsm8k_fr_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k_fr(max_rows=1, batch_size=4)
    result = suite.evaluate(FakeSession(["La reponse finale est 42."]))

    assert result.name == "gsm8k_fr"
    assert result.metrics == {"acc,num": 1.0}
    assert result.metadata["dataset_path"] == "cmh/gsm8k_fr"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "test"
    assert result.metadata["variant"] == "base"
    assert result.metadata["scoring_mode"] == "numeric_format_insensitive"
    assert result.samples[0].prompt == "Question: Combien font 40 plus 2 ?\nAnswer:"
    assert result.samples[0].extracted["numeric-extract"] == "42"
