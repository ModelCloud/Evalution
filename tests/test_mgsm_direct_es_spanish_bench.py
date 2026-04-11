# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

# Keep shared test fixtures and expectations explicit at module scope.
mgsm_module = importlib.import_module("evalution.benchmarks.mgsm")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def generate(self, requests, *, batch_size=None):
        """Generate generate."""
        assert batch_size in {1, 6}
        assert len(requests) == 1
        assert requests[0].prompt == (
            "Question: Si Maya tiene 7 canicas y compra 5 más, ¿cuántas canicas tiene?\n"
            "Answer:"
        )
        return [
            GenerationOutput(
                prompt=requests[0].prompt,
                text="La respuesta es 12.",
            )
        ]

    def close(self) -> None:
        """Release the resources owned by this object."""
        return None


def test_mgsm_direct_es_spanish_bench_scores_numeric_generation(monkeypatch) -> None:
    """Verify mgsm direct es spanish bench scores numeric generation. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "question": "Si Maya tiene 7 canicas y compra 5 más, ¿cuántas canicas tiene?",
                "answer_number": 12,
            }
        ]
    )
    monkeypatch.setattr(mgsm_module, "_load_mgsm_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.mgsm_direct_es_spanish_bench(max_rows=1, batch_size=6).evaluate(
        FakeSession()
    )

    assert result.name == "mgsm_direct_es_spanish_bench"
    assert result.metrics == {"acc,num": 1.0}
    assert result.metadata["dataset_path"] == "juletxara/mgsm"
    assert result.metadata["dataset_name"] == "es"
    assert result.metadata["split"] == "test"
    assert result.metadata["variant"] == "base"
    assert result.metadata["language"] == "es"
    assert result.samples[0].target == "12"
    assert result.samples[0].extracted["numeric-extract"] == "12"
