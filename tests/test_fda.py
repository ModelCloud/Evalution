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

fda_module = importlib.import_module("evalution.benchmarks.fda")


class FakeSession:
    def generate(self, requests, *, batch_size):
        assert batch_size == 1
        assert len(requests) == 1
        assert requests[0].prompt == "The sky is blue"
        return [
            GenerationOutput(
                prompt="The sky is blue",
                text="The sky is blue and clear in the morning.\n",
            )
        ]


def test_fda_scores_generated_contains(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "text": "The sky is blue",
                "value": "blue",
                "doc_id": "fda-test-001",
                "file_name": "sample.txt",
                "key": "abc",
            }
        ]
    )
    monkeypatch.setattr(fda_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.fda(
        max_rows=1,
        batch_size=1,
    ).evaluate(FakeSession())

    assert result.name == "fda"
    assert result.metrics == {"contains": 1.0}
    assert result.metadata == {
        "dataset_path": "hazyresearch/based-fda",
        "dataset_name": "default",
        "split": "validation",
        "order": "native",
        "stream": True,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_contains_match",
        "primary_metric": "contains",
    }

    sample = result.samples[0]
    assert sample.index == 0
    assert sample.prompt == "The sky is blue"
    assert sample.target == "blue"
    assert sample.prediction == "The sky is blue and clear in the morning.\n"
    assert sample.extracted == {
        "contains-target": "1",
        "target": "blue",
        "target-matched": "1",
    }
    assert sample.metadata == {
        "doc_id": "fda-test-001",
        "file_name": "sample.txt",
        "key": "abc",
    }
