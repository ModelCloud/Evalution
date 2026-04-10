# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

swde_module = importlib.import_module("evalution.benchmarks.swde")


class FakeSession:
    def generate(self, requests, *, batch_size):
        assert batch_size == 1
        assert len(requests) == 1
        assert requests[0].prompt.endswith("Summary of information above...\nyear:")
        assert requests[0].stop == ["\n"]
        return [
            GenerationOutput(
                prompt=requests[0].prompt,
                text="1983\n",
            )
        ]


def test_swde_scores_generated_contains(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "doc_id": "id0484",
                "file_name": "0484.htm",
                "key": "year",
                "value": "1983",
                "text": "Movie details here.\n\nSummary of information above...\nyear:",
            }
        ]
    )
    monkeypatch.setattr(swde_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.swde(max_rows=1, batch_size=1).evaluate(FakeSession())

    assert result.name == "swde"
    assert result.metrics == {"contains": 1.0}
    assert result.metadata == {
        "dataset_path": "hazyresearch/based-swde-v2",
        "dataset_name": "default",
        "split": "validation",
        "order": "native",
        "stream": True,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_contains_match",
        "primary_metric": "contains",
        "prompt_variant": "webpage_attribute_completion",
    }

    sample = result.samples[0]
    assert sample.target == "1983"
    assert sample.prediction == "1983\n"
    assert sample.extracted == {
        "contains-target": "1",
        "target": "1983",
        "target-matched": "1",
    }
    assert sample.metadata == {
        "doc_id": "id0484",
        "file_name": "0484.htm",
        "key": "year",
    }
