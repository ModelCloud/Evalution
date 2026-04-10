# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

squad_completion_module = importlib.import_module("evalution.benchmarks.squad_completion")


class FakeSession:
    def generate(self, requests, *, batch_size):
        assert batch_size == 1
        assert len(requests) == 1
        assert requests[0].prompt.endswith("represented the AFC at Super Bowl 50 was the")
        assert requests[0].stop == ["\n"]
        return [
            GenerationOutput(
                prompt=requests[0].prompt,
                text="Denver Broncos\n",
            )
        ]


def test_squad_completion_scores_generated_contains(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "doc_id": "56be4db0acb8001400a502ec",
                "text": "Super Bowl 50 was an American football game. The NFL team that represented the AFC at Super Bowl 50 was the",
                "value": "Denver Broncos",
                "title": "Super_Bowl_50",
                "context": "Super Bowl 50 was an American football game.",
                "question": "Which NFL team represented the AFC at Super Bowl 50?",
            }
        ]
    )
    monkeypatch.setattr(
        squad_completion_module,
        "load_dataset",
        lambda *args, **kwargs: dataset,
    )

    result = evalution.benchmarks.squad_completion(max_rows=1, batch_size=1).evaluate(FakeSession())

    assert result.name == "squad_completion"
    assert result.metrics == {"contains": 1.0}
    assert result.metadata == {
        "dataset_path": "hazyresearch/based-squad",
        "dataset_name": "default",
        "split": "validation",
        "order": "native",
        "stream": True,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_contains_match",
        "primary_metric": "contains",
        "prompt_variant": "truncated_context_completion",
    }

    sample = result.samples[0]
    assert sample.target == "Denver Broncos"
    assert sample.prediction == "Denver Broncos\n"
    assert sample.extracted == {
        "contains-target": "1",
        "target": "Denver Broncos",
        "target-matched": "1",
    }
    assert sample.metadata == {
        "doc_id": "56be4db0acb8001400a502ec",
        "title": "Super_Bowl_50",
        "question": "Which NFL team represented the AFC at Super Bowl 50?",
    }


def test_squad_completion_contains_match_is_case_insensitive() -> None:
    assert squad_completion_module._contains_target_prediction(
        "denver broncos closed it out",
        "Denver Broncos",
    )
