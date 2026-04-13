# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
prost_module = importlib.import_module("evalution.benchmarks.prost")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "A person is trying to roll an apple, a ball, a block, and a bottle.\n"
            "Question: Which is the hardest to roll?\n"
            "Answer:"
        )
        assert requests[0].continuation == " apple"
        assert requests[1].continuation == " ball"
        assert requests[2].continuation == " block"
        assert requests[3].continuation == " bottle"
        return [
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
        ]


def test_prost_scores_answer_text_multiple_choice(monkeypatch) -> None:
    """Verify prost scores answer text multiple choice. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "A": "apple",
                "B": "ball",
                "C": "block",
                "D": "bottle",
                "context": "A person is trying to roll an apple, a ball, a block, and a bottle.",
                "ex_question": "Which is the hardest to roll?",
                "group": "rolling",
                "label": 2,
                "name": "nonrolling_3",
                "question": "The [MASK] is the hardest to roll.",
            }
        ]
    )
    monkeypatch.setattr(prost_module, "_load_prost_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.prost(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "prost"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "corypaik/prost"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "test"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.target == "block"
    assert sample.prediction == "block"
    assert sample.extracted == {
        "gold_index": "2",
        "predicted_index": "2",
        "predicted_index_norm": "2",
    }
    assert sample.metadata["group"] == "rolling"
    assert sample.metadata["name"] == "nonrolling_3"
    assert sample.metadata["question"] == "The [MASK] is the hardest to roll."
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_texts"] == ["apple", "ball", "block", "bottle"]


def test_prost_loader_reads_raw_jsonl_via_json_builder() -> None:
    """Verify prost loader reads raw jsonl via JSON builder."""
    dataset = prost_module._load_prost_dataset("corypaik/prost", split="test", streaming=False)

    assert dataset == "prost-dataset"
    assert captured == {
        "args": ("json",),
        "kwargs": {
            "data_files": {"test": "https://huggingface.co/datasets/corypaik/prost/resolve/main/data/default.jsonl"},
            "split": "test",
            "cache_dir": None,
            "streaming": False,
        },
    }
