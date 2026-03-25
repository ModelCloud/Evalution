# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

record_module = importlib.import_module("evalution.benchmarks.record")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 7
        assert len(requests) == 3
        assert requests[0].context == (
            "record query: @placeholder won the race. "
            "entities: Alice, Bob, Carol "
            "passage: Alice trained every day. Bob watched from the stands. Carol arrived late"
        )
        assert requests[0].continuation == " Alice"
        assert requests[1].continuation == " Bob"
        assert requests[2].continuation == " Carol"
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
        ]


def test_record_scores_em_and_f1_over_answer_aliases(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "passage": "Alice trained every day.\n@highlight\nBob watched from the stands\n@highlight\nCarol arrived late",
                "query": "@placeholder won the race.",
                "entities": ["Alice", "Bob", "Carol"],
                "answers": ["Alice", "Alice Smith"],
                "idx": {"passage": 1, "query": 2},
            }
        ]
    )
    monkeypatch.setattr(record_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.record(max_rows=1, batch_size=7).evaluate(FakeSession())

    assert result.name == "record"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
        "em": 1.0,
        "f1": 1.0,
    }
    assert result.metadata["dataset_path"] == "super_glue"
    assert result.metadata["dataset_name"] == "record"
    assert result.metadata["split"] == "validation"
    assert result.metadata["primary_metric"] == "f1"

    sample = result.samples[0]
    assert sample.target == "Alice"
    assert sample.metadata["answers"] == ["Alice", "Alice Smith"]
    assert sample.metadata["entities"] == ["Alice", "Bob", "Carol"]


def test_record_helpers_match_upstream_prompt_cleanup() -> None:
    passage = 'Lead sentence.\n@highlight\nSecond item\n@highlight\nThird item'
    assert record_module._record_passage(passage) == "Lead sentence. Second item. Third item"
    assert record_module._record_prompt(
        {
            "passage": passage,
            "query": "@placeholder won.",
            "entities": ["Alice", "Bob"],
        }
    ) == "record query: @placeholder won. entities: Alice, Bob passage: Lead sentence. Second item. Third item"
