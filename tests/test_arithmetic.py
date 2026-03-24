# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

arithmetic_module = importlib.import_module("evalution.benchmarks.arithmetic")


class FakeLoglikelihoodSession:
    def __init__(
        self,
        *,
        expected_prompt: str,
        expected_continuation: str,
        output: LoglikelihoodOutput,
    ) -> None:
        self.expected_prompt = expected_prompt
        self.expected_continuation = expected_continuation
        self.output = output

    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 4
        assert len(requests) == 1
        assert requests[0].context == self.expected_prompt
        assert requests[0].continuation == self.expected_continuation
        return [self.output]


@pytest.mark.parametrize(
    ("factory_name", "expected_dataset_name"),
    [
        ("arithmetic_1dc", "arithmetic_1dc"),
        ("arithmetic_2da", "arithmetic_2da"),
        ("arithmetic_2dm", "arithmetic_2dm"),
        ("arithmetic_2ds", "arithmetic_2ds"),
        ("arithmetic_3da", "arithmetic_3da"),
        ("arithmetic_3ds", "arithmetic_3ds"),
        ("arithmetic_4da", "arithmetic_4da"),
        ("arithmetic_4ds", "arithmetic_4ds"),
        ("arithmetic_5da", "arithmetic_5da"),
        ("arithmetic_5ds", "arithmetic_5ds"),
    ],
)
def test_arithmetic_factories_set_expected_variant_names(factory_name: str, expected_dataset_name: str) -> None:
    suite = getattr(evalution.benchmarks, factory_name)()

    assert suite.dataset_name == expected_dataset_name
    assert suite.task_name() == expected_dataset_name


def test_arithmetic_scores_loglikelihood_without_perplexity(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "context": "\n\nQ: What is 98 plus 45?\n\nA:",
                "completion": " 143",
            }
        ]
    )
    monkeypatch.setattr(arithmetic_module, "_load_arithmetic_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.arithmetic_2da(max_rows=1, batch_size=4).evaluate(
        FakeLoglikelihoodSession(
            expected_prompt="Question: What is 98 plus 45?\nAnswer:",
            expected_continuation=" 143",
            output=LoglikelihoodOutput(
                logprob=-0.25,
                is_greedy=True,
                token_count=1,
            ),
        )
    )

    assert result.name == "arithmetic_2da"
    assert result.metrics == {"acc,ll": 1.0}
    assert result.metadata == {
        "dataset_path": "EleutherAI/arithmetic",
        "dataset_name": "arithmetic_2da",
        "split": "validation",
        "stream": False,
        "scoring_mode": "single_continuation_loglikelihood",
    }

    sample = result.samples[0]
    assert sample.prompt == "Question: What is 98 plus 45?\nAnswer:"
    assert sample.target == " 143"
    assert sample.prediction == " 143"
    assert sample.extracted == {
        "greedy_match": "1",
        "token_count": "1",
    }
    assert sample.scores == {"acc,ll": 1.0}
    assert sample.metadata == {
        "variant": "arithmetic_2da",
        "source_file": "data/two_digit_addition.jsonl",
        "raw_context": "\n\nQ: What is 98 plus 45?\n\nA:",
        "raw_completion": " 143",
        "logprob": -0.25,
        "token_count": 1,
        "is_greedy": True,
    }


def test_normalize_arithmetic_context_matches_upstream_dataset_script() -> None:
    assert arithmetic_module._normalize_arithmetic_context("\n\nQ: What is (9 + 8) * 2?\n\nA:") == (
        "Question: What is (9 + 8) * 2?\nAnswer:"
    )
