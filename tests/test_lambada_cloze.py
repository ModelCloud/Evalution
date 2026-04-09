# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib
import math

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

lambada_cloze_module = importlib.import_module("evalution.benchmarks.lambada_cloze")


class FakeSession:
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
        assert batch_size == 6
        assert len(requests) == 1
        assert requests[0].context == self.expected_prompt
        assert requests[0].continuation == self.expected_continuation
        return [self.output]


@pytest.mark.parametrize(
    ("factory_name", "doc", "expected_dataset_path", "expected_dataset_name", "expected_prompt", "expected_target"),
    [
        (
            "lambada_openai_cloze",
            {"text": "I look at him, feeling stunned. Like this is some sort of sign"},
            "EleutherAI/lambada_openai",
            "default",
            "I look at him, feeling stunned. Like this is some sort of ____. ->",
            " sign",
        ),
        (
            "lambada_standard_cloze",
            {
                "text": "i look at him , feeling stunned . like this is some sort of sign",
                "domain": "fiction",
            },
            "cimec/lambada",
            None,
            "i look at him , feeling stunned . like this is some sort of ____. ->",
            " sign",
        ),
    ],
)
def test_lambada_cloze_scores_single_continuation_loglikelihood(
    monkeypatch,
    factory_name: str,
    doc: dict[str, str],
    expected_dataset_path: str,
    expected_dataset_name: str | None,
    expected_prompt: str,
    expected_target: str,
) -> None:
    dataset = Dataset.from_list([doc])
    monkeypatch.setattr(lambada_cloze_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = getattr(evalution.benchmarks, factory_name)(max_rows=1, batch_size=6).evaluate(
        FakeSession(
            expected_prompt=expected_prompt,
            expected_continuation=expected_target,
            output=LoglikelihoodOutput(
                logprob=-0.10536051565782628,
                is_greedy=True,
                token_count=1,
            ),
        )
    )

    assert result.name == factory_name
    assert result.metrics == pytest.approx(
        {
            "acc,ll": 1.0,
            "ppl,ll": math.exp(0.10536051565782628),
        }
    )
    assert result.metadata == {
        "dataset_path": expected_dataset_path,
        "dataset_name": expected_dataset_name,
        "split": "test",
        "stream": False,
        "scoring_mode": "single_continuation_loglikelihood",
        "prompt_variant": "cloze",
    }

    sample = result.samples[0]
    assert sample.prompt == expected_prompt
    assert sample.target == expected_target
    assert sample.prediction == expected_target
    assert sample.extracted == {
        "greedy_match": "1",
        "token_count": "1",
    }
    assert sample.scores == pytest.approx(
        {
            "acc,ll": 1.0,
            "ppl,ll": math.exp(0.10536051565782628),
        }
    )
    assert sample.metadata["text"] == doc["text"]
    assert sample.metadata["target_token"] == "sign"
    assert sample.metadata["prompt_variant"] == "cloze"
    assert sample.metadata["logprob"] == pytest.approx(-0.10536051565782628)
    assert sample.metadata["token_count"] == 1
    assert sample.metadata["is_greedy"] is True
    if "domain" in doc:
        assert sample.metadata["domain"] == doc["domain"]


def test_lambada_cloze_marks_non_greedy_continuation_incorrect(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "text": "The detective paused and finally saw the hidden clue",
            }
        ]
    )
    monkeypatch.setattr(lambada_cloze_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.lambada_openai_cloze(max_rows=1, batch_size=6).evaluate(
        FakeSession(
            expected_prompt="The detective paused and finally saw the hidden ____. ->",
            expected_continuation=" clue",
            output=LoglikelihoodOutput(
                logprob=-1.5,
                is_greedy=False,
                token_count=1,
            ),
        )
    )

    assert result.metrics == pytest.approx(
        {
            "acc,ll": 0.0,
            "ppl,ll": math.exp(1.5),
        }
    )
    assert result.samples[0].prediction == "[not-greedy]"
    assert result.samples[0].extracted["greedy_match"] == "0"
