# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput, LoglikelihoodOutput

asdiv_module = importlib.import_module("evalution.benchmarks.asdiv")


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
        assert batch_size == 6
        assert len(requests) == 1
        assert requests[0].context == self.expected_prompt
        assert requests[0].continuation == self.expected_continuation
        return [self.output]


class FakeGenerationSession:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        assert batch_size in {1, 5}
        self.requests.extend(requests)
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]


def test_asdiv_scores_single_continuation_loglikelihood_without_prefix_space(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "body": "Seven red apples and two green apples are in the basket.",
                "question": "How many apples are in the basket?",
                "solution_type": "Addition",
                "answer": "9 (apples)",
                "formula": "7+2=9",
            }
        ]
    )
    monkeypatch.setattr(asdiv_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.asdiv(max_rows=1, batch_size=6).evaluate(
        FakeLoglikelihoodSession(
            expected_prompt=(
                "Seven red apples and two green apples are in the basket.\n"
                "Question:How many apples are in the basket?\n"
                "Answer:"
            ),
            expected_continuation="9",
            output=LoglikelihoodOutput(
                logprob=-0.75,
                is_greedy=True,
                token_count=1,
            ),
        )
    )

    assert result.name == "asdiv"
    assert result.metrics == {"acc,ll": 1.0}
    assert result.metadata == {
        "dataset_path": "EleutherAI/asdiv",
        "dataset_name": None,
        "split": "validation",
        "stream": False,
        "scoring_mode": "single_continuation_loglikelihood",
    }

    sample = result.samples[0]
    assert sample.prompt == (
        "Seven red apples and two green apples are in the basket.\n"
        "Question:How many apples are in the basket?\n"
        "Answer:"
    )
    assert sample.target == "9"
    assert sample.prediction == "9"
    assert sample.extracted == {
        "greedy_match": "1",
        "token_count": "1",
    }
    assert sample.scores == {"acc,ll": 1.0}
    assert sample.metadata == {
        "body": "Seven red apples and two green apples are in the basket.",
        "question": "How many apples are in the basket?",
        "answer": "9 (apples)",
        "solution_type": "Addition",
        "formula": "7+2=9",
        "logprob": -0.75,
        "token_count": 1,
        "is_greedy": True,
    }


def test_asdiv_cot_llama_scores_numeric_generation_with_chat_fewshots(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "body": "Seven red apples and two green apples are in the basket.",
                "question": "How many apples are in the basket?",
                "solution_type": "Addition",
                "answer": "9 (apples)",
                "formula": "7+2=9",
            }
        ]
    )
    monkeypatch.setattr(asdiv_module, "load_dataset", lambda *args, **kwargs: dataset)

    session = FakeGenerationSession(["Reason through it carefully. The final answer is 9"])
    result = evalution.benchmarks.asdiv_cot_llama(
        max_rows=1,
        batch_size=5,
        apply_chat_template=True,
    ).evaluate(session)

    assert result.name == "asdiv_cot_llama"
    assert result.metrics == {"acc,num": 1.0}
    assert result.metadata == {
        "dataset_path": "EleutherAI/asdiv",
        "dataset_name": None,
        "split": "validation",
        "stream": False,
        "order": "native",
        "generation_submission_mode": "fixed_batches",
        "variant": "cot_llama",
        "num_fewshot": 8,
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "scoring_mode": "numeric_format_insensitive",
        "primary_metric": "acc,num",
    }

    request = session.requests[0]
    assert request.messages is not None
    assert len(request.messages) == 17
    assert request.messages[-1] == {
        "role": "user",
        "content": (
            "Given the following problem, reason and give a final answer to the problem.\n"
            "Problem: Seven red apples and two green apples are in the basket. How many apples are in the basket?\n"
            'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
        ),
    }

    sample = result.samples[0]
    assert "Given the following problem" in sample.prompt
    assert sample.target == "9"
    assert sample.prediction == "Reason through it carefully. The final answer is 9"
    assert sample.extracted == {"numeric-extract": "9"}
    assert sample.scores == {"acc,num": 1.0}
    assert sample.metadata == {
        "solution_type": "Addition",
        "formula": "7+2=9",
    }


def test_asdiv_numeric_target_strips_parenthesized_units() -> None:
    assert asdiv_module._asdiv_numeric_target({"answer": "15 (balls)"}) == "15"
