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
webqs_module = importlib.import_module("evalution.benchmarks.webqs")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self, outputs: list[LoglikelihoodOutput]) -> None:
        """Initialize this object."""
        self.outputs = outputs
        self.requests = []

    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        self.requests.extend(requests)
        assert batch_size == 5
        return list(self.outputs)


def test_webqs_scores_exact_match_when_any_alias_is_greedy(monkeypatch) -> None:
    """Verify webqs scores exact match when any alias is greedy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "url": "http://www.freebase.com/view/en/jamaica",
                "question": "what does jamaican people speak?",
                "answers": ["Jamaican Creole English Language", "English"],
            }
        ]
    )
    monkeypatch.setattr(webqs_module, "load_dataset", lambda *args, **kwargs: dataset)

    session = FakeSession(
        [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=4),
        ]
    )
    result = evalution.benchmarks.webqs(max_rows=1, batch_size=5).evaluate(session)

    assert result.name == "webqs"
    assert result.metrics == {"em": 1.0}
    assert result.metadata["dataset_path"] == "web_questions"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "test"
    assert result.metadata["scoring_mode"] == "accepted_alias_greedy_exact_match"
    assert result.metadata["primary_metric"] == "em"

    assert len(session.requests) == 2
    assert session.requests[0].context == "Question: what does jamaican people speak?\nAnswer:"
    assert session.requests[0].continuation == " English"
    assert session.requests[1].continuation == " Jamaican Creole English Language"

    sample = result.samples[0]
    assert sample.prompt == "Question: what does jamaican people speak?\nAnswer:"
    assert sample.target == "English"
    assert sample.prediction == "English"
    assert sample.extracted == {
        "greedy_alias_index": "0",
        "highest_logprob_alias_index": "0",
    }
    assert sample.metadata["accepted_answers"] == [
        "English",
        "Jamaican Creole English Language",
    ]
    assert sample.metadata["choice_logprobs"] == [-0.1, -1.1]
    assert sample.metadata["choice_greedy"] == [True, False]
    assert sample.metadata["greedy_alias_indices"] == [0]


def test_webqs_marks_non_greedy_alias_set_incorrect(monkeypatch) -> None:
    """Verify webqs marks non greedy alias set incorrect."""
    dataset = Dataset.from_list(
        [
            {
                "url": "http://www.freebase.com/view/en/jamaica",
                "question": "what does jamaican people speak?",
                "answers": ["Jamaican Creole English Language", "English"],
            }
        ]
    )
    monkeypatch.setattr(webqs_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.webqs(max_rows=1, batch_size=5).evaluate(
        FakeSession(
            [
                LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
                LoglikelihoodOutput(logprob=-0.1, is_greedy=False, token_count=4),
            ]
        )
    )

    assert result.metrics == {"em": 0.0}
    assert result.samples[0].prediction == "Jamaican Creole English Language"
    assert result.samples[0].extracted["greedy_alias_index"] == "[none]"
    assert result.samples[0].extracted["highest_logprob_alias_index"] == "1"


def test_webqs_removes_prefix_aliases_like_upstream() -> None:
    """Verify webqs removes prefix aliases like upstream."""
    assert webqs_module._remove_prefix_answers(
        [
            "Jamaican Creole English",
            "Jamaican",
            "Jamaican English",
        ]
    ) == ["Jamaican"]
