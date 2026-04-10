# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

groundcocoa_module = importlib.import_module("evalution.benchmarks.groundcocoa")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 5
        assert len(requests) == 5
        assert "User Criteria: Need a direct business-class flight to Paris." in requests[0].context
        assert [request.continuation for request in requests] == [
            " The answer is Option A",
            " The answer is Option B",
            " The answer is Option C",
            " The answer is Option D",
            " The answer is Option E",
        ]
        return [
            LoglikelihoodOutput(logprob=-4.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.5, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-3.5, is_greedy=False, token_count=1),
        ]


def test_groundcocoa_scores_flight_selection(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "v1",
                "query": "Need a direct business-class flight to Paris.",
                "Option A": "{'Airline': 'A', 'Ticket Class': 'Economy'}",
                "Option B": "{'Airline': 'B', 'Ticket Class': 'Business'}",
                "Option C": "{'Airline': 'C', 'Ticket Class': 'Economy'}",
                "Option D": "{'Airline': 'D', 'Ticket Class': 'Economy'}",
                "Option E": "{'Airline': 'E', 'Ticket Class': 'Economy'}",
                "Answer": "B",
                "query_pos": "TicketClass",
                "is_typical": True,
            }
        ]
    )
    monkeypatch.setattr(groundcocoa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.groundcocoa(max_rows=1, batch_size=5, stream=False).evaluate(FakeSession())

    assert result.name == "groundcocoa"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "harsh147/GroundCocoa"
    assert result.metadata["prompt_variant"] == "flight_criteria_with_option_labels"

    sample = result.samples[0]
    assert sample.target == "The answer is Option B"
    assert sample.prediction == "The answer is Option B"
    assert sample.metadata["id"] == "v1"
    assert sample.metadata["query_pos"] == "TicketClass"
    assert sample.metadata["is_typical"] is True


def test_groundcocoa_rejects_unknown_answer_label() -> None:
    try:
        groundcocoa_module._groundcocoa_gold_index("Z")
    except ValueError as exc:
        assert "unsupported GroundCocoa answer label" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported answer label")
