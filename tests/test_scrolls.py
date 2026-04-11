# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
import json
import zipfile

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput, LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
scrolls_module = importlib.import_module("evalution.benchmarks.scrolls")


class MultipleChoiceSession:
    """Define the multiple choice session helper used by the surrounding tests."""
    def __init__(self, expected_context: str, expected_continuations: list[str]) -> None:
        """Initialize this object."""
        self.expected_context = expected_context
        self.expected_continuations = expected_continuations

    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for multiple choice session."""
        assert batch_size == 8
        request_items = list(requests)
        assert [request.context for request in request_items] == [self.expected_context] * len(request_items)
        assert [request.continuation for request in request_items] == self.expected_continuations
        return [
            LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-3.0, is_greedy=False, token_count=1),
        ][: len(request_items)]


class ContinuousSession:
    """Define the continuous session helper used by the surrounding tests."""
    def __init__(self, prediction: str, expected_prompt: str) -> None:
        """Initialize this object."""
        self.prediction = prediction
        self.expected_prompt = expected_prompt

    def generate_continuous(self, requests, *, batch_size=None):
        """Generate continuous."""
        assert batch_size == 4
        request_items = list(requests)
        assert len(request_items) == 1
        assert request_items[0][1].prompt == self.expected_prompt
        for item_id, request in request_items:
            yield item_id, GenerationOutput(prompt=request.prompt, text=self.prediction)


def test_scrolls_groups_duplicate_reference_rows() -> None:
    """Verify scrolls groups duplicate reference rows."""
    grouped = scrolls_module._group_scrolls_outputs(
        Dataset.from_list(
            [
                {"id": "a", "pid": "p1", "input": "Question\n\nText", "output": "first"},
                {"id": "a", "pid": "p1", "input": "Question\n\nText", "output": "second"},
                {"id": "b", "pid": "p2", "input": "Another\n\nText", "output": "only"},
            ]
        )
    )

    rows = list(grouped)
    assert rows == [
        {"id": "a", "pid": "p1", "input": "Question\n\nText", "outputs": ["first", "second"]},
        {"id": "b", "pid": "p2", "input": "Another\n\nText", "outputs": ["only"]},
    ]


def test_scrolls_loads_repo_zip_without_executing_dataset_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Verify scrolls loads repo zip without executing dataset script."""
    archive_path = tmp_path / "contract_nli.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "contract_nli/validation.jsonl",
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "row-1",
                            "pid": "pid-1",
                            "input": "Question\n\nText",
                            "output": "first",
                        }
                    ),
                    json.dumps(
                        {
                            "id": "row-1",
                            "pid": "pid-1",
                            "input": "Question\n\nText",
                            "output": "second",
                        }
                    ),
                ]
            ),
        )
    monkeypatch.setattr(scrolls_module, "hf_hub_download", lambda **_: str(archive_path))

    dataset = scrolls_module._load_scrolls_dataset(
        "tau/scrolls",
        "contract_nli",
        split="validation",
    )

    assert list(dataset) == [
        {
            "id": "row-1",
            "pid": "pid-1",
            "input": "Question\n\nText",
            "outputs": ["first", "second"],
        }
    ]


def test_scrolls_contractnli_scores_multiple_choice_rows(monkeypatch) -> None:
    """Verify scrolls contractnli scores multiple choice rows. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "id": "row-1",
                "pid": "pid-1",
                "input": "The agreement covers laptops.\n\nContract body text.",
                "outputs": ["Entailment"],
            }
        ]
    )
    monkeypatch.setattr(scrolls_module, "_load_scrolls_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.scrolls_contractnli(max_rows=1, batch_size=8).evaluate(
        MultipleChoiceSession(
            expected_context="Contract body text.\n\nHypothesis: The agreement covers laptops.\nConclusion:",
            expected_continuations=[" Not mentioned", " Entailment", " Contradiction"],
        )
    )

    assert result.name == "scrolls_contractnli"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata == {
        "dataset_path": "tau/scrolls",
        "dataset_name": "contract_nli",
        "split": "validation",
        "order": "native",
        "stream": False,
        "scoring_mode": "multiple_choice_loglikelihood",
    }
    sample = result.samples[0]
    assert sample.target == "Entailment"
    assert sample.metadata["variant"] == "contractnli"
    assert sample.metadata["outputs"] == ["Entailment"]


def test_scrolls_quality_parser_and_task_aliases() -> None:
    """Verify scrolls quality parser and task aliases."""
    choices, passage = scrolls_module._quality_choices_and_context(
        "(A) red (B) blue (C) green (D) yellow\n\nFull passage text."
    )
    assert choices == ["red", "blue", "green", "yellow"]
    assert passage == "Full passage text."

    suite = evalution.benchmarks.scrolls(subset="gov_report")
    assert suite.dataset_name == "gov_report"
    assert suite.task_name() == "scrolls_govreport"

    with pytest.raises(ValueError, match="unsupported scrolls variant"):
        evalution.benchmarks.scrolls(subset="unknown")


def test_scrolls_qasper_scores_qa_rows_against_multiple_references(monkeypatch) -> None:
    """Verify scrolls QASPER scores QA rows against multiple references. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "id": "row-2",
                "pid": "pid-2",
                "input": "What is the answer?\n\nLong paper body.",
                "outputs": ["first answer", "final answer"],
            }
        ]
    )
    monkeypatch.setattr(scrolls_module, "_load_scrolls_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.scrolls_qasper(max_rows=1, batch_size=4).evaluate(
        ContinuousSession(
            prediction="final answer",
            expected_prompt="Long paper body.\n\nQuestion: What is the answer?\nAnswer:",
        )
    )

    assert result.name == "scrolls_qasper"
    assert result.metrics == {"em": 1.0, "f1": 1.0}
    assert result.metadata["variant"] == "qasper"
    assert result.samples[0].metadata["outputs"] == ["first answer", "final answer"]


def test_scrolls_govreport_scores_summary_rows(monkeypatch) -> None:
    """Verify scrolls govreport scores summary rows. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "id": "row-3",
                "pid": "pid-3",
                "input": "A long report body.",
                "outputs": ["Short summary."],
            }
        ]
    )
    monkeypatch.setattr(scrolls_module, "_load_scrolls_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.scrolls_govreport(max_rows=1, batch_size=4).evaluate(
        ContinuousSession(
            prediction="Short summary.",
            expected_prompt=(
                "A long report body.\n\n"
                "Question: What is a summary of the preceding text?\n"
                "Answer:"
            ),
        )
    )

    assert result.name == "scrolls_govreport"
    assert result.metadata["variant"] == "govreport"
    assert result.metadata["primary_metric"] == "rougeLsum"
    assert result.metrics["rougeLsum"] == pytest.approx(1.0)
