# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

# Keep shared test fixtures and expectations explicit at module scope.
longbench_module = importlib.import_module("evalution.benchmarks.longbench")


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


def _row(
    *,
    context: str,
    question: str,
    answer_prefix: str,
    answers: list[str],
    dataset: str,
    task: str,
    language: str = "en",
    all_classes=None,
    max_new_tokens: str = "32",
) -> dict[str, object]:
    """Support the surrounding tests with row."""
    return {
        "context": context,
        "question": question,
        "answer_prefix": answer_prefix,
        "answers": answers,
        "all_classes": [] if all_classes is None else all_classes,
        "dataset": dataset,
        "task": task,
        "language": language,
        "length": "1234",
        "_id": f"{dataset}-row-1",
        "max_new_tokens": max_new_tokens,
    }


def test_longbench_scores_english_qa_rows(monkeypatch) -> None:
    """Verify longbench scores english QA rows. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            _row(
                context="Article body.\n\n",
                question="Question: What is the answer?\n\n",
                answer_prefix="Answer:",
                answers=["final answer", "backup answer"],
                dataset="qasper",
                task="qasper",
                max_new_tokens="148",
            )
        ]
    )
    monkeypatch.setattr(longbench_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.longbench_qasper(max_rows=1, batch_size=4).evaluate(
        ContinuousSession(
            prediction="final answer",
            expected_prompt="Article body.\n\nQuestion: What is the answer?\n\nAnswer:",
        )
    )

    assert result.name == "longbench_qasper"
    assert result.metrics == {"score": 1.0, "qa_f1_score": 1.0}
    assert result.metadata["dataset_path"] == "Xnhyacinth/LongBench"
    assert result.metadata["dataset_name"] == "qasper"
    assert result.metadata["task_root"] == "qasper"
    assert result.metadata["metric_name"] == "qa_f1_score"
    assert result.samples[0].metadata["answers"] == ["final answer", "backup answer"]


def test_longbench_scores_chinese_qa_rows(monkeypatch) -> None:
    """Verify longbench scores chinese QA rows. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            _row(
                context="阅读下面的文章。\n\n",
                question="问题：第十八届年会在哪里举办？\n",
                answer_prefix="回答：",
                answers=["厦门大学。"],
                dataset="multifieldqa_zh",
                task="multifieldqa_zh",
                language="zh",
                max_new_tokens="84",
            )
        ]
    )
    monkeypatch.setattr(longbench_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.longbench_multifieldqa_zh(max_rows=1, batch_size=4).evaluate(
        ContinuousSession(
            prediction="厦门大学。",
            expected_prompt="阅读下面的文章。\n\n问题：第十八届年会在哪里举办？\n回答：",
        )
    )

    assert result.name == "longbench_multifieldqa_zh"
    assert result.metrics == {"score": 1.0, "qa_f1_zh_score": 1.0}
    assert result.metadata["metric_name"] == "qa_f1_zh_score"
    assert result.samples[0].metadata["language"] == "zh"


def test_longbench_samsum_trims_first_line_and_avoids_duplicate_prefix(monkeypatch) -> None:
    """Verify longbench samsum trims first line and avoids duplicate prefix."""
    dataset = Dataset.from_list(
        [
            _row(
                context="Few-shot examples.\n\n",
                question="Dialogue: Hi there\nSummary: ",
                answer_prefix="Summary:",
                answers=["Short summary."],
                dataset="samsum",
                task="samsum",
                max_new_tokens="148",
            )
        ]
    )
    monkeypatch.setattr(longbench_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.longbench_samsum(max_rows=1, batch_size=4).evaluate(
        ContinuousSession(
            prediction="Short summary.\nIgnored extra line",
            expected_prompt="Few-shot examples.\n\nDialogue: Hi there\nSummary: ",
        )
    )

    assert result.name == "longbench_samsum"
    assert result.metrics["score"] == pytest.approx(1.0)
    assert result.metrics["rouge_score"] == pytest.approx(1.0)
    assert result.samples[0].extracted["prediction-scored"] == "Short summary."


def test_longbench_scores_classification_rows(monkeypatch) -> None:
    """Verify longbench scores classification rows. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            _row(
                context="Question: Example?\nType: Individual\n",
                question="Question: Where was the first golf course in the United States ?\n",
                answer_prefix="Type:",
                answers=["Other location"],
                all_classes=["Other location", "Location", "Individual"],
                dataset="trec",
                task="trec",
                max_new_tokens="84",
            )
        ]
    )
    monkeypatch.setattr(longbench_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.longbench_trec(max_rows=1, batch_size=4).evaluate(
        ContinuousSession(
            prediction="Other location\nextra line",
            expected_prompt=(
                "Question: Example?\nType: Individual\n"
                "Question: Where was the first golf course in the United States ?\n"
                "Type:"
            ),
        )
    )

    assert result.name == "longbench_trec"
    assert result.metrics == {"score": 1.0, "classification_score": 1.0}
    assert result.samples[0].metadata["all_classes"] == ["Other location", "Location", "Individual"]


@pytest.mark.parametrize(
    ("factory_name", "dataset_name", "prediction", "metric_name", "expected_prompt"),
    [
        (
            "longbench_passage_retrieval_en",
            "passage_retrieval_en",
            "Paragraph 15",
            "retrieval_score",
            "Paragraphs...\nAbstract text\nThe answer is: ",
        ),
        (
            "longbench_passage_count",
            "passage_count",
            "2",
            "count_score",
            "Paragraphs...\nHow many unique paragraphs are there?\nAnswer:",
        ),
        (
            "longbench_repobench_p",
            "repobench-p",
            "// comment\n    return value",
            "code_sim_score",
            "Code context\nCompletion prefix\nNext line of code:\n",
        ),
    ],
)
def test_longbench_scores_specialized_modes(
    monkeypatch,
    factory_name: str,
    dataset_name: str,
    prediction: str,
    metric_name: str,
    expected_prompt: str,
) -> None:
    """Verify longbench scores specialized modes. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    row = _row(
        context="Paragraphs...\n" if dataset_name != "repobench-p" else "Code context\n",
        question=(
            "Abstract text\n"
            if dataset_name == "passage_retrieval_en"
            else "How many unique paragraphs are there?\n"
            if dataset_name == "passage_count"
            else "Completion prefix\n"
        ),
        answer_prefix=(
            "The answer is: "
            if dataset_name == "passage_retrieval_en"
            else "Answer:"
            if dataset_name == "passage_count"
            else "Next line of code:\n"
        ),
        answers=(
            ["Paragraph 15"]
            if dataset_name == "passage_retrieval_en"
            else ["2"]
            if dataset_name == "passage_count"
            else ["    return value"]
        ),
        dataset=dataset_name,
        task=dataset_name,
    )
    dataset = Dataset.from_list([row])
    monkeypatch.setattr(longbench_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = getattr(evalution.benchmarks, factory_name)(max_rows=1, batch_size=4).evaluate(
        ContinuousSession(
            prediction=prediction,
            expected_prompt=expected_prompt,
        )
    )

    assert result.metrics["score"] == pytest.approx(1.0)
    assert result.metrics[metric_name] == pytest.approx(1.0)


def test_longbench_dispatcher_accepts_base_e_and_hyphen_aliases() -> None:
    """Verify longbench dispatcher accepts base e and hyphen aliases."""
    suite = evalution.benchmarks.longbench(subset="repobench-p_e")
    assert suite.dataset_name == "repobench-p_e"
    assert suite.task_name() == "longbench_repobench_p_e"

    hyphen_suite = evalution.benchmarks.longbench(subset="longbench_repobench-p")
    assert hyphen_suite.dataset_name == "repobench-p"
    assert hyphen_suite.task_name() == "longbench_repobench_p"

    with pytest.raises(ValueError, match="unsupported longbench subset"):
        evalution.benchmarks.longbench(subset="unknown")
