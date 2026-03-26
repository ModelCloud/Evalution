# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pcre
from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationRequest, GenerationOutput
from evalution.results import SampleResult

MULTIRC_DATASET_PATH = "super_glue"
MULTIRC_DATASET_NAME = "multirc"
MULTIRC_SPLIT = "validation"


def _group_questions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        p_idx = int(row["idx"]["paragraph"])
        q_idx = int(row["idx"]["question"])
        a_idx = int(row["idx"]["answer"])
        key = (p_idx, q_idx)
        if key not in grouped:
            grouped[key] = {
                "paragraph": row["paragraph"],
                "question": row["question"],
                "answers": [],
                "paragraph_idx": p_idx,
                "question_idx": q_idx,
            }
        grouped[key]["answers"].append(
            {"text": row["answer"], "label": int(row["label"]), "answer_idx": a_idx}
        )
    return sorted(grouped.values(), key=lambda d: (d["paragraph_idx"], d["question_idx"]))


def _extract_indices(text: str, num_answers: int) -> set[int]:
    # Pull out all integers and clamp to valid range; allow "none".
    if "none" in text.lower():
        return set()
    matches = {int(m) for m in pcre.findall(r"\d+", text)}
    return {m for m in matches if 0 <= m < num_answers}


def _precision_recall_f1(pred: set[int], gold: set[int]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold) if gold else 0.0
    denom = precision + recall
    f1 = 0.0 if denom == 0.0 else 2 * precision * recall / denom
    return precision, recall, f1


@dataclass(slots=True)
class MultiRC(BaseTestSuite):
    """
    MultiRC is a multi-sentence reading comprehension task with multiple correct answers.
    We score per-question exact match and F1 over selected answer options.
    """

    dataset_path: str = MULTIRC_DATASET_PATH
    dataset_name: str | None = MULTIRC_DATASET_NAME
    split: str = MULTIRC_SPLIT
    stream: bool = False
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    def task_name(self) -> str:
        return "multirc"

    def dataset_loader(self) -> Any:
        return load_dataset

    def requires_full_doc_materialization(self) -> bool:
        return True

    def result_metadata(self, *, generation_submission_mode: str) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "primary_metric": "em",
            "scoring_mode": "multi_label_extraction",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        questions = _group_questions(list(docs))
        if self.max_rows is not None:
            questions = questions[: self.max_rows]
        for index, q in enumerate(questions):
            options = "\n".join([f"{ans['answer_idx']}. {ans['text']}" for ans in q["answers"]])
            prompt = (
                "Read the passage and answer the question. "
                "Return the indices of all correct answers as a comma-separated list "
                "in ascending order. If none are correct, return \"none\". "
                "Reply with numbers only.\n\n"
                f"Passage:\n{q['paragraph']}\n\n"
                f"Question: {q['question']}\n"
                "Answer options:\n"
                f"{options}\n"
                "Indices:"
            )
            yield PreparedSample(
                index=index,
                doc=q,
                target=",".join(str(ans["answer_idx"]) for ans in q["answers"] if ans["label"] == 1) or "none",
                request=GenerationRequest(
                    prompt=prompt,
                    stop=[],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def _score_question(self, prediction: str, doc: dict[str, Any]) -> tuple[float, float]:
        gold = {ans["answer_idx"] for ans in doc["answers"] if ans["label"] == 1}
        pred = _extract_indices(prediction, len(doc["answers"]))
        em = 1.0 if pred == gold else 0.0
        _, _, f1 = _precision_recall_f1(pred, gold)
        return em, f1

    def score_sample(self, prepared_sample: PreparedSample, output: GenerationOutput) -> SampleResult:
        em, f1 = self._score_question(output.text, prepared_sample.doc)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "em": em,
                "f1a": f1,
                "indices": list(_extract_indices(output.text, len(prepared_sample.doc["answers"]))),
            },
            scores={"em": em, "f1a": f1},
            metadata={
                "paragraph_idx": prepared_sample.doc["paragraph_idx"],
                "question_idx": prepared_sample.doc["question_idx"],
                "num_answers": len(prepared_sample.doc["answers"]),
            },
        )


def multirc(**kwargs: Any) -> MultiRC:
    return MultiRC(**kwargs)
