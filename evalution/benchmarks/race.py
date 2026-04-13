# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Keep benchmark defaults and public task ids explicit at module scope.
_RACE_CHOICE_LABELS = ["A", "B", "C", "D"]
_RACE_ANSWER_LABEL_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _parse_race_problems(serialized_problems: str) -> list[dict[str, Any]]:
    """Parse race problems."""
    return ast.literal_eval(serialized_problems)


def _race_answer_option(problem: dict[str, Any]) -> str:
    """Implement race answer option for this module."""
    return str(problem["options"][_RACE_ANSWER_LABEL_TO_INDEX[str(problem["answer"]).strip()]])


def _race_prompt(article: str, previous_problems: list[dict[str, Any]], question: str) -> str:
    """Implement race prompt for this module."""
    prompt = f"Article: {article}\n\n"
    for problem in previous_problems:
        if str(problem["question"])[-6:] == "  _  .":
            prompt += str(problem["question"])[-5:] + _race_answer_option(problem) + "\n"
        else:
            prompt += f"Question: {problem['question']}\n"
            prompt += f"Answer: {_race_answer_option(problem)}\n"
    prompt += question
    return prompt


def _load_race_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> Dataset:
    """Load race dataset. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    if dataset_path != "EleutherAI/race":
        raise ValueError(f"unsupported RACE dataset path: {dataset_path!r}")
    if dataset_name != "high":
        raise ValueError(f"unsupported RACE dataset name: {dataset_name!r}")

    article_dataset = load_dataset(
        dataset_path,
        dataset_name,
        split=split,
        cache_dir=cache_dir,
        streaming=stream,
    )
    if stream:
        raise ValueError("RACE flattening requires non-stream dataset materialization")

    flattened_rows: list[dict[str, Any]] = []
    for article_index, article_row in enumerate(article_dataset):
        article = str(article_row["article"])
        problems = _parse_race_problems(str(article_row["problems"]))
        for problem_index, problem in enumerate(problems):
            flattened_rows.append(
                {
                    "article_index": article_index,
                    "problem_index": problem_index,
                    "article": article,
                    "question": str(problem["question"]),
                    "answer": str(problem["answer"]).strip(),
                    "options": [str(option) for option in problem["options"]],
                    "previous_problems": [
                        {
                            "question": str(previous_problem["question"]),
                            "answer": str(previous_problem["answer"]).strip(),
                            "options": [str(option) for option in previous_problem["options"]],
                        }
                        for previous_problem in problems[:problem_index]
                    ],
                }
            )
    return Dataset.from_list(flattened_rows)


@dataclass(slots=True)
class RACE(BaseMultipleChoiceSuite):
    """Implement the race benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "EleutherAI/race"
    dataset_name: str | None = "high"
    split: str = "test"
    stream: bool = False

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return _load_race_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "race"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        return MultipleChoiceSample(
            index=index,
            prompt=_race_prompt(
                str(doc["article"]),
                list(doc["previous_problems"]),
                str(doc["question"]),
            ),
            choices=[str(option) for option in doc["options"]],
            gold_index=_RACE_ANSWER_LABEL_TO_INDEX[str(doc["answer"]).strip()],
            metadata={
                "article_index": int(doc["article_index"]),
                "problem_index": int(doc["problem_index"]),
                "question": str(doc["question"]),
                "choice_labels": list(_RACE_CHOICE_LABELS),
                "choice_texts": [str(option) for option in doc["options"]],
                "previous_problem_count": len(doc["previous_problems"]),
            },
        )


def race(**kwargs: Any) -> RACE:
    """Implement race for this module."""
    return RACE(**kwargs)
