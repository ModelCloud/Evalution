# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Keep benchmark defaults and public task ids explicit at module scope.
_QA4MRE_CONFIGS = {
    "2011": "2011.main.EN",
    "2012": "2012.main.EN",
    "2013": "2013.main.EN",
}


def _qa4mre_prompt(document: str, question: str) -> str:
    """Implement qa4mre prompt for this module."""
    return f"{document.strip()}\nQuestion: {question.strip()}\nAnswer:"


@dataclass(slots=True)
class QA4MRE(BaseMultipleChoiceSuite):
    """Implement the qa4 mre benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "qa4mre"
    dataset_name: str | None = "2011.main.EN"
    split: str = "train"
    year: str = "2011"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        expected_name = _QA4MRE_CONFIGS.get(self.year)
        if expected_name is None:
            raise ValueError(f"unsupported qa4mre year: {self.year!r}")
        if self.dataset_name in {None, expected_name}:
            self.dataset_name = expected_name
            return
        raise ValueError("qa4mre dataset_name must match the configured year")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"qa4mre_{self.year}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        choices = [str(choice).strip() for choice in doc["answer_options"]["answer_str"]]
        correct_answer_id = int(doc["correct_answer_id"]) - 1
        return MultipleChoiceSample(
            index=index,
            prompt=_qa4mre_prompt(str(doc["document_str"]), str(doc["question_str"])),
            choices=choices,
            gold_index=correct_answer_id,
            metadata={
                "year": self.year,
                "topic_id": str(doc["topic_id"]),
                "topic_name": str(doc["topic_name"]),
                "test_id": str(doc["test_id"]),
                "document_id": str(doc["document_id"]),
                "question_id": str(doc["question_id"]),
                "question": str(doc["question_str"]).strip(),
                "correct_answer_id": str(doc["correct_answer_id"]),
                "choice_texts": choices,
                "choice_labels": [str(answer_id) for answer_id in doc["answer_options"]["answer_id"]],
            },
        )


def qa4mre(*, year: str, **kwargs: Any) -> QA4MRE:
    """Implement qa4mre for this module."""
    return QA4MRE(year=year, dataset_name=_QA4MRE_CONFIGS[year], **kwargs)


def qa4mre_2011(**kwargs: Any) -> QA4MRE:
    """Implement qa4mre 2011 for this module."""
    return qa4mre(year="2011", **kwargs)


def qa4mre_2012(**kwargs: Any) -> QA4MRE:
    """Implement qa4mre 2012 for this module."""
    return qa4mre(year="2012", **kwargs)


def qa4mre_2013(**kwargs: Any) -> QA4MRE:
    """Implement qa4mre 2013 for this module."""
    return qa4mre(year="2013", **kwargs)
