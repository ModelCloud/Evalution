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
KOBEST_SUBSETS = ("boolq", "copa", "hellaswag", "sentineg", "wic")
KOBEST_TASKS = tuple(f"kobest_{subset}" for subset in KOBEST_SUBSETS)

_BOOL_CHOICES = ("아니오", "예")
_SENTINEG_CHOICES = ("부정", "긍정")
_COPA_CONNECTORS = {
    "cause": "왜냐하면",
    "effect": "그래서",
    "원인": "왜냐하면",
    "결과": "그래서",
}


def _format_question(question: str) -> str:
    """Format question."""
    stripped = question.strip()
    return stripped if stripped.endswith("?") else f"{stripped}?"


def _copa_prompt(premise: str, question: str) -> str:
    """Implement COPA prompt for this module."""
    prompt_stem = premise.strip().rstrip(".?!")
    connector = _COPA_CONNECTORS[question.strip()]
    return f"{prompt_stem} {connector}"


def _sentineg_prompt(sentence: str) -> str:
    """Implement sentineg prompt for this module."""
    return (
        f"문장: {sentence.strip()}\n"
        "질문: 이 문장의 감성은 무엇입니까?\n"
        "답변:"
    )


def _wic_prompt(*, word: str, context_1: str, context_2: str) -> str:
    """Implement wic prompt for this module."""
    return (
        f"문장 1: {context_1.strip()}\n"
        f"문장 2: {context_2.strip()}\n"
        f"질문: 두 문장에서 '{word.strip()}'의 의미가 같습니까?\n"
        "답변:"
    )


@dataclass(slots=True)
class KoBEST(BaseMultipleChoiceSuite):
    """Implement the ko best benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "skt/kobest_v1"
    dataset_name: str | None = None
    split: str = "test"
    subset: str = ""

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.subset not in KOBEST_SUBSETS:
            raise ValueError(f"unsupported kobest subset: {self.subset!r}")
        if self.dataset_name in {None, self.subset}:
            self.dataset_name = self.subset
            return
        raise ValueError("kobest dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"kobest_{self.subset}"

    def result_metadata(self) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        metadata = super().result_metadata()
        metadata["subset"] = self.subset
        return metadata

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row. Preserve the fallback order expected by the surrounding caller."""
        if self.subset == "boolq":
            return MultipleChoiceSample(
                index=index,
                prompt=(
                    f"지문: {str(doc['paragraph']).strip()}\n"
                    f"질문: {_format_question(str(doc['question']))}\n"
                    "답변:"
                ),
                choices=list(_BOOL_CHOICES),
                gold_index=int(doc["label"]),
                metadata={
                    "subset": self.subset,
                    "paragraph": str(doc["paragraph"]).strip(),
                    "question": str(doc["question"]).strip(),
                },
            )

        if self.subset == "copa":
            raw_choices = [
                str(doc["alternative_1"]).strip(),
                str(doc["alternative_2"]).strip(),
            ]
            return MultipleChoiceSample(
                index=index,
                prompt=_copa_prompt(str(doc["premise"]), str(doc["question"])),
                choices=raw_choices,
                gold_index=int(doc["label"]),
                metadata={
                    "subset": self.subset,
                    "premise": str(doc["premise"]).strip(),
                    "question": str(doc["question"]).strip(),
                    "raw_choices": raw_choices,
                },
            )

        if self.subset == "hellaswag":
            raw_choices = [
                str(doc["ending_1"]).strip(),
                str(doc["ending_2"]).strip(),
                str(doc["ending_3"]).strip(),
                str(doc["ending_4"]).strip(),
            ]
            return MultipleChoiceSample(
                index=index,
                prompt=str(doc["context"]).strip(),
                choices=raw_choices,
                gold_index=int(doc["label"]),
                metadata={
                    "subset": self.subset,
                    "context": str(doc["context"]).strip(),
                    "raw_choices": raw_choices,
                },
            )

        if self.subset == "sentineg":
            return MultipleChoiceSample(
                index=index,
                prompt=_sentineg_prompt(str(doc["sentence"])),
                choices=list(_SENTINEG_CHOICES),
                gold_index=int(doc["label"]),
                metadata={
                    "subset": self.subset,
                    "sentence": str(doc["sentence"]).strip(),
                },
            )

        if self.subset == "wic":
            return MultipleChoiceSample(
                index=index,
                prompt=_wic_prompt(
                    word=str(doc["word"]),
                    context_1=str(doc["context_1"]),
                    context_2=str(doc["context_2"]),
                ),
                choices=list(_BOOL_CHOICES),
                gold_index=int(doc["label"]),
                metadata={
                    "subset": self.subset,
                    "word": str(doc["word"]).strip(),
                    "context_1": str(doc["context_1"]).strip(),
                    "context_2": str(doc["context_2"]).strip(),
                },
            )

        raise AssertionError(f"unsupported kobest subset branch: {self.subset!r}")


def kobest(*, subset: str, **kwargs: Any) -> KoBEST:
    """Implement kobest for this module."""
    return KoBEST(subset=subset, **kwargs)


def kobest_boolq(**kwargs: Any) -> KoBEST:
    """Implement kobest boolq for this module."""
    return kobest(subset="boolq", **kwargs)


def kobest_copa(**kwargs: Any) -> KoBEST:
    """Implement kobest COPA for this module."""
    return kobest(subset="copa", **kwargs)


def kobest_hellaswag(**kwargs: Any) -> KoBEST:
    """Implement kobest hellaswag for this module."""
    return kobest(subset="hellaswag", **kwargs)


def kobest_sentineg(**kwargs: Any) -> KoBEST:
    """Implement kobest sentineg for this module."""
    return kobest(subset="sentineg", **kwargs)


def kobest_wic(**kwargs: Any) -> KoBEST:
    """Implement kobest wic for this module."""
    return kobest(subset="wic", **kwargs)
