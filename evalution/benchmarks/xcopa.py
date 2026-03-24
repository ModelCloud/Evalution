# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_SUPPORTED_LANGUAGES = ("et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh")


def _xcopa_relation_text(question: str) -> str:
    return {
        "cause": "cause",
        "effect": "effect",
    }[question]


def _xcopa_prompt(
    premise: str,
    question: str,
    choice1: str,
    choice2: str,
) -> str:
    relation = _xcopa_relation_text(question)
    return (
        f"Premise: {premise.strip()}\n"
        f"Question: Which option is the more likely {relation}?\n"
        f"A. {choice1.strip()}\n"
        f"B. {choice2.strip()}\n"
        "Answer:"
    )


@dataclass(slots=True)
class XCOPA(BaseMultipleChoiceSuite):
    dataset_path: str = "xcopa"
    dataset_name: str | None = "it"
    split: str = "validation"
    language: str = "it"

    def __post_init__(self) -> None:
        if self.language not in _SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported xcopa language: {self.language!r}")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
            return
        raise ValueError("xcopa dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"xcopa_{self.language}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        prompt = _xcopa_prompt(
            str(doc["premise"]),
            str(doc["question"]),
            str(doc["choice1"]),
            str(doc["choice2"]),
        )
        return MultipleChoiceSample(
            index=index,
            prompt=prompt,
            choices=["A", "B"],
            gold_index=int(doc["label"]),
            metadata={
                "idx": int(doc["idx"]),
                "language": self.language,
                "question": str(doc["question"]),
                "premise": str(doc["premise"]).strip(),
                "raw_choices": [
                    str(doc["choice1"]).strip(),
                    str(doc["choice2"]).strip(),
                ],
                "changed": bool(doc["changed"]),
            },
        )


def xcopa(*, language: str, **kwargs: Any) -> XCOPA:
    return XCOPA(language=language, dataset_name=language, **kwargs)


def xcopa_et(**kwargs: Any) -> XCOPA:
    return xcopa(language="et", **kwargs)


def xcopa_ht(**kwargs: Any) -> XCOPA:
    return xcopa(language="ht", **kwargs)


def xcopa_id(**kwargs: Any) -> XCOPA:
    return xcopa(language="id", **kwargs)


def xcopa_it(**kwargs: Any) -> XCOPA:
    return xcopa(language="it", **kwargs)


def xcopa_qu(**kwargs: Any) -> XCOPA:
    return xcopa(language="qu", **kwargs)


def xcopa_sw(**kwargs: Any) -> XCOPA:
    return xcopa(language="sw", **kwargs)


def xcopa_ta(**kwargs: Any) -> XCOPA:
    return xcopa(language="ta", **kwargs)


def xcopa_th(**kwargs: Any) -> XCOPA:
    return xcopa(language="th", **kwargs)


def xcopa_tr(**kwargs: Any) -> XCOPA:
    return xcopa(language="tr", **kwargs)


def xcopa_vi(**kwargs: Any) -> XCOPA:
    return xcopa(language="vi", **kwargs)


def xcopa_zh(**kwargs: Any) -> XCOPA:
    return xcopa(language="zh", **kwargs)
