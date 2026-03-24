# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

XSTORYCLOZE_LANGUAGES = (
    "ar",
    "en",
    "es",
    "eu",
    "hi",
    "id",
    "my",
    "ru",
    "sw",
    "te",
    "zh",
)


def _xstorycloze_prompt(doc: dict[str, Any]) -> str:
    return " ".join(
        str(doc[field]).strip()
        for field in (
            "input_sentence_1",
            "input_sentence_2",
            "input_sentence_3",
            "input_sentence_4",
        )
    )


@dataclass(slots=True)
class XStoryCloze(BaseMultipleChoiceSuite):
    dataset_path: str = "juletxara/xstory_cloze"
    dataset_name: str | None = "en"
    split: str = "eval"
    language: str = "en"

    def __post_init__(self) -> None:
        if self.language not in XSTORYCLOZE_LANGUAGES:
            raise ValueError(f"unsupported xstorycloze language: {self.language!r}")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
            return
        raise ValueError("xstorycloze dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"xstorycloze_{self.language}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [
            str(doc["sentence_quiz1"]).strip(),
            str(doc["sentence_quiz2"]).strip(),
        ]
        return MultipleChoiceSample(
            index=index,
            prompt=_xstorycloze_prompt(doc),
            choices=choices,
            gold_index=int(doc["answer_right_ending"]) - 1,
            metadata={
                "story_id": str(doc["story_id"]),
                "language": self.language,
                "input_sentences": [
                    str(doc["input_sentence_1"]).strip(),
                    str(doc["input_sentence_2"]).strip(),
                    str(doc["input_sentence_3"]).strip(),
                    str(doc["input_sentence_4"]).strip(),
                ],
                "choice_texts": choices,
            },
        )


def xstorycloze(*, language: str, **kwargs: Any) -> XStoryCloze:
    return XStoryCloze(language=language, dataset_name=language, **kwargs)


def xstorycloze_ar(**kwargs: Any) -> XStoryCloze:
    return xstorycloze(language="ar", **kwargs)


def xstorycloze_en(**kwargs: Any) -> XStoryCloze:
    return xstorycloze(language="en", **kwargs)


def xstorycloze_es(**kwargs: Any) -> XStoryCloze:
    return xstorycloze(language="es", **kwargs)


def xstorycloze_eu(**kwargs: Any) -> XStoryCloze:
    return xstorycloze(language="eu", **kwargs)


def xstorycloze_hi(**kwargs: Any) -> XStoryCloze:
    return xstorycloze(language="hi", **kwargs)


def xstorycloze_id(**kwargs: Any) -> XStoryCloze:
    return xstorycloze(language="id", **kwargs)


def xstorycloze_my(**kwargs: Any) -> XStoryCloze:
    return xstorycloze(language="my", **kwargs)


def xstorycloze_ru(**kwargs: Any) -> XStoryCloze:
    return xstorycloze(language="ru", **kwargs)


def xstorycloze_sw(**kwargs: Any) -> XStoryCloze:
    return xstorycloze(language="sw", **kwargs)


def xstorycloze_te(**kwargs: Any) -> XStoryCloze:
    return xstorycloze(language="te", **kwargs)


def xstorycloze_zh(**kwargs: Any) -> XStoryCloze:
    return xstorycloze(language="zh", **kwargs)
