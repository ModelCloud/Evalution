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
    """Implement xstorycloze prompt for this module."""
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
    """Implement the xstory cloze benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "juletxara/xstory_cloze"
    dataset_name: str | None = "en"
    split: str = "eval"
    language: str = "en"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.language not in XSTORYCLOZE_LANGUAGES:
            raise ValueError(f"unsupported xstorycloze language: {self.language!r}")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
            return
        raise ValueError("xstorycloze dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"xstorycloze_{self.language}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
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
    """Implement xstorycloze for this module."""
    return XStoryCloze(language=language, dataset_name=language, **kwargs)


def xstorycloze_ar(**kwargs: Any) -> XStoryCloze:
    """Implement xstorycloze ar for this module."""
    return xstorycloze(language="ar", **kwargs)


def xstorycloze_en(**kwargs: Any) -> XStoryCloze:
    """Implement xstorycloze en for this module."""
    return xstorycloze(language="en", **kwargs)


def xstorycloze_es(**kwargs: Any) -> XStoryCloze:
    """Implement xstorycloze es for this module."""
    return xstorycloze(language="es", **kwargs)


def xstorycloze_eu(**kwargs: Any) -> XStoryCloze:
    """Implement xstorycloze eu for this module."""
    return xstorycloze(language="eu", **kwargs)


def xstorycloze_hi(**kwargs: Any) -> XStoryCloze:
    """Implement xstorycloze hi for this module."""
    return xstorycloze(language="hi", **kwargs)


def xstorycloze_id(**kwargs: Any) -> XStoryCloze:
    """Implement xstorycloze id for this module."""
    return xstorycloze(language="id", **kwargs)


def xstorycloze_my(**kwargs: Any) -> XStoryCloze:
    """Implement xstorycloze my for this module."""
    return xstorycloze(language="my", **kwargs)


def xstorycloze_ru(**kwargs: Any) -> XStoryCloze:
    """Implement xstorycloze ru for this module."""
    return xstorycloze(language="ru", **kwargs)


def xstorycloze_sw(**kwargs: Any) -> XStoryCloze:
    """Implement xstorycloze sw for this module."""
    return xstorycloze(language="sw", **kwargs)


def xstorycloze_te(**kwargs: Any) -> XStoryCloze:
    """Implement xstorycloze te for this module."""
    return xstorycloze(language="te", **kwargs)


def xstorycloze_zh(**kwargs: Any) -> XStoryCloze:
    """Implement xstorycloze zh for this module."""
    return xstorycloze(language="zh", **kwargs)
