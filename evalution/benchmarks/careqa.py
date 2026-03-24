# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

CAREQA_CONFIGS = {
    "en": "CareQA_en",
    "es": "CareQA_es",
}
CAREQA_TASKS = tuple(f"careqa_{language}" for language in CAREQA_CONFIGS)
_CHOICE_LABELS = ("A", "B", "C", "D")


def _careqa_prompt(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question.strip()}"]
    lines.extend(
        f"{label}. {choice.strip()}"
        for label, choice in zip(_CHOICE_LABELS, choices, strict=True)
    )
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass(slots=True)
class CareQA(BaseMultipleChoiceSuite):
    dataset_path: str = "HPAI-BSC/CareQA"
    dataset_name: str | None = "CareQA_en"
    split: str = "test"
    language: str = "en"

    def __post_init__(self) -> None:
        if self.language not in CAREQA_CONFIGS:
            raise ValueError(f"unsupported careqa language: {self.language!r}")
        expected_dataset_name = CAREQA_CONFIGS[self.language]
        if self.dataset_name in {None, expected_dataset_name}:
            self.dataset_name = expected_dataset_name
            return
        raise ValueError("careqa dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"careqa_{self.language}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [str(doc[field]).strip() for field in ("op1", "op2", "op3", "op4")]
        gold_index = int(doc["cop"]) - 1
        return MultipleChoiceSample(
            index=index,
            prompt=_careqa_prompt(str(doc["question"]), choices),
            choices=list(_CHOICE_LABELS),
            gold_index=gold_index,
            metadata={
                "language": self.language,
                "category": str(doc["category"]).strip(),
                "exam_id": int(doc["exam_id"]),
                "year": int(doc["year"]),
                "unique_id": str(doc["unique_id"]),
                "raw_choices": choices,
            },
        )


def careqa(*, language: str, **kwargs: Any) -> CareQA:
    if language not in CAREQA_CONFIGS:
        raise ValueError(f"unsupported careqa language: {language!r}")
    kwargs.setdefault("dataset_name", CAREQA_CONFIGS[language])
    return CareQA(language=language, **kwargs)


def careqa_en(**kwargs: Any) -> CareQA:
    return careqa(language="en", **kwargs)


def careqa_es(**kwargs: Any) -> CareQA:
    return careqa(language="es", **kwargs)
