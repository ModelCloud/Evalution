# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Frozen upstream config snapshot for import safety. Refresh deliberately if the dataset adds
# or removes languages.
AFRIMMLU_LANGUAGES = (
    "amh",
    "eng",
    "ewe",
    "fra",
    "hau",
    "ibo",
    "kin",
    "lin",
    "lug",
    "orm",
    "sna",
    "sot",
    "swa",
    "twi",
    "wol",
    "xho",
    "yor",
    "zul",
)
AFRIMMLU_TASKS = tuple(f"afrimmlu_{language}" for language in AFRIMMLU_LANGUAGES)
_CHOICE_LABELS = ["A", "B", "C", "D"]


def _afrimmlu_prompt(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question.strip()}"]
    lines.extend(f"{label}. {choice}" for label, choice in zip(_CHOICE_LABELS, choices, strict=True))
    lines.append("Answer:")
    return "\n".join(lines)


def _parse_choices(raw_choices: Any) -> list[str]:
    if isinstance(raw_choices, list):
        return [str(choice).strip() for choice in raw_choices]
    parsed = ast.literal_eval(str(raw_choices))
    if not isinstance(parsed, list):
        raise ValueError("afrimmlu choices must decode to a list")
    return [str(choice).strip() for choice in parsed]


@dataclass(slots=True)
class AfriMMLU(BaseMultipleChoiceSuite):
    """AfriMMLU suite backed by a frozen language registry to keep imports offline-safe."""

    dataset_path: str = "masakhane/afrimmlu"
    dataset_name: str | None = "eng"
    split: str = "test"
    language: str = "eng"

    def __post_init__(self) -> None:
        if self.language not in AFRIMMLU_LANGUAGES:
            raise ValueError(f"unsupported afrimmlu language: {self.language!r}")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
            return
        raise ValueError("afrimmlu dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"afrimmlu_{self.language}"

    def result_metadata(self) -> dict[str, Any]:
        metadata = super().result_metadata()
        metadata["language"] = self.language
        return metadata

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = _parse_choices(doc["choices"])
        answer_label = str(doc["answer"]).strip().upper()
        return MultipleChoiceSample(
            index=index,
            prompt=_afrimmlu_prompt(str(doc["question"]), choices),
            choices=list(_CHOICE_LABELS),
            gold_index=_CHOICE_LABELS.index(answer_label),
            metadata={
                "language": self.language,
                "subject": str(doc["subject"]).strip(),
                "question": str(doc["question"]).strip(),
                "answer_label": answer_label,
                "choice_labels": list(_CHOICE_LABELS),
                "raw_choices": choices,
            },
        )


def afrimmlu(*, language: str, **kwargs: Any) -> AfriMMLU:
    kwargs.setdefault("dataset_name", language)
    return AfriMMLU(language=language, **kwargs)


def _make_afrimmlu_factory(language: str) -> Any:
    def factory(**kwargs: Any) -> AfriMMLU:
        return afrimmlu(language=language, **kwargs)

    factory.__name__ = f"afrimmlu_{language}"
    return factory


for _language in AFRIMMLU_LANGUAGES:
    globals()[f"afrimmlu_{_language}"] = _make_afrimmlu_factory(_language)

del _language
