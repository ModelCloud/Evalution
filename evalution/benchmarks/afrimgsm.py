# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.gsm8k_common import BaseGSM8KSuite, GSM8KVariant, VariantSpec

# Frozen upstream config snapshot for import safety. Refresh deliberately if the dataset adds
# or removes languages.
AFRIMGSM_LANGUAGES = (
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
    "vai",
    "wol",
    "xho",
    "yor",
    "zul",
)
AFRIMGSM_TASKS = tuple(f"afrimgsm_{language}" for language in AFRIMGSM_LANGUAGES)


def _direct_prompt(doc: dict[str, Any]) -> str:
    return f"Question: {str(doc['question']).strip()}\nAnswer:"


def _numeric_target(doc: dict[str, Any]) -> str:
    value = doc["answer_number"]
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


_DIRECT_VARIANTS = {
    "base": VariantSpec(
        task_name="afrimgsm",
        stop_strings=("Question:", "</s>", "<|im_end|>"),
        prompt_builder=_direct_prompt,
        target_builder=_numeric_target,
        num_fewshot=0,
        fewshots=(),
    )
}


@dataclass(slots=True)
class AfriMGSM(BaseGSM8KSuite):
    """AfriMGSM suite backed by a frozen language registry to keep imports offline-safe."""

    VARIANTS = _DIRECT_VARIANTS
    SCORING_MODE = "numeric_format_insensitive"
    dataset_path: str = "masakhane/afrimgsm"
    dataset_name: str | None = "eng"
    split: str = "test"
    language: str = "eng"
    variant: GSM8KVariant = "default"

    def __post_init__(self) -> None:
        if self.language not in AFRIMGSM_LANGUAGES:
            raise ValueError(f"unsupported afrimgsm language: {self.language!r}")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
        else:
            raise ValueError("afrimgsm dataset_name must match the configured language")
        if self.variant not in {"default", "base"}:
            raise ValueError("afrimgsm only supports the direct base variant")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"afrimgsm_{self.language}"

    def numeric_target_from_doc(self, doc: dict[str, Any]) -> str:
        return _numeric_target(doc)

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        metadata = super().result_metadata(generation_submission_mode=generation_submission_mode)
        metadata["language"] = self.language
        return metadata

    def _sample_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        return {
            "language": self.language,
            "question": str(doc["question"]).strip(),
            "answer_number": _numeric_target(doc),
            "equation_solution": None if doc["equation_solution"] is None else str(doc["equation_solution"]).strip(),
        }


def afrimgsm(*, language: str, **kwargs: Any) -> AfriMGSM:
    kwargs.setdefault("dataset_name", language)
    return AfriMGSM(language=language, **kwargs)


def _make_afrimgsm_factory(language: str) -> Any:
    def factory(**kwargs: Any) -> AfriMGSM:
        return afrimgsm(language=language, **kwargs)

    factory.__name__ = f"afrimgsm_{language}"
    return factory


for _language in AFRIMGSM_LANGUAGES:
    globals()[f"afrimgsm_{_language}"] = _make_afrimgsm_factory(_language)

del _language
