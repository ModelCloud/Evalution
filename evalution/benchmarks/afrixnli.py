# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

AFRIXNLI_LANGUAGES = (
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
AFRIXNLI_TASKS = tuple(f"afrixnli_{language}" for language in AFRIXNLI_LANGUAGES)
_AFRIXNLI_CHOICES = ["entailment", "neutral", "contradiction"]


def _afrixnli_prompt(premise: str, hypothesis: str) -> str:
    return (
        f"Premise: {premise.strip()}\n"
        f"Hypothesis: {hypothesis.strip()}\n"
        "Question: What is the relationship between the premise and hypothesis: entailment, neutral, or contradiction?\n"
        "Answer:"
    )


@dataclass(slots=True)
class AfriXNLI(BaseMultipleChoiceSuite):
    dataset_path: str = "masakhane/afrixnli"
    dataset_name: str | None = "eng"
    split: str = "test"
    language: str = "eng"

    def __post_init__(self) -> None:
        if self.language not in AFRIXNLI_LANGUAGES:
            raise ValueError(f"unsupported afrixnli language: {self.language!r}")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
            return
        raise ValueError("afrixnli dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"afrixnli_{self.language}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        return MultipleChoiceSample(
            index=index,
            prompt=_afrixnli_prompt(str(doc["premise"]), str(doc["hypothesis"])),
            choices=list(_AFRIXNLI_CHOICES),
            gold_index=int(doc["label"]),
            metadata={
                "language": self.language,
                "premise": str(doc["premise"]).strip(),
                "hypothesis": str(doc["hypothesis"]).strip(),
                "choice_labels": ["A", "B", "C"],
                "choice_texts": list(_AFRIXNLI_CHOICES),
            },
        )


def afrixnli(*, language: str, **kwargs: Any) -> AfriXNLI:
    kwargs.setdefault("dataset_name", language)
    return AfriXNLI(language=language, **kwargs)


def afrixnli_amh(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="amh", **kwargs)


def afrixnli_eng(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="eng", **kwargs)


def afrixnli_ewe(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="ewe", **kwargs)


def afrixnli_fra(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="fra", **kwargs)


def afrixnli_hau(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="hau", **kwargs)


def afrixnli_ibo(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="ibo", **kwargs)


def afrixnli_kin(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="kin", **kwargs)


def afrixnli_lin(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="lin", **kwargs)


def afrixnli_lug(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="lug", **kwargs)


def afrixnli_orm(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="orm", **kwargs)


def afrixnli_sna(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="sna", **kwargs)


def afrixnli_sot(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="sot", **kwargs)


def afrixnli_swa(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="swa", **kwargs)


def afrixnli_twi(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="twi", **kwargs)


def afrixnli_wol(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="wol", **kwargs)


def afrixnli_xho(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="xho", **kwargs)


def afrixnli_yor(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="yor", **kwargs)


def afrixnli_zul(**kwargs: Any) -> AfriXNLI:
    return afrixnli(language="zul", **kwargs)
