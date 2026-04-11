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
    """Implement afrixnli prompt for this module."""
    return (
        f"Premise: {premise.strip()}\n"
        f"Hypothesis: {hypothesis.strip()}\n"
        "Question: What is the relationship between the premise and hypothesis: entailment, neutral, or contradiction?\n"
        "Answer:"
    )


@dataclass(slots=True)
class AfriXNLI(BaseMultipleChoiceSuite):
    """Implement the afri XNLI benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "masakhane/afrixnli"
    dataset_name: str | None = "eng"
    split: str = "test"
    language: str = "eng"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.language not in AFRIXNLI_LANGUAGES:
            raise ValueError(f"unsupported afrixnli language: {self.language!r}")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
            return
        raise ValueError("afrixnli dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"afrixnli_{self.language}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
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
    """Implement afrixnli for this module."""
    kwargs.setdefault("dataset_name", language)
    return AfriXNLI(language=language, **kwargs)


def afrixnli_amh(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli amh for this module."""
    return afrixnli(language="amh", **kwargs)


def afrixnli_eng(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli eng for this module."""
    return afrixnli(language="eng", **kwargs)


def afrixnli_ewe(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli ewe for this module."""
    return afrixnli(language="ewe", **kwargs)


def afrixnli_fra(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli fra for this module."""
    return afrixnli(language="fra", **kwargs)


def afrixnli_hau(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli hau for this module."""
    return afrixnli(language="hau", **kwargs)


def afrixnli_ibo(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli ibo for this module."""
    return afrixnli(language="ibo", **kwargs)


def afrixnli_kin(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli kin for this module."""
    return afrixnli(language="kin", **kwargs)


def afrixnli_lin(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli lin for this module."""
    return afrixnli(language="lin", **kwargs)


def afrixnli_lug(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli lug for this module."""
    return afrixnli(language="lug", **kwargs)


def afrixnli_orm(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli orm for this module."""
    return afrixnli(language="orm", **kwargs)


def afrixnli_sna(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli sna for this module."""
    return afrixnli(language="sna", **kwargs)


def afrixnli_sot(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli sot for this module."""
    return afrixnli(language="sot", **kwargs)


def afrixnli_swa(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli swa for this module."""
    return afrixnli(language="swa", **kwargs)


def afrixnli_twi(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli twi for this module."""
    return afrixnli(language="twi", **kwargs)


def afrixnli_wol(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli wol for this module."""
    return afrixnli(language="wol", **kwargs)


def afrixnli_xho(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli xho for this module."""
    return afrixnli(language="xho", **kwargs)


def afrixnli_yor(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli yor for this module."""
    return afrixnli(language="yor", **kwargs)


def afrixnli_zul(**kwargs: Any) -> AfriXNLI:
    """Implement afrixnli zul for this module."""
    return afrixnli(language="zul", **kwargs)
