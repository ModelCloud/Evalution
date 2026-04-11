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
XNLI_LANGUAGES = (
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
)
XNLI_TASKS = tuple(f"xnli_{language}" for language in XNLI_LANGUAGES)
_XNLI_CHOICES = ["entailment", "neutral", "contradiction"]


def _xnli_prompt(premise: str, hypothesis: str) -> str:
    """Implement XNLI prompt for this module."""
    return (
        f"Premise: {premise.strip()}\n"
        f"Hypothesis: {hypothesis.strip()}\n"
        "Question: What is the relationship between the premise and hypothesis: entailment, neutral, or contradiction?\n"
        "Answer:"
    )


@dataclass(slots=True)
class XNLI(BaseMultipleChoiceSuite):
    """Implement the XNLI benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "facebook/xnli"
    dataset_name: str | None = "en"
    split: str = "validation"
    language: str = "en"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.language not in XNLI_LANGUAGES:
            raise ValueError(f"unsupported xnli language: {self.language!r}")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
            return
        raise ValueError("xnli dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"xnli_{self.language}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        return MultipleChoiceSample(
            index=index,
            prompt=_xnli_prompt(str(doc["premise"]), str(doc["hypothesis"])),
            choices=list(_XNLI_CHOICES),
            gold_index=int(doc["label"]),
            metadata={
                "language": self.language,
                "premise": str(doc["premise"]).strip(),
                "hypothesis": str(doc["hypothesis"]).strip(),
                "choice_labels": ["A", "B", "C"],
                "choice_texts": list(_XNLI_CHOICES),
            },
        )


def xnli(*, language: str, **kwargs: Any) -> XNLI:
    """Implement XNLI for this module."""
    kwargs.setdefault("dataset_name", language)
    return XNLI(language=language, **kwargs)


def _make_xnli_factory(language: str) -> Any:
    """Make XNLI factory."""
    def factory(**kwargs: Any) -> XNLI:
        """Implement factory for this module."""
        return xnli(language=language, **kwargs)

    factory.__name__ = f"xnli_{language}"
    return factory


for _language in XNLI_LANGUAGES:
    globals()[f"xnli_{_language}"] = _make_xnli_factory(_language)

del _language
