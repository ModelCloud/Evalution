# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.scorers.classification import f1_for_label

# Keep benchmark defaults and public task ids explicit at module scope.
_SUPPORTED_LANGUAGES = ("de", "en", "es", "fr", "ja", "ko", "zh")


def _paws_x_prompt(sentence1: str, sentence2: str) -> str:
    """Implement paws x prompt for this module."""
    return (
        f"Sentence 1: {sentence1.strip()}\n"
        f"Sentence 2: {sentence2.strip()}\n"
        "Question: Do both sentences mean the same thing?\n"
        "Answer:"
    )


@dataclass(slots=True)
class PAWSX(BaseMultipleChoiceSuite):
    """Implement the pawsx benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "paws-x"
    dataset_name: str | None = "en"
    # Align the default split with current benchmark-style harness usage.
    split: str = "test"
    language: str = "en"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.language not in _SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported paws-x language: {self.language!r}")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
            return
        raise ValueError("paws-x dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"paws_x_{self.language}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        return MultipleChoiceSample(
            index=index,
            prompt=_paws_x_prompt(str(doc["sentence1"]), str(doc["sentence2"])),
            choices=["no", "yes"],
            gold_index=int(doc["label"]),
            metadata={"id": int(doc["id"]), "language": self.language},
        )

    def extra_metrics(
        self,
        *,
        samples: list[MultipleChoiceSample],
        raw_predictions: list[int],
        normalized_predictions: list[int],
    ) -> dict[str, float]:
        """Compute extra metrics from the collected predictions. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        gold_labels = [sample.gold_index for sample in samples]
        return {
            "f1,ll_boolean": f1_for_label(gold_labels, raw_predictions, label=1),
            "f1,ll_avg_boolean": f1_for_label(
                gold_labels,
                normalized_predictions,
                label=1,
            ),
        }


def paws_x(*, language: str, **kwargs: Any) -> PAWSX:
    """Implement paws x for this module."""
    return PAWSX(language=language, dataset_name=language, **kwargs)


def paws_x_de(**kwargs: Any) -> PAWSX:
    """Implement paws x de for this module."""
    return paws_x(language="de", **kwargs)


def paws_x_en(**kwargs: Any) -> PAWSX:
    """Implement paws x en for this module."""
    return paws_x(language="en", **kwargs)


def paws_x_es(**kwargs: Any) -> PAWSX:
    """Implement paws x es for this module."""
    return paws_x(language="es", **kwargs)


def paws_x_fr(**kwargs: Any) -> PAWSX:
    """Implement paws x fr for this module."""
    return paws_x(language="fr", **kwargs)


def paws_x_ja(**kwargs: Any) -> PAWSX:
    """Implement paws x ja for this module."""
    return paws_x(language="ja", **kwargs)


def paws_x_ko(**kwargs: Any) -> PAWSX:
    """Implement paws x ko for this module."""
    return paws_x(language="ko", **kwargs)


def paws_x_zh(**kwargs: Any) -> PAWSX:
    """Implement paws x zh for this module."""
    return paws_x(language="zh", **kwargs)
