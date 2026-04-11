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
BANGLA_SUBSETS = (
    "boolqa",
    "commonsenseqa",
    "mmlu",
    "openbookqa",
    "piqa",
)
BANGLA_TASKS = tuple(f"bangla_{subset}" for subset in BANGLA_SUBSETS)
_BOOLQA_CHOICES = ["yes", "no"]
_BINARY_LABELS = ["A", "B"]
_FOUR_WAY_LABELS = ["A", "B", "C", "D"]
_FIVE_WAY_LABELS = ["A", "B", "C", "D", "E"]
_BANGLA_MMLU_DESCRIPTION = (
    "The following are multiple choice questions (with answers) about range of topics in Bangla"
)
_BOOLQA_ANSWER_MAP = {
    "0": 0,
    "1": 1,
    "false": 1,
    "na": 1,
    "no": 1,
    "true": 0,
    "yes": 0,
    "না": 1,
    "হ্যাঁ": 0,
}


@dataclass(frozen=True, slots=True)
class _BanglaSubsetConfig:
    """Define the bangla subset config helper class."""
    # Keep the class-level state explicit for this helper.
    dataset_path: str
    dataset_name: str | None
    split: str


# Keep benchmark defaults and public task ids explicit at module scope.
_BANGLA_SUBSET_CONFIGS = {
    "boolqa": _BanglaSubsetConfig(
        dataset_path="hishab/boolq_bn",
        dataset_name=None,
        split="validation",
    ),
    "commonsenseqa": _BanglaSubsetConfig(
        dataset_path="hishab/commonsenseqa-bn",
        dataset_name=None,
        split="validation",
    ),
    "mmlu": _BanglaSubsetConfig(
        dataset_path="hishab/titulm-bangla-mmlu",
        dataset_name="all",
        split="test",
    ),
    "openbookqa": _BanglaSubsetConfig(
        dataset_path="hishab/openbookqa-bn",
        dataset_name=None,
        split="test",
    ),
    "piqa": _BanglaSubsetConfig(
        dataset_path="hishab/piqa-bn",
        dataset_name=None,
        split="validation",
    ),
}


def _resolve_boolqa_gold_index(doc: dict[str, Any]) -> int:
    """Resolve boolqa gold index. Preserve the fallback order expected by the surrounding caller."""
    for key in ("answer", "answer_bn"):
        if key not in doc:
            continue
        value = doc[key]
        if isinstance(value, bool):
            return 0 if value else 1
        if isinstance(value, int):
            if value in {0, 1}:
                return value
        normalized = str(value).strip().lower()
        if normalized in _BOOLQA_ANSWER_MAP:
            return _BOOLQA_ANSWER_MAP[normalized]
    raise ValueError("bangla boolqa answer must resolve to yes/no")


def _resolve_label_index(value: Any, labels: list[str]) -> int:
    """Resolve label index. Preserve the fallback order expected by the surrounding caller."""
    if isinstance(value, int):
        if 0 <= value < len(labels):
            return value
        if 1 <= value <= len(labels):
            return value - 1

    normalized = str(value).strip().upper()
    if normalized in labels:
        return labels.index(normalized)
    if normalized.isdigit():
        numeric = int(normalized)
        if 0 <= numeric < len(labels):
            return numeric
        if 1 <= numeric <= len(labels):
            return numeric - 1
    raise ValueError(f"unsupported label value: {value!r}")


def _choice_texts(doc: dict[str, Any]) -> list[str]:
    """Implement choice texts for this module."""
    return [str(text).strip() for text in doc["choices"]["text"]]


def _choice_labels(doc: dict[str, Any]) -> list[str]:
    """Implement choice labels for this module."""
    return [str(label).strip() for label in doc["choices"]["label"]]


def _labeled_prompt(stem: str, labels: list[str], choices: list[str]) -> str:
    """Implement labeled prompt for this module."""
    lines = [stem.strip()]
    lines.extend(f"{label}. {choice}" for label, choice in zip(labels, choices, strict=True))
    lines.append("Answer:")
    return "\n".join(lines)


@dataclass(slots=True)
class Bangla(BaseMultipleChoiceSuite):
    """Implement the bangla benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = ""
    dataset_name: str | None = None
    split: str = ""
    subset: str = "boolqa"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization. Preserve the fallback order expected by the surrounding caller."""
        if self.subset not in BANGLA_SUBSETS:
            raise ValueError(f"unsupported bangla subset: {self.subset!r}")
        config = _BANGLA_SUBSET_CONFIGS[self.subset]
        if self.dataset_path in {"", config.dataset_path}:
            self.dataset_path = config.dataset_path
        else:
            raise ValueError("bangla dataset_path must match the configured subset")
        if self.dataset_name in {None, config.dataset_name}:
            self.dataset_name = config.dataset_name
        else:
            raise ValueError("bangla dataset_name must match the configured subset")
        if self.split in {"", config.split}:
            self.split = config.split
        else:
            raise ValueError("bangla split must match the configured subset")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"bangla_{self.subset}"

    def result_metadata(self) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        metadata = super().result_metadata()
        metadata["subset"] = self.subset
        return metadata

    def label_prompt(
        self,
        sample: MultipleChoiceSample,
        *,
        choice_order: tuple[int, ...],
        labels: tuple[str, ...],
    ) -> str:
        """Implement label prompt for bangla."""
        if self.subset == "boolqa":
            return super().label_prompt(sample, choice_order=choice_order, labels=labels)
        raw_choices = sample.metadata["raw_choices"]
        question = sample.metadata["question"]
        if self.subset in {"commonsenseqa", "openbookqa", "piqa"}:
            lines = [question]
            lines.extend(
                f"{label}. {raw_choices[choice_index]}"
                for label, choice_index in zip(labels, choice_order, strict=True)
            )
            lines.append("Answer:")
            return "\n".join(lines)
        if self.subset == "mmlu":
            parts = [question]
            parts.extend(
                f"{label}. {raw_choices[choice_index]}"
                for label, choice_index in zip(labels, choice_order, strict=True)
            )
            parts.append("Answer:")
            return " ".join(parts)
        raise AssertionError(f"unsupported bangla subset branch: {self.subset!r}")

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row. Preserve the fallback order expected by the surrounding caller."""
        if self.subset == "boolqa":
            return MultipleChoiceSample(
                index=index,
                prompt=(
                    f"Passage:\n{str(doc['passage']).strip()}\n\n"
                    f"Question:\n{str(doc['question']).strip()}\n\n"
                    "Answer:"
                ),
                choices=list(_BOOLQA_CHOICES),
                gold_index=_resolve_boolqa_gold_index(doc),
                metadata={
                    "subset": self.subset,
                    "passage": str(doc["passage"]).strip(),
                    "question": str(doc["question"]).strip(),
                },
            )

        if self.subset == "commonsenseqa":
            labels = _choice_labels(doc)
            raw_choices = _choice_texts(doc)
            return MultipleChoiceSample(
                index=index,
                prompt=_labeled_prompt(str(doc["question_stem"]), labels, raw_choices),
                choices=labels,
                gold_index=_resolve_label_index(doc["answerKey"], labels),
                metadata={
                    "subset": self.subset,
                    "question": str(doc["question_stem"]).strip(),
                    "choice_labels": labels,
                    "raw_choices": raw_choices,
                },
            )

        if self.subset == "openbookqa":
            labels = _choice_labels(doc)
            raw_choices = _choice_texts(doc)
            return MultipleChoiceSample(
                index=index,
                prompt=_labeled_prompt(str(doc["question_stem"]), labels, raw_choices),
                choices=labels,
                gold_index=_resolve_label_index(doc["answerKey"], labels),
                metadata={
                    "subset": self.subset,
                    "question": str(doc["question_stem"]).strip(),
                    "choice_labels": labels,
                    "raw_choices": raw_choices,
                },
            )

        if self.subset == "piqa":
            raw_choices = [str(doc["sol1"]).strip(), str(doc["sol2"]).strip()]
            return MultipleChoiceSample(
                index=index,
                prompt=_labeled_prompt(str(doc["goal"]), list(_BINARY_LABELS), raw_choices),
                choices=list(_BINARY_LABELS),
                gold_index=_resolve_label_index(doc["label"], list(_BINARY_LABELS)),
                metadata={
                    "subset": self.subset,
                    "question": str(doc["goal"]).strip(),
                    "choice_labels": list(_BINARY_LABELS),
                    "raw_choices": raw_choices,
                },
            )

        if self.subset == "mmlu":
            raw_choices = [str(choice).strip() for choice in doc["options"]]
            if len(raw_choices) != 4:
                raise ValueError("bangla mmlu expects exactly four answer options")
            question = str(doc["question"]).strip()
            prompt = " ".join(
                [
                    f"{_BANGLA_MMLU_DESCRIPTION}{question}",
                    f"A. {raw_choices[0]}",
                    f"B. {raw_choices[1]}",
                    f"C. {raw_choices[2]}",
                    f"D. {raw_choices[3]}",
                    "Answer:",
                ]
            )
            return MultipleChoiceSample(
                index=index,
                prompt=prompt,
                choices=list(_FOUR_WAY_LABELS),
                gold_index=_resolve_label_index(doc["answer"], list(_FOUR_WAY_LABELS)),
                metadata={
                    "subset": self.subset,
                    "question": question,
                    "choice_labels": list(_FOUR_WAY_LABELS),
                    "raw_choices": raw_choices,
                    "subject": str(doc.get("subject", "")).strip(),
                },
            )

        raise AssertionError(f"unsupported bangla subset branch: {self.subset!r}")


def bangla(*, subset: str, **kwargs: Any) -> Bangla:
    """Implement bangla for this module."""
    return Bangla(subset=subset, **kwargs)


def bangla_boolqa(**kwargs: Any) -> Bangla:
    """Implement bangla boolqa for this module."""
    return bangla(subset="boolqa", **kwargs)


def bangla_commonsenseqa(**kwargs: Any) -> Bangla:
    """Implement bangla commonsenseqa for this module."""
    return bangla(subset="commonsenseqa", **kwargs)


def bangla_mmlu(**kwargs: Any) -> Bangla:
    """Implement bangla MMLU for this module."""
    return bangla(subset="mmlu", **kwargs)


def bangla_openbookqa(**kwargs: Any) -> Bangla:
    """Implement bangla openbookqa for this module."""
    return bangla(subset="openbookqa", **kwargs)


def bangla_piqa(**kwargs: Any) -> Bangla:
    """Implement bangla PIQA for this module."""
    return bangla(subset="piqa", **kwargs)
