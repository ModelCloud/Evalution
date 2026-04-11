# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# HAE-RAE exposes five public multiple-choice evaluation subsets in lm-eval.
HAERAE_SUBSETS = {
    "general_knowledge": "general_knowledge",
    "history": "history",
    "loan_word": "loan_words",
    "rare_word": "rare_words",
    "standard_nomenclature": "standard_nomenclature",
}
HAERAE_TASKS = (
    "haerae",
    "haerae_general_knowledge",
    "haerae_history",
    "haerae_loan_word",
    "haerae_rare_word",
    "haerae_standard_nomenclature",
)
_HAERAE_CHOICE_LABELS = ("(A)", "(B)", "(C)", "(D)", "(E)")


def _haerae_task_name(subset: str) -> str:
    """Implement haerae task name for this module."""
    if subset == "haerae":
        return subset
    return f"haerae_{subset}"


def _haerae_raw_choices(doc: dict[str, Any]) -> list[str]:
    """Implement haerae raw choices for this module."""
    options = literal_eval(str(doc["options"]))
    return [str(option).strip() for option in options]


def _haerae_group_rows(*, dataset_path: str, split: str, cache_dir: str | None = None) -> Dataset:
    """Implement haerae group rows for this module. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    rows_by_subset: list[list[dict[str, Any]]] = []
    for subset, dataset_name in HAERAE_SUBSETS.items():
        dataset = load_dataset(dataset_path, dataset_name, split=split, cache_dir=cache_dir)
        rows = []
        for doc in dataset:
            row = dict(doc)
            row["haerae_subset"] = subset
            rows.append(row)
        rows_by_subset.append(rows)

    # Round-robin interleaving keeps the grouped benchmark representative instead of front-loading one config.
    combined_rows: list[dict[str, Any]] = []
    max_length = max(len(rows) for rows in rows_by_subset)
    for row_index in range(max_length):
        for rows in rows_by_subset:
            if row_index < len(rows):
                combined_rows.append(rows[row_index])
    return Dataset.from_list(combined_rows)


@dataclass(slots=True)
class Haerae(BaseMultipleChoiceSuite):
    # HAE-RAE measures Korean knowledge and vocabulary understanding with fixed five-label options.
    """Implement the haerae benchmark suite."""
    dataset_path: str = "HAERAE-HUB/HAE_RAE_BENCH"
    dataset_name: str | None = "general_knowledge"
    split: str = "test"
    stream: bool = (False)
    subset: str = "general_knowledge"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        if self.subset == "haerae":
            self.dataset_name = None
            return
        if self.subset not in HAERAE_SUBSETS:
            raise ValueError(f"unsupported haerae subset: {self.subset!r}")
        expected_dataset_name = HAERAE_SUBSETS[self.subset]
        if self.dataset_name in {None, expected_dataset_name}:
            self.dataset_name = expected_dataset_name
            return
        raise ValueError("haerae dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        if self.subset != "haerae":
            return load_dataset

        dataset_path = self.dataset_path
        split = self.split

        def loader(*_args: Any, cache_dir: str | None = None, **_kwargs: Any) -> Dataset:
            """Implement loader for haerae."""
            return _haerae_group_rows(dataset_path=dataset_path, split=split, cache_dir=cache_dir)

        return loader

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return _haerae_task_name(self.subset)

    def result_metadata(self) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        metadata = super().result_metadata()
        metadata["subset"] = self.subset
        return metadata

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        raw_choices = _haerae_raw_choices(doc)
        gold_label = str(doc["answer"]).strip()
        try:
            gold_index = _HAERAE_CHOICE_LABELS.index(gold_label)
        except ValueError as exc:
            raise ValueError(f"unsupported haerae answer label: {gold_label!r}") from exc
        return MultipleChoiceSample(
            index=index,
            prompt=str(doc["query"]).strip(),
            choices=list(_HAERAE_CHOICE_LABELS),
            gold_index=gold_index,
            metadata={
                "subset": self.subset,
                "dataset_name": str(doc.get("haerae_subset") or self.subset).strip(),
                "query": str(doc["query"]).strip(),
                "answer": gold_label,
                "raw_choices": raw_choices,
            },
        )


def haerae(**kwargs: Any) -> Haerae:
    """Implement haerae for this module."""
    return Haerae(subset="haerae", dataset_name=None, **kwargs)


def haerae_general_knowledge(**kwargs: Any) -> Haerae:
    """Implement haerae general knowledge for this module."""
    kwargs.setdefault("dataset_name", HAERAE_SUBSETS["general_knowledge"])
    return Haerae(subset="general_knowledge", **kwargs)


def haerae_history(**kwargs: Any) -> Haerae:
    """Implement haerae history for this module."""
    kwargs.setdefault("dataset_name", HAERAE_SUBSETS["history"])
    return Haerae(subset="history", **kwargs)


def haerae_loan_word(**kwargs: Any) -> Haerae:
    """Implement haerae loan word for this module."""
    kwargs.setdefault("dataset_name", HAERAE_SUBSETS["loan_word"])
    return Haerae(subset="loan_word", **kwargs)


def haerae_rare_word(**kwargs: Any) -> Haerae:
    """Implement haerae rare word for this module."""
    kwargs.setdefault("dataset_name", HAERAE_SUBSETS["rare_word"])
    return Haerae(subset="rare_word", **kwargs)


def haerae_standard_nomenclature(**kwargs: Any) -> Haerae:
    """Implement haerae standard nomenclature for this module."""
    kwargs.setdefault("dataset_name", HAERAE_SUBSETS["standard_nomenclature"])
    return Haerae(subset="standard_nomenclature", **kwargs)
