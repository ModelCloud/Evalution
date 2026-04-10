# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

from evalution.benchmarks.mmlu import _MMLU_SUBSETS
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.benchmarks.subsets import ResolvedSubsets

_CHOICE_LABELS = ("A", "B", "C", "D")
# Freeze the full subject list at import so the "all" loader path does not depend on hidden tree helpers.
_MMLU_REDUX_ALL_SUBJECTS = _MMLU_SUBSETS.resolve("all").leaf_values


def _mmlu_redux_prompt(*, question: str, choices: list[str]) -> str:
    # Match the benchmark's letter-only answer format while keeping the choice texts in the prompt.
    lines = [question.strip()]
    for label, choice in zip(_CHOICE_LABELS, choices, strict=True):
        lines.append(f"{label}. {choice}")
    lines.append(
        "Please respond with the correct letter (A, B, C or D) without any additional comments, "
        "only the correct letter:"
    )
    return "\n".join(lines)


def _mmlu_redux_gold_index(answer: Any) -> int:
    if isinstance(answer, str):
        stripped = answer.strip().upper()
        if stripped in _CHOICE_LABELS:
            return _CHOICE_LABELS.index(stripped)
        return int(stripped)
    return int(answer)


def _load_mmlu_redux_subject_dataset(
    dataset_path: str,
    subject: str,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> Any:
    return load_dataset(
        dataset_path,
        subject,
        split=split,
        cache_dir=cache_dir,
        streaming=stream,
        trust_remote_code=True,
    )


@dataclass(slots=True)
class MMLURedux(BaseMultipleChoiceSuite):
    # Evaluate the MMLU-Redux subject shards through letter-only answer scoring while preserving subset metadata.
    dataset_path: str = "fxmarty/mmlu-redux-2.0-ok"
    dataset_name: str | None = None
    split: str = "test"
    stream: bool = False
    subsets: str | list[str] = "all"

    def dataset_loader(self) -> Any:
        selected_subjects = self._selected_subjects()
        dataset_path = self.dataset_path
        split = self.split

        def loader(*_args: Any, cache_dir: str | None = None, stream: bool = False, **_kwargs: Any) -> Any:
            if stream and (selected_subjects is None or len(selected_subjects) != 1):
                raise ValueError("mmlu_redux only supports stream=True for a single selected subject")
            if selected_subjects is None:
                rows: list[dict[str, Any]] = []
                for subject in _MMLU_REDUX_ALL_SUBJECTS:
                    dataset = _load_mmlu_redux_subject_dataset(
                        dataset_path,
                        subject,
                        split=split,
                        cache_dir=cache_dir,
                        stream=False,
                    )
                    for doc in dataset:
                        row = dict(doc)
                        row.setdefault("subject", subject)
                        rows.append(row)
                return Dataset.from_list(rows)

            if len(selected_subjects) == 1:
                dataset = _load_mmlu_redux_subject_dataset(
                    dataset_path,
                    selected_subjects[0],
                    split=split,
                    cache_dir=cache_dir,
                    stream=stream,
                )
                if stream:
                    return dataset
                rows = []
                for doc in dataset:
                    row = dict(doc)
                    row.setdefault("subject", selected_subjects[0])
                    rows.append(row)
                return Dataset.from_list(rows)

            rows = []
            for subject in selected_subjects:
                dataset = _load_mmlu_redux_subject_dataset(
                    dataset_path,
                    subject,
                    split=split,
                    cache_dir=cache_dir,
                    stream=False,
                )
                for doc in dataset:
                    row = dict(doc)
                    row.setdefault("subject", subject)
                    rows.append(row)
            return Dataset.from_list(rows)

        return loader

    def task_name(self) -> str:
        resolved_subsets = self._resolved_subsets()
        if resolved_subsets.selection_mode == "single" and resolved_subsets.kinds[0] == "all":
            return "mmlu_redux"
        suffix = "__".join(canonical.replace(".", "_") for canonical in resolved_subsets.canonicals)
        return f"mmlu_redux_{suffix}"

    def result_metadata(self) -> dict[str, Any]:
        resolved_subsets = self._resolved_subsets()
        metadata = super().result_metadata()
        metadata.update(
            {
                "dataset_name": self.dataset_name,
                "subsets": list(resolved_subsets.canonicals),
                "subset_paths": [list(path) for path in resolved_subsets.paths],
                "subset_kinds": list(resolved_subsets.kinds),
                "selection_mode": resolved_subsets.selection_mode,
            }
        )
        return metadata

    def continuation_for_choice(self, choice: str) -> str:
        return f" {choice}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [str(choice).strip() for choice in doc["choices"]]
        if len(choices) != len(_CHOICE_LABELS):
            raise ValueError(f"mmlu_redux expects four answer choices, got {len(choices)}")
        subject = str(doc.get("subject", "")).strip()
        leaf_subset = _MMLU_SUBSETS.leaf_subset(subject) if subject else None
        return MultipleChoiceSample(
            index=index,
            prompt=_mmlu_redux_prompt(
                question=str(doc["question"]),
                choices=choices,
            ),
            choices=list(_CHOICE_LABELS),
            gold_index=_mmlu_redux_gold_index(doc["answer"]),
            metadata={
                "question": str(doc["question"]).strip(),
                "subject": subject,
                "subset": leaf_subset,
                "subset_path": leaf_subset.split(".") if leaf_subset else [],
                "subset_kind": "leaf" if leaf_subset else "",
                "choice_texts": choices,
            },
        )

    def _resolved_subsets(self) -> ResolvedSubsets:
        return _MMLU_SUBSETS.resolve_many(self.subsets)

    def _selected_subjects(self) -> list[str] | None:
        resolved_subsets = self._resolved_subsets()
        if resolved_subsets.selection_mode == "single" and resolved_subsets.kinds[0] == "all":
            return None
        return list(resolved_subsets.leaf_values)


def mmlu_redux(**kwargs: Any) -> MMLURedux:
    # Publish one generic constructor because subset selection is already encoded in the suite options.
    return MMLURedux(**kwargs)
