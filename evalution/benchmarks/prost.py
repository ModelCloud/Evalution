# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_PROST_DATA_URL = "https://huggingface.co/datasets/corypaik/prost/resolve/main/data/default.jsonl"
_PROST_CHOICE_LABELS = ["A", "B", "C", "D"]


def _prost_prompt(context: str, ex_question: str) -> str:
    return f"{context.strip()}\nQuestion: {ex_question.strip()}\nAnswer:"


def _load_prost_dataset(
    dataset_path: str,
    *,
    split: str,
    cache_dir: str | None = None,
    streaming: bool = False,
) -> Any:
    if dataset_path != "corypaik/prost":
        raise ValueError(f"unsupported PROST dataset path: {dataset_path!r}")
    if split != "test":
        raise ValueError(f"unsupported PROST split: {split!r}")
    return load_dataset(
        "json",
        data_files={split: _PROST_DATA_URL},
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )


@dataclass(slots=True)
class Prost(BaseMultipleChoiceSuite):
    # Score PROST by ranking the answer text choices after the zero-shot prompt stem.
    dataset_path: str = "corypaik/prost"
    split: str = "test"
    streaming: bool = False

    def dataset_loader(self) -> Any:
        return _load_prost_dataset

    def task_name(self) -> str:
        return "prost"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [str(doc[label]).strip() for label in _PROST_CHOICE_LABELS]
        context = str(doc["context"]).strip()
        ex_question = str(doc["ex_question"]).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_prost_prompt(context, ex_question),
            choices=choices,
            gold_index=int(doc["label"]),
            metadata={
                "context": context,
                "question": str(doc["question"]).strip(),
                "ex_question": ex_question,
                "group": str(doc["group"]).strip(),
                "name": str(doc["name"]).strip(),
                "choice_labels": list(_PROST_CHOICE_LABELS),
                "choice_texts": choices,
            },
        )


def prost(**kwargs: Any) -> Prost:
    return Prost(**kwargs)
