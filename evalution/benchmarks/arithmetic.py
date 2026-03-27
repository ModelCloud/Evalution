# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from evalution.benchmarks.single_continuation import (
    BaseSingleContinuationSuite,
    SingleContinuationSample,
)

_ARITHMETIC_DATA_FILES = {
    "arithmetic_1dc": "data/single_digit_three_ops.jsonl",
    "arithmetic_2da": "data/two_digit_addition.jsonl",
    "arithmetic_2dm": "data/two_digit_multiplication.jsonl",
    "arithmetic_2ds": "data/two_digit_subtraction.jsonl",
    "arithmetic_3da": "data/three_digit_addition.jsonl",
    "arithmetic_3ds": "data/three_digit_subtraction.jsonl",
    "arithmetic_4da": "data/four_digit_addition.jsonl",
    "arithmetic_4ds": "data/four_digit_subtraction.jsonl",
    "arithmetic_5da": "data/five_digit_addition.jsonl",
    "arithmetic_5ds": "data/five_digit_subtraction.jsonl",
}


def _normalize_arithmetic_context(context: str) -> str:
    return context.strip().replace("\n\n", "\n").replace("Q:", "Question:").replace("A:", "Answer:")


def _load_arithmetic_dataset(
    dataset_path: str,
    dataset_name: str,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = True,
) -> Any:
    if dataset_path != "EleutherAI/arithmetic":
        raise ValueError(f"unsupported arithmetic dataset path: {dataset_path!r}")
    if split != "validation":
        raise ValueError(f"unsupported arithmetic split: {split!r}")

    relative_path = _ARITHMETIC_DATA_FILES.get(dataset_name)
    if relative_path is None:
        raise KeyError(f"unknown arithmetic subset: {dataset_name!r}")

    file_path = hf_hub_download(
        repo_id=dataset_path,
        filename=relative_path,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return load_dataset(
        "json",
        data_files={split: file_path},
        split=split,
        cache_dir=cache_dir,
        streaming=stream,
    )


@dataclass(slots=True)
class Arithmetic(BaseSingleContinuationSuite):
    dataset_path: str = "EleutherAI/arithmetic"
    dataset_name: str | None = "arithmetic_1dc"
    split: str = "validation"

    def dataset_loader(self) -> Any:
        return _load_arithmetic_dataset

    def task_name(self) -> str:
        if self.dataset_name is None:
            raise ValueError("Arithmetic requires a dataset_name variant")
        return self.dataset_name

    def include_perplexity(self) -> bool:
        return False

    def build_sample(self, doc: dict[str, Any], *, index: int) -> SingleContinuationSample:
        if self.dataset_name is None:
            raise ValueError("Arithmetic requires a dataset_name variant")
        raw_context = str(doc["context"])
        raw_completion = str(doc["completion"])
        return SingleContinuationSample(
            index=index,
            prompt=_normalize_arithmetic_context(raw_context),
            target=raw_completion,
            metadata={
                "variant": self.dataset_name,
                "source_file": _ARITHMETIC_DATA_FILES[self.dataset_name],
                "raw_context": raw_context,
                "raw_completion": raw_completion,
            },
        )


def _arithmetic_variant(dataset_name: str, **kwargs: Any) -> Arithmetic:
    return Arithmetic(dataset_name=dataset_name, **kwargs)


def arithmetic_1dc(**kwargs: Any) -> Arithmetic:
    return _arithmetic_variant("arithmetic_1dc", **kwargs)


def arithmetic_2da(**kwargs: Any) -> Arithmetic:
    return _arithmetic_variant("arithmetic_2da", **kwargs)


def arithmetic_2dm(**kwargs: Any) -> Arithmetic:
    return _arithmetic_variant("arithmetic_2dm", **kwargs)


def arithmetic_2ds(**kwargs: Any) -> Arithmetic:
    return _arithmetic_variant("arithmetic_2ds", **kwargs)


def arithmetic_3da(**kwargs: Any) -> Arithmetic:
    return _arithmetic_variant("arithmetic_3da", **kwargs)


def arithmetic_3ds(**kwargs: Any) -> Arithmetic:
    return _arithmetic_variant("arithmetic_3ds", **kwargs)


def arithmetic_4da(**kwargs: Any) -> Arithmetic:
    return _arithmetic_variant("arithmetic_4da", **kwargs)


def arithmetic_4ds(**kwargs: Any) -> Arithmetic:
    return _arithmetic_variant("arithmetic_4ds", **kwargs)


def arithmetic_5da(**kwargs: Any) -> Arithmetic:
    return _arithmetic_variant("arithmetic_5da", **kwargs)


def arithmetic_5ds(**kwargs: Any) -> Arithmetic:
    return _arithmetic_variant("arithmetic_5ds", **kwargs)
