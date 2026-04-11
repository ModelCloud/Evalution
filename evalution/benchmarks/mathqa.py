# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve
from zipfile import ZipFile

from datasets import Dataset
import pcre

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Keep benchmark defaults and public task ids explicit at module scope.
_MATHQA_ARCHIVE_URL = "https://math-qa.github.io/math-QA/data/MathQA.zip"
_MATHQA_ARCHIVE_NAME = "MathQA.zip"
_MATHQA_SPLIT_FILES = {
    "train": "train.json",
    "validation": "dev.json",
    "test": "test.json",
}
_MATHQA_CHOICE_LABELS = ["A", "B", "C", "D", "E"]
_MATHQA_ANSWER_LABELS = ["a", "b", "c", "d", "e"]
_MATHQA_OPTION_RE = pcre.compile(r"[abcd] \) .*?, |e \) .*?$")


def _mathqa_prompt(problem: str) -> str:
    """Implement mathqa prompt for this module."""
    return f"Question: {problem.strip()}\nAnswer:"


def _mathqa_cache_path(cache_dir: str | None) -> Path:
    """Implement mathqa cache path for this module."""
    if cache_dir is not None:
        base_dir = Path(cache_dir)
    else:
        base_dir = Path.home() / ".cache" / "evalution" / "downloads"
    return base_dir / _MATHQA_ARCHIVE_NAME


def _ensure_mathqa_archive(*, cache_dir: str | None) -> Path:
    """Ensure mathqa archive."""
    archive_path = _mathqa_cache_path(cache_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        urlretrieve(_MATHQA_ARCHIVE_URL, archive_path)
    return archive_path


def _parse_mathqa_options(options_text: str) -> list[str]:
    """Parse mathqa options."""
    return [
        choice[4:].rstrip(" ,")
        for choice in _MATHQA_OPTION_RE.findall(options_text)
    ]


def _load_mathqa_dataset(
    dataset_path: str,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = (False),
) -> Dataset:
    """Load mathqa dataset."""
    del stream
    if dataset_path != "math_qa":
        raise ValueError(f"unsupported MathQA dataset path: {dataset_path!r}")

    split_member = _MATHQA_SPLIT_FILES.get(split)
    if split_member is None:
        raise ValueError(f"unsupported MathQA split: {split!r}")

    archive_path = _ensure_mathqa_archive(cache_dir=cache_dir)
    with ZipFile(archive_path) as archive:
        with archive.open(split_member) as split_file:
            rows = json.load(split_file)
    return Dataset.from_list(rows)


@dataclass(slots=True)
class MathQA(BaseMultipleChoiceSuite):
    """Implement the math QA benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "math_qa"
    # Align the default split with current benchmark-style harness usage.
    split: str = "test"
    stream: bool = (False)

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return _load_mathqa_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "mathqa"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        options_text = str(doc["options"])
        choices = _parse_mathqa_options(options_text)
        if len(choices) != 5:
            raise ValueError(f"expected 5 MathQA choices, found {len(choices)} in {options_text!r}")
        answer_key = str(doc["correct"]).strip().lower()
        return MultipleChoiceSample(
            index=index,
            prompt=_mathqa_prompt(str(doc["Problem"])),
            choices=choices,
            gold_index=_MATHQA_ANSWER_LABELS.index(answer_key),
            metadata={
                "problem": str(doc["Problem"]),
                "rationale": str(doc["Rationale"]),
                "annotated_formula": str(doc["annotated_formula"]),
                "linear_formula": str(doc["linear_formula"]),
                "category": str(doc["category"]),
                "answer_key": answer_key,
                "raw_options": options_text,
                "choice_labels": list(_MATHQA_CHOICE_LABELS),
                "choice_texts": choices,
            },
        )


def mathqa(**kwargs: Any) -> MathQA:
    """Implement mathqa for this module."""
    return MathQA(**kwargs)
