# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from datasets import Dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Keep benchmark defaults and public task ids explicit at module scope.
_LOGIQA_SOURCE_URLS = {
    "train": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Train.txt",
    "validation": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt",
    "test": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt",
}
_LOGIQA_CHOICE_LABELS = ["A", "B", "C", "D"]
_LOGIQA_ANSWER_LABELS = ["a", "b", "c", "d"]


def _normalize_logiqa_text(text: str) -> str:
    """Normalize logiqa text. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return text.replace(".", ". ").strip()


def _logiqa_prompt(context: str, question: str, options: list[str]) -> str:
    """Implement logiqa prompt for this module."""
    lines = [
        f"Passage: {context}",
        f"Question: {question}",
        "Choices:",
    ]
    for label, option in zip(_LOGIQA_CHOICE_LABELS, options, strict=True):
        lines.append(f"{label}. {option}")
    lines.append("Answer:")
    return "\n".join(lines)


def _logiqa_cache_dir(cache_dir: str | None) -> Path:
    """Implement logiqa cache dir for this module."""
    if cache_dir is not None:
        return Path(cache_dir)
    return Path.home() / ".cache" / "evalution" / "downloads" / "logiqa"


def _ensure_logiqa_split_file(split: str, *, cache_dir: str | None) -> Path:
    """Ensure logiqa split file."""
    source_url = _LOGIQA_SOURCE_URLS.get(split)
    if source_url is None:
        raise ValueError(f"unsupported LogiQA split: {split!r}")

    cache_path = _logiqa_cache_dir(cache_dir) / Path(source_url).name
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        urlretrieve(source_url, cache_path)
    return cache_path


def _load_logiqa_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = (False),
) -> Dataset:
    """Load logiqa dataset."""
    del stream
    if dataset_path != "EleutherAI/logiqa":
        raise ValueError(f"unsupported LogiQA dataset path: {dataset_path!r}")
    if dataset_name != "logiqa":
        raise ValueError(f"unsupported LogiQA dataset name: {dataset_name!r}")

    split_path = _ensure_logiqa_split_file(split, cache_dir=cache_dir)
    blocks = split_path.read_text(encoding="utf-8").strip().split("\n\n")
    rows: list[dict[str, Any]] = []
    for block in blocks:
        lines = block.splitlines()
        rows.append(
            {
                "label": lines[0].strip(),
                "context": _normalize_logiqa_text(lines[1]),
                "question": _normalize_logiqa_text(lines[2]),
                "options": [_normalize_logiqa_text(option[2:]) for option in lines[3:]],
            }
        )
    return Dataset.from_list(rows)


@dataclass(slots=True)
class LogiQA(BaseMultipleChoiceSuite):
    """Implement the logi QA benchmark suite."""
    # Keep the suite defaults explicit on the class body so CLI, YAML, and Python stay aligned.
    dataset_path: str = "EleutherAI/logiqa"
    dataset_name: str | None = "logiqa"
    split: str = "validation"
    stream: bool = (False)

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return _load_logiqa_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "logiqa"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        """Build one benchmark sample from a dataset row."""
        options = [str(option) for option in doc["options"]]
        answer_key = str(doc["label"]).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_logiqa_prompt(str(doc["context"]), str(doc["question"]), options),
            choices=options,
            gold_index=_LOGIQA_ANSWER_LABELS.index(answer_key),
            metadata={
                "context": str(doc["context"]),
                "question": str(doc["question"]),
                "answer_key": answer_key,
                "choice_labels": list(_LOGIQA_CHOICE_LABELS),
                "choice_texts": options,
            },
        )


def logiqa(**kwargs: Any) -> LogiQA:
    """Implement logiqa for this module."""
    return LogiQA(**kwargs)
