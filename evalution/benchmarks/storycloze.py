# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlretrieve
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Keep the public StoryCloze surface small and explicit because upstream only ships two yearly releases.
STORYCLOZE_YEARS = ("2016", "2018")
STORYCLOZE_TASKS = tuple(f"storycloze_{year}" for year in STORYCLOZE_YEARS)
# Reuse the original public CSV filenames so local manual copies still work without renaming.
_STORYCLOZE_CSV_FILENAMES = {
    ("2016", "validation"): "cloze_test_val__spring2016 - cloze_test_ALL_val.csv",
    ("2016", "test"): "cloze_test_test__spring2016 - cloze_test_ALL_test.csv",
    ("2018", "validation"): "cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv",
}
# Prefer direct CSV mirrors over deprecated dataset scripts when a public file is available.
_STORYCLOZE_CSV_URLS = {
    ("2016", "validation"): (
        "https://bdgit.educoder.net/pjwofns8z/tensorflow/raw/branch/master/datas/"
        "cloze_test_val__spring2016%20-%20cloze_test_ALL_val.csv"
    ),
    ("2016", "test"): (
        "https://bdgit.educoder.net/pjwofns8z/tensorflow/raw/branch/master/datas/"
        "cloze_test_test__spring2016%20-%20cloze_test_ALL_test.csv"
    ),
}
# Allow users to point Evalution at a manually downloaded StoryCloze directory when mirrors are unavailable.
_STORYCLOZE_DATA_DIR_ENV = "EVALUTION_STORYCLOZE_DATA_DIR"
# Normalize the original CSV headers to the canonical field names used throughout Evalution.
_STORYCLOZE_COLUMN_MAP = {
    "InputStoryid": "story_id",
    "InputSentence1": "input_sentence_1",
    "InputSentence2": "input_sentence_2",
    "InputSentence3": "input_sentence_3",
    "InputSentence4": "input_sentence_4",
    "RandomFifthSentenceQuiz1": "sentence_quiz1",
    "RandomFifthSentenceQuiz2": "sentence_quiz2",
    "AnswerRightEnding": "answer_right_ending",
}


def _storycloze_prompt(doc: dict[str, Any]) -> str:
    # Concatenate the observed story sentences once so each ending is scored against identical context.
    return " ".join(
        str(doc[field]).strip()
        for field in (
            "input_sentence_1",
            "input_sentence_2",
            "input_sentence_3",
            "input_sentence_4",
        )
    )


def _storycloze_csv_path(
    dataset_name: str,
    *,
    split: str,
    cache_dir: str | None = None,
) -> str:
    filename = _STORYCLOZE_CSV_FILENAMES.get((dataset_name, split))
    if filename is None:
        raise ValueError(f"unsupported storycloze split {split!r} for dataset {dataset_name!r}")

    local_data_dir = os.environ.get(_STORYCLOZE_DATA_DIR_ENV)
    if local_data_dir:
        local_path = Path(local_data_dir) / filename
        if local_path.exists():
            return str(local_path)

    url = _STORYCLOZE_CSV_URLS.get((dataset_name, split))
    if url is None:
        raise RuntimeError(
            "storycloze requires manually downloaded CSV files for this split; "
            f"set {_STORYCLOZE_DATA_DIR_ENV} to a directory containing {filename!r}"
        )

    target_dir = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "evalution" / "storycloze"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    if not target_path.exists():
        urlretrieve(url, target_path)
    return str(target_path)


def _load_storycloze_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> Any:
    if dataset_path != "LSDSem/story_cloze":
        raise ValueError(f"unsupported storycloze dataset_path: {dataset_path!r}")
    if dataset_name is None:
        raise ValueError("storycloze dataset_name cannot be None")
    csv_path = _storycloze_csv_path(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
    )
    dataset = load_dataset(
        "csv",
        data_files={split: csv_path},
        split=split,
        streaming=stream,
    )
    return dataset.rename_columns(_STORYCLOZE_COLUMN_MAP)


@dataclass(slots=True)
class StoryCloze(BaseMultipleChoiceSuite):
    # Mirror the public yearly StoryCloze releases while keeping dataset_name and year locked together.
    dataset_path: str = "LSDSem/story_cloze"
    dataset_name: str | None = "2016"
    split: str = "validation"
    year: str = "2016"

    def __post_init__(self) -> None:
        if self.year not in STORYCLOZE_YEARS:
            raise ValueError(f"unsupported storycloze year: {self.year!r}")
        if self.dataset_name in {None, self.year}:
            self.dataset_name = self.year
            return
        raise ValueError("storycloze dataset_name must match the configured year")

    def dataset_loader(self) -> Any:
        return _load_storycloze_dataset

    def task_name(self) -> str:
        return f"storycloze_{self.year}"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choices = [
            str(doc["sentence_quiz1"]).strip(),
            str(doc["sentence_quiz2"]).strip(),
        ]
        return MultipleChoiceSample(
            index=index,
            prompt=_storycloze_prompt(doc),
            choices=choices,
            gold_index=int(doc["answer_right_ending"]) - 1,
            metadata={
                "year": self.year,
                "story_id": str(doc.get("story_id", "")),
                "input_sentences": [
                    str(doc["input_sentence_1"]).strip(),
                    str(doc["input_sentence_2"]).strip(),
                    str(doc["input_sentence_3"]).strip(),
                    str(doc["input_sentence_4"]).strip(),
                ],
                "choice_texts": choices,
            },
        )


def storycloze(*, year: str = "2016", **kwargs: Any) -> StoryCloze:
    # Default the dataset config to the requested public release year.
    kwargs.setdefault("dataset_name", year)
    return StoryCloze(year=year, **kwargs)


def _make_storycloze_factory(year: str) -> Any:
    # Publish one import-stable zero-argument factory per yearly release.
    def factory(**kwargs: Any) -> StoryCloze:
        return storycloze(year=year, **kwargs)

    factory.__name__ = f"storycloze_{year}"
    return factory


for _year in STORYCLOZE_YEARS:
    globals()[f"storycloze_{_year}"] = _make_storycloze_factory(_year)

del _year
