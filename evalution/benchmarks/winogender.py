# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_SUPPORTED_VARIANTS = ("all", "gotcha")
_SUPPORTED_GENDERS = ("male", "female", "neutral")


def _winogender_prompt(sentence: str, pronoun: str) -> str:
    return f"{sentence.strip()} '{pronoun.strip().capitalize()}' refers to the"


def _load_winogender_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
    gender_filter: str | None = None,
) -> Any:
    if dataset_path != "oskarvanderwal/winogender":
        raise ValueError(f"unsupported WinoGender dataset path: {dataset_path!r}")
    if dataset_name not in _SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported WinoGender dataset variant: {dataset_name!r}")
    if split != "test":
        raise ValueError(f"unsupported WinoGender split: {split!r}")
    dataset = load_dataset(
        dataset_path,
        dataset_name,
        split=split,
        cache_dir=cache_dir,
        stream=stream,
    )
    if gender_filter is None:
        return dataset
    return dataset.filter(lambda row: str(row["gender"]) == gender_filter)


@dataclass(slots=True)
class WinoGender(BaseMultipleChoiceSuite):
    dataset_path: str = "oskarvanderwal/winogender"
    dataset_name: str | None = "all"
    split: str = "test"
    variant: str = "all"
    gender_filter: str | None = None

    def __post_init__(self) -> None:
        if self.variant not in _SUPPORTED_VARIANTS:
            raise ValueError(f"unsupported winogender variant: {self.variant!r}")
        if self.gender_filter is not None and self.gender_filter not in _SUPPORTED_GENDERS:
            raise ValueError(f"unsupported winogender gender filter: {self.gender_filter!r}")
        if self.dataset_name in {None, self.variant}:
            self.dataset_name = self.variant
            return
        raise ValueError("winogender dataset_name must match the configured variant")

    def dataset_loader(self) -> Any:
        return partial(_load_winogender_dataset, gender_filter=self.gender_filter)

    def task_name(self) -> str:
        if self.variant == "gotcha":
            if self.gender_filter is None:
                return "winogender_gotcha"
            return f"winogender_gotcha_{self.gender_filter}"
        if self.gender_filter is None:
            return "winogender_all"
        return f"winogender_{self.gender_filter}"

    def result_metadata(self) -> dict[str, Any]:
        metadata = super().result_metadata()
        metadata["prompt_variant"] = "pronoun_reference_prompt"
        if self.gender_filter is not None:
            metadata["gender_filter"] = self.gender_filter
        return metadata

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        occupation = str(doc["occupation"]).strip()
        participant = str(doc["participant"]).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_winogender_prompt(str(doc["sentence"]), str(doc["pronoun"])),
            choices=[occupation, participant],
            gold_index=int(doc["label"]),
            metadata={
                "sentid": str(doc["sentid"]),
                "sentence": str(doc["sentence"]).strip(),
                "pronoun": str(doc["pronoun"]).strip(),
                "gender": str(doc["gender"]).strip(),
                "target": str(doc["target"]).strip(),
                "variant": self.variant,
                "gotcha": bool(doc.get("gotcha", self.variant == "gotcha")),
                "choice_texts": [occupation, participant],
            },
        )


def winogender(*, variant: str = "all", gender_filter: str | None = None, **kwargs: Any) -> WinoGender:
    return WinoGender(variant=variant, dataset_name=variant, gender_filter=gender_filter, **kwargs)


def winogender_all(**kwargs: Any) -> WinoGender:
    return winogender(**kwargs)


def winogender_female(**kwargs: Any) -> WinoGender:
    return winogender(gender_filter="female", **kwargs)


def winogender_male(**kwargs: Any) -> WinoGender:
    return winogender(gender_filter="male", **kwargs)


def winogender_neutral(**kwargs: Any) -> WinoGender:
    return winogender(gender_filter="neutral", **kwargs)


def winogender_gotcha(**kwargs: Any) -> WinoGender:
    return winogender(variant="gotcha", **kwargs)


def winogender_gotcha_female(**kwargs: Any) -> WinoGender:
    return winogender(variant="gotcha", gender_filter="female", **kwargs)


def winogender_gotcha_male(**kwargs: Any) -> WinoGender:
    return winogender(variant="gotcha", gender_filter="male", **kwargs)
