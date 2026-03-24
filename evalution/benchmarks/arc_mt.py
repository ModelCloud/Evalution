# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.arc_exam import BaseARCExamSuite

ARC_MT_LANGUAGES = (
    "da",
    "de",
    "el",
    "es",
    "fi",
    "hu",
    "is",
    "it",
    "nb",
    "pl",
    "pt",
    "sv",
)
ARC_MT_TASKS = tuple(f"arc_mt_{language}" for language in ARC_MT_LANGUAGES)


@dataclass(frozen=True, slots=True)
class _ARCMTConfig:
    dataset_path: str
    dataset_name: str | None


_ARC_MT_CONFIGS = {
    "da": _ARCMTConfig("LumiOpen/arc_challenge_mt", "da"),
    "de": _ARCMTConfig("LumiOpen/arc_challenge_mt", "de"),
    "el": _ARCMTConfig("LumiOpen/arc_challenge_mt", "el"),
    "es": _ARCMTConfig("LumiOpen/arc_challenge_mt", "es"),
    "fi": _ARCMTConfig("LumiOpen/arc_challenge_mt", "fi"),
    "hu": _ARCMTConfig("LumiOpen/arc_challenge_mt", "hu"),
    "is": _ARCMTConfig("mideind/icelandic-arc-challenge", None),
    "it": _ARCMTConfig("LumiOpen/arc_challenge_mt", "it"),
    "nb": _ARCMTConfig("LumiOpen/arc_challenge_mt", "nb"),
    "pl": _ARCMTConfig("LumiOpen/arc_challenge_mt", "pl"),
    "pt": _ARCMTConfig("LumiOpen/arc_challenge_mt", "pt"),
    "sv": _ARCMTConfig("LumiOpen/arc_challenge_mt", "sv"),
}


@dataclass(slots=True)
class ARCMT(BaseARCExamSuite):
    dataset_path: str = ""
    dataset_name: str | None = None
    split: str = "test"
    language: str = "fi"

    def __post_init__(self) -> None:
        if self.language not in ARC_MT_LANGUAGES:
            raise ValueError(f"unsupported arc_mt language: {self.language!r}")
        config = _ARC_MT_CONFIGS[self.language]
        if self.dataset_path in {"", config.dataset_path}:
            self.dataset_path = config.dataset_path
        else:
            raise ValueError("arc_mt dataset_path must match the configured language")
        if self.dataset_name in {None, config.dataset_name}:
            self.dataset_name = config.dataset_name
        else:
            raise ValueError("arc_mt dataset_name must match the configured language")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"arc_mt_{self.language}"

    def result_metadata(self) -> dict[str, Any]:
        metadata = super().result_metadata()
        metadata["language"] = self.language
        return metadata

    def build_sample(self, doc: dict[str, Any], *, index: int):
        sample = super().build_sample(doc, index=index)
        sample.metadata["language"] = self.language
        return sample


def arc_mt(*, language: str, **kwargs: Any) -> ARCMT:
    return ARCMT(language=language, **kwargs)


def arc_mt_da(**kwargs: Any) -> ARCMT:
    return arc_mt(language="da", **kwargs)


def arc_mt_de(**kwargs: Any) -> ARCMT:
    return arc_mt(language="de", **kwargs)


def arc_mt_el(**kwargs: Any) -> ARCMT:
    return arc_mt(language="el", **kwargs)


def arc_mt_es(**kwargs: Any) -> ARCMT:
    return arc_mt(language="es", **kwargs)


def arc_mt_fi(**kwargs: Any) -> ARCMT:
    return arc_mt(language="fi", **kwargs)


def arc_mt_hu(**kwargs: Any) -> ARCMT:
    return arc_mt(language="hu", **kwargs)


def arc_mt_is(**kwargs: Any) -> ARCMT:
    return arc_mt(language="is", **kwargs)


def arc_mt_it(**kwargs: Any) -> ARCMT:
    return arc_mt(language="it", **kwargs)


def arc_mt_nb(**kwargs: Any) -> ARCMT:
    return arc_mt(language="nb", **kwargs)


def arc_mt_pl(**kwargs: Any) -> ARCMT:
    return arc_mt(language="pl", **kwargs)


def arc_mt_pt(**kwargs: Any) -> ARCMT:
    return arc_mt(language="pt", **kwargs)


def arc_mt_sv(**kwargs: Any) -> ARCMT:
    return arc_mt(language="sv", **kwargs)
