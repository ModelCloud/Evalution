# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.rolling_perplexity import (
    BaseRollingPerplexitySuite,
    RollingPerplexitySample,
    preview_text,
)
from evalution.benchmarks.subsets import normalize_subset_token

# Preserve the upstream subset ids for dataset loading while deriving normalized task names separately.
PALOMA_SUBSETS = (
    "4chan_meta_sep",
    "c4_100_domains",
    "c4_en",
    "dolma-v1_5",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "falcon-refinedweb",
    "gab",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep",
    "mc4",
    "ptb",
    "redpajama",
    "twitterAAE_HELM_fixed",
    "wikitext_103",
)
PALOMA_SUBSET_BY_TOKEN = {
    normalize_subset_token(subset): subset
    for subset in PALOMA_SUBSETS
}
PALOMA_TASKS = tuple(f"paloma_{token}" for token in PALOMA_SUBSET_BY_TOKEN)


def _paloma_task_token(subset: str) -> str:
    """Implement paloma task token for this module."""
    return normalize_subset_token(subset)


def _paloma_word_count(text: str) -> int:
    """Implement paloma word count for this module."""
    return len(text.split())


def _paloma_byte_count(text: str) -> int:
    """Implement paloma byte count for this module."""
    return len(text.encode("utf-8"))


@dataclass(slots=True)
class Paloma(BaseRollingPerplexitySuite):
    # Score one PALOMA subset as rolling perplexity over its raw text field.
    """Define the paloma helper class."""
    dataset_path: str = "allenai/paloma"
    dataset_name: str | None = "c4_en"
    split: str = "test"
    stream: bool = True
    subset: str = "c4_en"

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
        raw_subset = PALOMA_SUBSET_BY_TOKEN.get(_paloma_task_token(self.subset))
        if raw_subset is None:
            raise ValueError(f"unsupported paloma subset: {self.subset!r}")
        self.subset = raw_subset
        if self.dataset_name in {None, raw_subset}:
            self.dataset_name = raw_subset
            return
        raise ValueError("paloma dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"paloma_{_paloma_task_token(self.subset)}"

    def result_metadata(self) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        metadata = super().result_metadata()
        metadata["subset"] = self.subset
        return metadata

    def build_sample(self, doc: dict[str, Any], *, index: int) -> RollingPerplexitySample:
        """Build one benchmark sample from a dataset row."""
        text = str(doc["text"])
        return RollingPerplexitySample(
            index=index,
            source_text=text,
            scored_text=text,
            word_count=_paloma_word_count(text),
            byte_count=_paloma_byte_count(text),
            metadata={
                "subset": self.subset,
                "text_preview": preview_text(text),
                "text_char_count": len(text),
            },
        )


def paloma(*, subset: str = "c4_en", **kwargs: Any) -> Paloma:
    # Accept either the raw subset id or its normalized task token.
    """Implement paloma for this module."""
    kwargs.setdefault("dataset_name", PALOMA_SUBSET_BY_TOKEN.get(_paloma_task_token(subset), subset))
    return Paloma(subset=subset, **kwargs)


def _make_paloma_factory(subset: str) -> Any:
    # Register normalized factory names so YAML and Python lookups stay identifier-safe.
    """Make paloma factory."""
    def factory(**kwargs: Any) -> Paloma:
        """Implement factory for this module."""
        return paloma(subset=subset, **kwargs)

    factory.__name__ = f"paloma_{_paloma_task_token(subset)}"
    return factory


for _subset in PALOMA_SUBSETS:
    globals()[f"paloma_{_paloma_task_token(_subset)}"] = _make_paloma_factory(_subset)

del _subset
