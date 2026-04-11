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


def _c4_word_count(text: str) -> int:
    """Implement c4 word count for this module."""
    return len(text.split())


def _c4_byte_count(text: str) -> int:
    """Implement c4 byte count for this module."""
    return len(text.encode("utf-8"))


@dataclass(slots=True)
class C4(BaseRollingPerplexitySuite):
    """Define the c4 helper class."""
    # Keep the class-level state explicit for this helper.
    dataset_path: str = "allenai/c4"
    dataset_name: str | None = "en"
    split: str = "validation"
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    stream: bool = True

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "c4"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> RollingPerplexitySample:
        """Build one benchmark sample from a dataset row."""
        text = str(doc["text"])
        return RollingPerplexitySample(
            index=index,
            source_text=text,
            scored_text=text,
            word_count=_c4_word_count(text),
            byte_count=_c4_byte_count(text),
            metadata={
                "text_preview": preview_text(text),
                "text_char_count": len(text),
                "url": str(doc.get("url", "")),
                "timestamp": str(doc.get("timestamp", "")),
            },
        )


def c4(**kwargs: Any) -> C4:
    """Implement c4 for this module."""
    return C4(**kwargs)
