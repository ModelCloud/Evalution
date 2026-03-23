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


def _pile_word_count(text: str) -> int:
    return len(text.split())


def _pile_byte_count(text: str) -> int:
    return len(text.encode("utf-8"))


@dataclass(slots=True)
class Pile10K(BaseRollingPerplexitySuite):
    dataset_path: str = "monology/pile-uncopyrighted"
    dataset_name: str | None = None
    split: str = "train"
    max_rows: int | None = 10_000
    batch_size: int | None = None
    cache_dir: str | None = None
    streaming: bool = True

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "pile_10k"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> RollingPerplexitySample:
        text = str(doc["text"])
        meta = doc.get("meta")
        pile_set_name = str(meta.get("pile_set_name", "")) if isinstance(meta, dict) else ""
        return RollingPerplexitySample(
            index=index,
            source_text=text,
            scored_text=text,
            word_count=_pile_word_count(text),
            byte_count=_pile_byte_count(text),
            metadata={
                "text_preview": preview_text(text),
                "text_char_count": len(text),
                "pile_set_name": pile_set_name,
            },
        )


def pile_10k(**kwargs: Any) -> Pile10K:
    return Pile10K(**kwargs)
