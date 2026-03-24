# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset
import pcre

from evalution.benchmarks.rolling_perplexity import (
    BaseRollingPerplexitySuite,
    RollingPerplexitySample,
    preview_text,
)


def _wikitext_detokenizer(doc: dict[str, Any]) -> str:
    string = str(doc["page"])
    string = string.replace("s '", "s'")
    string = pcre.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = pcre.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = pcre.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = pcre.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = pcre.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = pcre.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def _wikitext_word_count(page: str) -> int:
    return len(pcre.split(r"\s+", page))


def _wikitext_byte_count(page: str) -> int:
    return len(page.encode("utf-8"))


@dataclass(slots=True)
class WikiText(BaseRollingPerplexitySuite):
    dataset_path: str = "EleutherAI/wikitext_document_level"
    dataset_name: str | None = "wikitext-2-raw-v1"
    split: str = "test"
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    streaming: bool = False

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "wikitext"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> RollingPerplexitySample:
        page = str(doc["page"])
        text = _wikitext_detokenizer(doc)
        return RollingPerplexitySample(
            index=index,
            source_text=page,
            scored_text=text,
            word_count=_wikitext_word_count(page),
            byte_count=_wikitext_byte_count(page),
            metadata={
                "page_preview": preview_text(page),
                "detokenized_preview": preview_text(text),
                "page_char_count": len(page),
            },
        )


def wikitext(**kwargs: Any) -> WikiText:
    return WikiText(**kwargs)
