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

# Keep benchmark defaults and public task ids explicit at module scope.
_SLASH_NUMBER_RE = pcre.compile(r"/' [0-9]/")
_PAREN_SPACING_RE = pcre.compile(r"\(\s*([^\)]*?)\s*\)")
_BRACKET_SPACING_RE = pcre.compile(r"\[\s*([^\]]*?)\s*\]")
_BRACE_SPACING_RE = pcre.compile(r"{\s*([^}]*?)\s*}")
_DOUBLE_QUOTE_SPACING_RE = pcre.compile(r"\"\s*([^\"]*?)\s*\"")
_SINGLE_QUOTE_SPACING_RE = pcre.compile(r"'\s*([^']*?)\s*'")
_WHITESPACE_SPLIT_RE = pcre.compile(r"\s+")


def _wikitext_detokenizer(doc: dict[str, Any]) -> str:
    """Implement wikitext detokenizer for this module."""
    string = str(doc["page"])
    string = string.replace("s '", "s'")
    string = _SLASH_NUMBER_RE.sub(r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = _PAREN_SPACING_RE.sub(r"(\1)", string)
    string = _BRACKET_SPACING_RE.sub(r"[\1]", string)
    string = _BRACE_SPACING_RE.sub(r"{\1}", string)
    string = _DOUBLE_QUOTE_SPACING_RE.sub(r'"\1"', string)
    string = _SINGLE_QUOTE_SPACING_RE.sub(r"'\1'", string)
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
    """Implement wikitext word count for this module."""
    return len(_WHITESPACE_SPLIT_RE.split(page))


def _wikitext_byte_count(page: str) -> int:
    """Implement wikitext byte count for this module."""
    return len(page.encode("utf-8"))


@dataclass(slots=True)
class WikiText(BaseRollingPerplexitySuite):
    """Define the wiki text helper class."""
    # Keep the class-level state explicit for this helper.
    dataset_path: str = "EleutherAI/wikitext_document_level"
    dataset_name: str | None = "wikitext-2-raw-v1"
    split: str = "test"
    stream: bool = True
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "wikitext"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> RollingPerplexitySample:
        """Build one benchmark sample from a dataset row."""
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
    """Implement wikitext for this module."""
    return WikiText(**kwargs)
