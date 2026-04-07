# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pcre
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_DETOKENIZE_TRAILING_PUNCTUATION_RE = pcre.compile(r" (['.,])")


def _general_detokenize(text: str) -> str:
    text = text.replace(" n't", "n't")
    text = text.replace(" )", ")")
    text = text.replace("( ", "(")
    text = text.replace('" ', '"')
    text = text.replace(' "', '"')
    return _DETOKENIZE_TRAILING_PUNCTUATION_RE.sub(r"\1", text)


def _wsc_prompt(doc: dict[str, Any]) -> str:
    raw_passage = str(doc["text"])
    pronoun = str(doc["span2_text"])
    pronoun_index = int(doc["span2_index"])
    prefix = " ".join(raw_passage.split()[:pronoun_index])
    suffix_start = len(prefix) + len(pronoun) + 1
    suffix = raw_passage[suffix_start:]
    passage = _general_detokenize(f"{prefix} *{pronoun}*{suffix}")
    noun = str(doc["span1_text"])
    return (
        f"Passage: {passage}\n"
        f'Question: In the passage above, does the pronoun "*{pronoun}*" refer to "*{noun}*"?\n'
        "Answer:"
    )


@dataclass(slots=True)
class WSC(BaseMultipleChoiceSuite):
    dataset_path: str = "super_glue"
    dataset_name: str | None = "wsc.fixed"
    split: str = "validation"

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "wsc"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        metadata = {
            "noun": str(doc["span1_text"]),
            "pronoun": str(doc["span2_text"]),
            "span2_index": int(doc["span2_index"]),
        }
        if "idx" in doc:
            metadata["idx"] = int(doc["idx"])
        return MultipleChoiceSample(
            index=index,
            prompt=_wsc_prompt(doc),
            choices=["no", "yes"],
            gold_index=int(doc["label"]),
            metadata=metadata,
        )


def wsc(**kwargs: Any) -> WSC:
    return WSC(**kwargs)
