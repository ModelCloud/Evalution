# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Keep the public task names explicit because PortugueseBench exposes ASSIN as two distinct evaluation modes.
ASSIN_VARIANTS = ("assin_entailment", "assin_paraphrase")


def _assin_entailment_choices(premise: str, hypothesis: str) -> list[str]:
    return [
        f"{premise}, certo? Também, {hypothesis}",
        f"{premise}, certo? Sim, {hypothesis}",
    ]


def _assin_paraphrase_choices(premise: str, hypothesis: str) -> list[str]:
    return [
        f"{premise}, certo? Não, {hypothesis}",
        f"{premise}, certo? Sim, {hypothesis}",
    ]


_ASSIN_CHOICE_BUILDERS = {
    "assin_entailment": _assin_entailment_choices,
    "assin_paraphrase": _assin_paraphrase_choices,
}


@dataclass(slots=True)
class ASSIN(BaseMultipleChoiceSuite):
    # Score Portuguese ASSIN entailment and paraphrase with the benchmark's fixed two-choice verbalizations.
    dataset_path: str = "nilc-nlp/assin"
    dataset_name: str | None = None
    split: str = "test"
    variant: str = "assin_entailment"

    def __post_init__(self) -> None:
        if self.variant not in ASSIN_VARIANTS:
            raise ValueError(f"unsupported assin variant: {self.variant!r}")
        if self.dataset_name is not None:
            raise ValueError("assin does not use a dataset_name")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return self.variant

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        premise = str(doc["premise"]).strip()
        hypothesis = str(doc["hypothesis"]).strip()
        choices = _ASSIN_CHOICE_BUILDERS[self.variant](premise, hypothesis)
        gold_index = 0 if int(doc["entailment_judgment"]) == 0 else 1
        return MultipleChoiceSample(
            index=index,
            prompt="",
            choices=choices,
            gold_index=gold_index,
            metadata={
                "variant": self.variant,
                "sentence_pair_id": str(doc["sentence_pair_id"]),
                "premise": premise,
                "hypothesis": str(doc["hypothesis"]).strip(),
                "relatedness_score": float(doc["relatedness_score"]),
            },
        )


def assin(*, variant: str = "assin_entailment", **kwargs: Any) -> ASSIN:
    if variant not in ASSIN_VARIANTS:
        raise ValueError(f"unsupported assin variant: {variant!r}")
    return ASSIN(variant=variant, **kwargs)


def assin_entailment(**kwargs: Any) -> ASSIN:
    return assin(variant="assin_entailment", **kwargs)


def assin_paraphrase(**kwargs: Any) -> ASSIN:
    return assin(variant="assin_paraphrase", **kwargs)
