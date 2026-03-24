# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.scorers.classification import f1_for_label

_MC_TACO_DATA_FILES = {
    "validation": "https://raw.githubusercontent.com/CogComp/MCTACO/master/dataset/dev_3783.tsv",
    "test": "https://raw.githubusercontent.com/CogComp/MCTACO/master/dataset/test_9442.tsv",
}
_MC_TACO_COLUMNS = ["sentence", "question", "answer", "label", "category"]


def _mc_taco_prompt(sentence: str, question: str, answer: str) -> str:
    return (
        f"{sentence.strip()}\n"
        f"Question: {question.strip()}\n"
        f"Answer: {answer.strip()}\n"
        "Plausible:"
    )


def _load_mc_taco_dataset(
    dataset_path: str,
    *,
    split: str,
    cache_dir: str | None = None,
    streaming: bool = False,
) -> Any:
    if dataset_path != "CogComp/mc_taco":
        raise ValueError(f"unsupported MC-TACO dataset path: {dataset_path!r}")
    source_url = _MC_TACO_DATA_FILES.get(split)
    if source_url is None:
        raise ValueError(f"unsupported MC-TACO split: {split!r}")
    return load_dataset(
        "csv",
        data_files={split: source_url},
        delimiter="\t",
        column_names=_MC_TACO_COLUMNS,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )


@dataclass(slots=True)
class MCTACO(BaseMultipleChoiceSuite):
    dataset_path: str = "CogComp/mc_taco"
    # Align the default split with current benchmark-style harness usage.
    split: str = "test"

    def dataset_loader(self) -> Any:
        return _load_mc_taco_dataset

    def task_name(self) -> str:
        return "mc_taco"

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        label = str(doc["label"]).strip().lower()
        return MultipleChoiceSample(
            index=index,
            prompt=_mc_taco_prompt(str(doc["sentence"]), str(doc["question"]), str(doc["answer"])),
            choices=["no", "yes"],
            gold_index=0 if label == "no" else 1,
            metadata={
                "sentence": str(doc["sentence"]),
                "question": str(doc["question"]),
                "answer": str(doc["answer"]),
                "category": str(doc["category"]),
            },
        )

    def extra_metrics(
        self,
        *,
        samples: list[MultipleChoiceSample],
        raw_predictions: list[int],
        normalized_predictions: list[int],
    ) -> dict[str, float]:
        gold_labels = [sample.gold_index for sample in samples]
        return {
            "f1,ll_yes": f1_for_label(gold_labels, raw_predictions, label=1),
            "f1,ll_avg_yes": f1_for_label(gold_labels, normalized_predictions, label=1),
        }


def mc_taco(**kwargs: Any) -> MCTACO:
    return MCTACO(**kwargs)
