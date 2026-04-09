# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

# Preserve the upstream task ids and dataset config names so YAML, Python, and result payloads stay stable.
LONG_BENCH2_TASK_TO_DATASET = {
    "longbench2_academic_multi": "academic_multi",
    "longbench2_academic_single": "academic_single",
    "longbench2_agent_history": "agent_history_qa",
    "longbench2_code": "code_repo_qa",
    "longbench2_detective": "detective",
    "longbench2_dialogue_history": "dialogue_history_qa",
    "longbench2_event_order": "event_ordering",
    "longbench2_fin_multi": "financial_multi",
    "longbench2_fin_single": "financial_single",
    "longbench2_govt_multi": "government_multi",
    "longbench2_govt_single": "government_single",
    "longbench2_graph": "graph_reasoning",
    "longbench2_legal_multi": "legal_multi",
    "longbench2_legal_single": "legal_single",
    "longbench2_lit_single": "literary",
    "longbench2_many_shot": "manyshot_learning",
    "longbench2_news_multi": "multinews",
    "longbench2_table": "table_qa",
    "longbench2_translate": "new_language_translation",
    "longbench2_user_guide": "user_guide_qa",
}
LONG_BENCH2_TASKS = tuple(LONG_BENCH2_TASK_TO_DATASET)
_CHOICE_LABELS = ["A", "B", "C", "D"]
_LONG_BENCH2_ALIAS_TO_TASK = {
    alias: task
    for task, dataset_name in LONG_BENCH2_TASK_TO_DATASET.items()
    for alias in (task, task.removeprefix("longbench2_"), dataset_name)
}


def _longbench2_task_name(value: str) -> str:
    task_name = _LONG_BENCH2_ALIAS_TO_TASK.get(value)
    if task_name is None:
        raise ValueError(f"unsupported longbench2 subset: {value!r}")
    return task_name


def _longbench2_prompt(*, context: str, question: str, choice_texts: list[str]) -> str:
    if len(choice_texts) != len(_CHOICE_LABELS):
        raise ValueError(f"longbench2 expects four choices, got {len(choice_texts)}")
    lines = [
        "Please read the following text and answer the question below.",
        "",
        "<text>",
        context.strip(),
        "</text>",
        "",
        f"What is the correct answer to this question: {question.strip()}",
        "Choices:",
    ]
    for label, choice_text in zip(_CHOICE_LABELS, choice_texts, strict=True):
        lines.append(f"({label}) {choice_text}")
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


@dataclass(slots=True)
class LongBench2(BaseMultipleChoiceSuite):
    # Score one LongBench v2 config by ranking answer labels against the shared author prompt.
    dataset_path: str = "recursal/longbench-v2"
    dataset_name: str | None = "academic_single"
    split: str = "train"
    subset: str = "academic_single"

    def __post_init__(self) -> None:
        task_name = _longbench2_task_name(self.subset)
        dataset_name = LONG_BENCH2_TASK_TO_DATASET[task_name]
        self.subset = task_name
        if self.dataset_name in {None, dataset_name}:
            self.dataset_name = dataset_name
            return
        raise ValueError("longbench2 dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return self.subset

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choice_texts = [str(choice).strip() for choice in doc["choices"]]
        prompt = _longbench2_prompt(
            context=str(doc["context"]),
            question=str(doc["question"]),
            choice_texts=choice_texts,
        )
        gold_index = int(doc["answer"])
        return MultipleChoiceSample(
            index=index,
            prompt=prompt,
            choices=list(_CHOICE_LABELS),
            gold_index=gold_index,
            metadata={
                "dataset_name": self.dataset_name,
                "domain": str(doc.get("domain", "")).strip(),
                "difficulty": str(doc.get("difficulty", "")).strip(),
                "length": str(doc.get("length", "")).strip(),
                "choice_texts": choice_texts,
            },
        )


def longbench2(*, subset: str = "academic_single", **kwargs: Any) -> LongBench2:
    task_name = _longbench2_task_name(subset)
    kwargs.setdefault("dataset_name", LONG_BENCH2_TASK_TO_DATASET[task_name])
    return LongBench2(subset=task_name, **kwargs)


def _make_longbench2_factory(task_name: str) -> Any:
    def factory(**kwargs: Any) -> LongBench2:
        return longbench2(subset=task_name, **kwargs)

    factory.__name__ = task_name
    return factory


for _task_name in LONG_BENCH2_TASKS:
    globals()[_task_name] = _make_longbench2_factory(_task_name)

del _task_name
