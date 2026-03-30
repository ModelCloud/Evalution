# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from datasets import load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

CLICK_LANG_SUBSETS = ("text", "grammar", "function")
CLICK_CUL_SUBSETS = (
    "economy",
    "geography",
    "history",
    "kpop",
    "law",
    "politics",
    "society",
    "tradition",
)
CLICK_TASKS = (
    "click",
    "click_lang",
    "click_lang_text",
    "click_lang_grammar",
    "click_lang_function",
    "click_cul",
    "click_cul_economy",
    "click_cul_geography",
    "click_cul_history",
    "click_cul_kpop",
    "click_cul_law",
    "click_cul_politics",
    "click_cul_society",
    "click_cul_tradition",
)
_CLICK_CHOICE_LABELS = ("A", "B", "C", "D", "E")


def _click_prompt(doc: dict[str, Any]) -> str:
    context = str(doc["paragraph"]).strip()
    question = str(doc["question"]).strip()
    choices = [str(choice).strip() for choice in doc["choices"]]
    choice_labels = _CLICK_CHOICE_LABELS[: len(choices)]
    formatted_choices = ", ".join(
        f"{label}:{choice}" if index == 0 else f"{label}: {choice}"
        for index, (label, choice) in enumerate(zip(choice_labels, choices, strict=True))
    )
    labels = ", ".join(choice_labels)
    if context:
        return (
            "주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 "
            f"{labels} 중에 골라 알파벳 하나로 답하시오.\n\n"
            f"맥락: {context}\n질문: {question}\n보기:\n{formatted_choices}\n정답:"
        )
    return (
        "주어진 질문을 천천히 읽고, 적절한 정답을 "
        f"{labels} 중에 골라 알파벳 하나로 답하시오.\n\n"
        f"질문: {question}\n보기:\n{formatted_choices}\n정답:"
    )


def _click_choice_labels(doc: dict[str, Any]) -> list[str]:
    choice_count = len(doc["choices"])
    if choice_count < 2 or choice_count > len(_CLICK_CHOICE_LABELS):
        raise ValueError("click supports between two and five answer choices")
    if "CSAT" in str(doc["id"]):
        return list(_CLICK_CHOICE_LABELS[:5])
    return list(_CLICK_CHOICE_LABELS[:choice_count])


def _click_gold_index(doc: dict[str, Any]) -> int:
    answer = str(doc["answer"])
    choices = [str(choice) for choice in doc["choices"]]
    try:
        return choices.index(answer)
    except ValueError as exc:
        raise ValueError("click answer is not present in choices") from exc


def _is_click_lang_text(doc: dict[str, Any]) -> bool:
    doc_id = str(doc["id"])
    return (
        "CSAT_korean_22" in doc_id
        or ("CSAT_korean_23" in doc_id and int(doc_id.split("_")[-1]) < 35)
        or ("TK" in doc_id and int(doc_id.split("_")[-1]) > 4)
    )


def _is_click_lang_grammar(doc: dict[str, Any]) -> bool:
    doc_id = str(doc["id"])
    question = str(doc["question"])
    return (
        (
            "CSAT_korean" in doc_id
            and int(doc_id.split("_")[2]) < 21
            and int(doc_id.split("_")[3]) > 10
        )
        or (
            "Kedu_1" in doc_id
            and (
                doc_id.split("_")[1] != "16"
                or not any(token in question for token in ("대화", "발화", "질의"))
            )
        )
        or ("TK" in doc_id and int(doc_id.split("_")[-1]) < 5)
    )


def _is_click_lang_function(doc: dict[str, Any]) -> bool:
    doc_id = str(doc["id"])
    question = str(doc["question"])
    return (
        (
            "CSAT_korean" in doc_id
            and (
                int(doc_id.split("_")[-1]) > 34
                or (
                    int(doc_id.split("_")[2]) < 21
                    and int(doc_id.split("_")[3]) < 11
                )
            )
        )
        or (
            "Kedu_16" in doc_id
            and any(token in question for token in ("대화", "발화", "질의"))
        )
        or "PSE_korean" in doc_id
    )


def _is_click_cul_economy(doc: dict[str, Any]) -> bool:
    return "economy" in str(doc["id"]).lower()


def _is_click_cul_geography(doc: dict[str, Any]) -> bool:
    return "geography" in str(doc["id"]).lower()


def _is_click_cul_history(doc: dict[str, Any]) -> bool:
    doc_id = str(doc["id"])
    return "KHB" in doc_id or "history" in doc_id.lower()


def _is_click_cul_kpop(doc: dict[str, Any]) -> bool:
    return "popular" in str(doc["id"]).lower()


def _is_click_cul_law(doc: dict[str, Any]) -> bool:
    doc_id = str(doc["id"])
    return "law" in doc_id.lower() or "PSAT" in doc_id


def _is_click_cul_politics(doc: dict[str, Any]) -> bool:
    return "politics" in str(doc["id"]).lower()


def _is_click_cul_society(doc: dict[str, Any]) -> bool:
    return "society" in str(doc["id"]).lower()


def _is_click_cul_tradition(doc: dict[str, Any]) -> bool:
    return "tradition" in str(doc["id"]).lower()


_CLICK_FILTERS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "click": lambda _doc: True,
    "click_lang": lambda doc: (
        _is_click_lang_text(doc)
        or _is_click_lang_grammar(doc)
        or _is_click_lang_function(doc)
    ),
    "click_lang_text": _is_click_lang_text,
    "click_lang_grammar": _is_click_lang_grammar,
    "click_lang_function": _is_click_lang_function,
    "click_cul": lambda doc: (
        _is_click_cul_economy(doc)
        or _is_click_cul_geography(doc)
        or _is_click_cul_history(doc)
        or _is_click_cul_kpop(doc)
        or _is_click_cul_law(doc)
        or _is_click_cul_politics(doc)
        or _is_click_cul_society(doc)
        or _is_click_cul_tradition(doc)
    ),
    "click_cul_economy": _is_click_cul_economy,
    "click_cul_geography": _is_click_cul_geography,
    "click_cul_history": _is_click_cul_history,
    "click_cul_kpop": _is_click_cul_kpop,
    "click_cul_law": _is_click_cul_law,
    "click_cul_politics": _is_click_cul_politics,
    "click_cul_society": _is_click_cul_society,
    "click_cul_tradition": _is_click_cul_tradition,
}


@dataclass(slots=True)
class Click(BaseMultipleChoiceSuite):
    # CLIcK scores Korean linguistic and cultural knowledge with label-only answer continuations.
    dataset_path: str = "EunsuKim/CLIcK"
    dataset_name: str | None = None
    split: str = "train"
    stream: bool = (False)
    subset: str = "click"

    def __post_init__(self) -> None:
        if self.subset not in CLICK_TASKS:
            raise ValueError(f"unsupported click subset: {self.subset!r}")

    def dataset_loader(self) -> Any:
        subset = self.subset

        def loader(*args: Any, **kwargs: Any) -> Any:
            stream = kwargs.pop("stream", None)
            if stream is not None:
                kwargs["streaming"] = bool(stream)
            dataset = load_dataset(*args, **kwargs)
            if subset == "click":
                return dataset
            return dataset.filter(_CLICK_FILTERS[subset])

        return loader

    def task_name(self) -> str:
        return self.subset

    def result_metadata(self) -> dict[str, Any]:
        metadata = super().result_metadata()
        metadata["subset"] = self.subset
        return metadata

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        choice_labels = _click_choice_labels(doc)
        raw_choices = [str(choice).strip() for choice in doc["choices"]]
        return MultipleChoiceSample(
            index=index,
            prompt=_click_prompt(doc),
            choices=choice_labels,
            gold_index=_click_gold_index(doc),
            metadata={
                "subset": self.subset,
                "id": str(doc["id"]),
                "paragraph": str(doc["paragraph"]).strip(),
                "question": str(doc["question"]).strip(),
                "answer_text": str(doc["answer"]).strip(),
                "raw_choices": raw_choices,
                "choice_labels": choice_labels,
            },
        )


def click(**kwargs: Any) -> Click:
    return Click(subset="click", **kwargs)


def click_lang(**kwargs: Any) -> Click:
    return Click(subset="click_lang", **kwargs)


def click_cul(**kwargs: Any) -> Click:
    return Click(subset="click_cul", **kwargs)


def _make_click_factory(task_name: str) -> Any:
    def factory(**kwargs: Any) -> Click:
        return Click(subset=task_name, **kwargs)

    factory.__name__ = task_name
    return factory


for _task_name in CLICK_TASKS:
    globals()[_task_name] = _make_click_factory(_task_name)

del _task_name
