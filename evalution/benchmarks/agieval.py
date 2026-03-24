# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from datasets import get_dataset_config_names, load_dataset

from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample

_UNSUPPORTED_SUBSETS = frozenset(
    {
        "gaokao-mathcloze",
        "jec-qa-ca",
        "jec-qa-kd",
        "math",
    }
)
AGIEVAL_SUBSETS = tuple(
    subset
    for subset in get_dataset_config_names("RUCAIBox/AGIEval")
    if subset not in _UNSUPPORTED_SUBSETS
)
_OPTION_LABEL_RE = re.compile(r"^\(?([A-Z])\)?(?:[.:：、]|．)?\s*")


def _slugify_subset_name(subset: str) -> str:
    return subset.replace("-", "_")


AGIEVAL_TASKS = tuple(f"agieval_{_slugify_subset_name(subset)}" for subset in AGIEVAL_SUBSETS)
_SUBSET_TO_TASK = dict(zip(AGIEVAL_SUBSETS, AGIEVAL_TASKS, strict=True))


def _agieval_prompt(*, passage: str | None, question: str, choices: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    if passage:
        lines.extend(("Passage:", passage, ""))
    lines.append(f"Question: {question}")
    for label, choice in choices:
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def _parse_choice(option: str, *, index: int) -> tuple[str, str]:
    normalized = str(option).strip()
    default_label = chr(ord("A") + index)
    match = _OPTION_LABEL_RE.match(normalized)
    if match is None:
        return default_label, normalized
    label = match.group(1)
    choice_text = normalized[match.end() :].strip()
    if not choice_text:
        choice_text = normalized
    return label, choice_text


@dataclass(slots=True)
class AGIEval(BaseMultipleChoiceSuite):
    dataset_path: str = "RUCAIBox/AGIEval"
    dataset_name: str | None = None
    split: str = "test"
    subset: str = ""

    def __post_init__(self) -> None:
        if self.subset not in AGIEVAL_SUBSETS:
            raise ValueError(f"unsupported agieval subset: {self.subset!r}")
        if self.dataset_name in {None, self.subset}:
            self.dataset_name = self.subset
            return
        raise ValueError("agieval dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return _SUBSET_TO_TASK[self.subset]

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        raw_options = doc["options"]
        if not isinstance(raw_options, list) or not raw_options:
            raise ValueError(f"agieval subset {self.subset!r} does not expose answer options")
        answer_label = doc["label"]
        if not isinstance(answer_label, str):
            raise ValueError(f"agieval subset {self.subset!r} does not use single answer labels")

        parsed_choices = [_parse_choice(option, index=option_index) for option_index, option in enumerate(raw_options)]
        choice_labels = [label for label, _choice_text in parsed_choices]
        answer_label = answer_label.strip()
        if answer_label not in choice_labels:
            raise ValueError(f"agieval answer label {answer_label!r} is not present in subset {self.subset!r}")

        passage = None if doc["passage"] is None else str(doc["passage"]).strip() or None
        question = str(doc["question"]).strip()
        return MultipleChoiceSample(
            index=index,
            prompt=_agieval_prompt(
                passage=passage,
                question=question,
                choices=parsed_choices,
            ),
            choices=choice_labels,
            gold_index=choice_labels.index(answer_label),
            metadata={
                "subset": self.subset,
                "passage": passage,
                "question": question,
                "answer_label": answer_label,
                "choice_labels": choice_labels,
                "raw_choices": [choice_text for _label, choice_text in parsed_choices],
                "solution": None if doc["other"] is None else doc["other"],
                "explanation": None if doc["explanation"] is None else str(doc["explanation"]).strip() or None,
            },
        )


def agieval(*, subset: str, **kwargs: Any) -> AGIEval:
    return AGIEval(subset=subset, dataset_name=subset, **kwargs)


def _make_agieval_factory(subset: str) -> Any:
    def factory(**kwargs: Any) -> AGIEval:
        return agieval(subset=subset, **kwargs)

    factory.__name__ = _SUBSET_TO_TASK[subset]
    return factory


for _subset in AGIEVAL_SUBSETS:
    globals()[_SUBSET_TO_TASK[_subset]] = _make_agieval_factory(_subset)

del _subset
