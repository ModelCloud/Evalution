# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pcre
from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.choice_label import choice_label_exact_match

GPQA_SUBSETS = ("main", "diamond", "extended")
GPQA_TASKS = tuple(f"gpqa_{subset}" for subset in GPQA_SUBSETS)

_CHOICE_LABELS = ("A", "B", "C", "D")
_INVALID_CHOICE = "[invalid]"
_STOP_STRINGS = ("\n\n", "\nQuestion:", "</s>", "<|im_end|>", "<|eot_id|>")
_NON_ALNUM_PATTERN = pcre.compile(r"[^a-z0-9]+")
_EXPLICIT_ANSWER_PATTERNS = (
    pcre.compile(r"(?i)\bthe correct answer is\s*\(?([A-D])\)?"),
    pcre.compile(r"(?i)\bthe answer is\s*\(?([A-D])\)?"),
    pcre.compile(r"(?i)\banswer\s*[:\-]\s*\(?([A-D])\)?"),
)
_CHOICE_TOKEN_PATTERN = pcre.compile(r"\b([A-D])\b")


def _dataset_name_for_subset(subset: str) -> str:
    return f"gpqa_{subset}"


def _normalize_choice_text(text: Any) -> str:
    normalized = _NON_ALNUM_PATTERN.sub(" ", str(text).lower())
    return normalized.strip()


def _extract_choice_label(text: str, valid_labels: set[str]) -> str:
    response = text or ""
    for pattern in _EXPLICIT_ANSWER_PATTERNS:
        for match in pattern.findall(response):
            candidate = str(match).strip().upper()
            if candidate in valid_labels:
                return candidate

    matches = list(_CHOICE_TOKEN_PATTERN.findall(response))
    for match in reversed(matches):
        candidate = str(match).strip().upper()
        if candidate in valid_labels:
            return candidate
    return _INVALID_CHOICE


def _extract_choice_label_from_text(text: str, choice_texts: list[str]) -> str:
    normalized_response = _normalize_choice_text(text)
    if not normalized_response:
        return _INVALID_CHOICE

    exact_matches = [
        label
        for label, choice_text in zip(_CHOICE_LABELS, choice_texts, strict=True)
        if _normalize_choice_text(choice_text) == normalized_response
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    contained_matches = [
        (label, len(_normalize_choice_text(choice_text)))
        for label, choice_text in zip(_CHOICE_LABELS, choice_texts, strict=True)
        if _normalize_choice_text(choice_text)
        and _normalize_choice_text(choice_text) in normalized_response
    ]
    if not contained_matches:
        return _INVALID_CHOICE

    contained_matches.sort(key=lambda item: item[1], reverse=True)
    if len(contained_matches) > 1 and contained_matches[0][1] == contained_matches[1][1]:
        return _INVALID_CHOICE
    return contained_matches[0][0]


def _gpqa_prompt(question: str, choice_texts: list[str]) -> str:
    lines = [f"What is the correct answer to this question: {question.strip()}", "", "Choices:"]
    for label, choice_text in zip(_CHOICE_LABELS, choice_texts, strict=True):
        lines.append(f"({label}) {choice_text.strip()}")
    lines.append("")
    lines.append('Format your response as follows: "The correct answer is (insert answer here)"')
    return "\n".join(lines)


def _shuffled_choice_payload(doc: dict[str, Any], *, rng: random.Random) -> tuple[list[str], str]:
    shuffled_choices = [
        ("incorrect", str(doc["Incorrect Answer 1"]).strip()),
        ("incorrect", str(doc["Incorrect Answer 2"]).strip()),
        ("incorrect", str(doc["Incorrect Answer 3"]).strip()),
        ("correct", str(doc["Correct Answer"]).strip()),
    ]
    rng.shuffle(shuffled_choices)
    choice_texts = [choice_text for _, choice_text in shuffled_choices]
    gold_index = next(index for index, (kind, _) in enumerate(shuffled_choices) if kind == "correct")
    return choice_texts, _CHOICE_LABELS[gold_index]


@dataclass(slots=True)
class GPQA(BaseTestSuite):
    dataset_path: str = "Idavidrein/gpqa"
    dataset_name: str | None = None
    split: str = "train"
    subset: str = "main"
    seed: int = 0
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.subset not in GPQA_SUBSETS:
            raise ValueError(f"unsupported gpqa subset: {self.subset!r}")
        expected_dataset_name = _dataset_name_for_subset(self.subset)
        if self.dataset_name in {None, expected_dataset_name}:
            self.dataset_name = expected_dataset_name
            return
        raise ValueError("gpqa dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"gpqa_{self.subset}"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "subset": self.subset,
            "shuffle_seed": self.seed,
            "prompt_variant": "author_zero_shot_label_response",
            "choice_order_mode": "seeded_shuffle",
            "scoring_mode": "generated_choice_label_exact_match",
            "primary_metric": "em,choice_label",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        rng = random.Random(self.seed)
        for index, doc in enumerate(docs):
            choice_texts, gold_label = _shuffled_choice_payload(doc, rng=rng)
            prepared_doc = {
                **doc,
                "_choice_texts": choice_texts,
                "_gold_label": gold_label,
            }
            yield PreparedSample(
                index=index,
                doc=prepared_doc,
                target=gold_label,
                request=GenerationRequest(
                    prompt=_gpqa_prompt(str(doc["Question"]), choice_texts),
                    stop=list(_STOP_STRINGS),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        choice_texts = list(prepared_sample.doc["_choice_texts"])
        valid_labels = set(_CHOICE_LABELS[: len(choice_texts)])
        predicted_label = _extract_choice_label(output.text, valid_labels)
        if predicted_label == _INVALID_CHOICE:
            predicted_label = _extract_choice_label_from_text(output.text, choice_texts)

        if predicted_label == _INVALID_CHOICE:
            predicted_choice_text = _INVALID_CHOICE
        else:
            predicted_choice_text = choice_texts[_CHOICE_LABELS.index(predicted_label)]

        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "choice-label": predicted_label,
                "choice-text": predicted_choice_text,
            },
            scores={
                "em,choice_label": choice_label_exact_match(predicted_label, prepared_sample.target),
            },
            metadata={
                "subset": self.subset,
                "record_id": str(prepared_sample.doc["Record ID"]),
                "question": str(prepared_sample.doc["Question"]).strip(),
                "high_level_domain": str(prepared_sample.doc["High-level domain"]).strip(),
                "subdomain": str(prepared_sample.doc["Subdomain"]).strip(),
                "choice_labels": list(_CHOICE_LABELS[: len(choice_texts)]),
                "choice_texts": choice_texts,
                "gold_choice": choice_texts[_CHOICE_LABELS.index(prepared_sample.target)],
                "shuffle_seed": self.seed,
            },
        )


def gpqa(*, subset: str, **kwargs: Any) -> GPQA:
    return GPQA(subset=subset, **kwargs)


def gpqa_main(**kwargs: Any) -> GPQA:
    return gpqa(subset="main", **kwargs)


def gpqa_diamond(**kwargs: Any) -> GPQA:
    return gpqa(subset="diamond", **kwargs)


def gpqa_extended(**kwargs: Any) -> GPQA:
    return gpqa(subset="extended", **kwargs)
