from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pcre
from datasets import load_dataset

from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.suites.base import BaseTestSuite
from evalution.suites.execution import PreparedSample

_INVALID_CHOICE = "[invalid]"
_STOP_STRINGS = ("</s>", "<|im_end|>", "<|eot_id|>")
_CHOICE_LABEL_PATTERNS = (
    pcre.compile(r"(?i)\banswer(?:\s+is)?\s*[:\-]?\s*\(?([A-Z0-9])\)?\b"),
    pcre.compile(r"(?i)\b(?:option|choice)\s*\(?([A-Z0-9])\)?\b"),
    pcre.compile(r"^\s*\(?([A-Z0-9])\)?(?:[\)\].:\-]|$)"),
)


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    label: str
    text: str


def _normalize_choice_label(label: Any) -> str:
    return str(label).strip().upper()


def _normalize_choice_text(text: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return normalized.strip()


def _choice_options(doc: dict[str, Any]) -> list[ChoiceOption]:
    choices = doc.get("choices") or {}
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    if len(labels) != len(texts):
        raise ValueError("ARC-Challenge choices must provide matching label/text lists")
    return [
        ChoiceOption(
            label=_normalize_choice_label(label),
            text=str(text).strip(),
        )
        for label, text in zip(labels, texts, strict=True)
    ]


def _correct_choice(doc: dict[str, Any]) -> ChoiceOption:
    options = _choice_options(doc)
    option_by_label = {option.label: option for option in options}
    answer_label = _normalize_choice_label(doc.get("answerKey"))
    if answer_label not in option_by_label:
        raise ValueError(f"ARC-Challenge answerKey {answer_label!r} is not present in choices")
    return option_by_label[answer_label]


def _choice_block(doc: dict[str, Any]) -> str:
    return "\n".join(f"{option.label}. {option.text}" for option in _choice_options(doc))


def _prompt_text(doc: dict[str, Any]) -> str:
    return (
        f"Question: {doc['question']}\n"
        f"Choices:\n{_choice_block(doc)}\n"
        "Answer with the correct choice label.\n"
        "Answer:"
    )


# Prefer explicit answer markers, then fall back to a leading label token.
def _extract_choice_label(text: str, valid_labels: set[str]) -> str:
    response = text or ""
    for pattern in _CHOICE_LABEL_PATTERNS:
        for match in pattern.findall(response):
            candidate = _normalize_choice_label(match)
            if candidate in valid_labels:
                return candidate

    candidate = _normalize_choice_label(response)
    if candidate in valid_labels:
        return candidate
    return _INVALID_CHOICE


# Only infer a label from answer text when exactly one option text matches.
def _extract_choice_label_from_text(text: str, options: list[ChoiceOption]) -> str:
    normalized_response = _normalize_choice_text(text)
    if not normalized_response:
        return _INVALID_CHOICE

    normalized_options = [
        (option.label, option.text, _normalize_choice_text(option.text))
        for option in options
    ]
    exact_matches = [
        label
        for label, _text, normalized_text in normalized_options
        if normalized_text and normalized_text == normalized_response
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]

    contained_matches = [
        (label, len(normalized_text))
        for label, _text, normalized_text in normalized_options
        if normalized_text and normalized_text in normalized_response
    ]
    if not contained_matches:
        return _INVALID_CHOICE

    contained_matches.sort(key=lambda item: item[1], reverse=True)
    if len(contained_matches) > 1 and contained_matches[0][1] == contained_matches[1][1]:
        return _INVALID_CHOICE
    return contained_matches[0][0]


@dataclass(slots=True)
class ARCChallenge(BaseTestSuite):
    dataset_path: str = "allenai/ai2_arc"
    dataset_name: str | None = "ARC-Challenge"
    split: str = "test"
    apply_chat_template: bool = False
    max_new_tokens: int = 8
    do_sample: bool = False
    temperature: float = 0.0

    # Use the Hugging Face datasets loader for the ARC-Challenge benchmark.
    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "arc_challenge"

    # Show label accuracy and invalid parses in the live progress title.
    def score_progress_title(
        self,
        *,
        processed: int,
        aggregate_scores: dict[str, float],
        invalid_predictions: int,
    ) -> str:
        accuracy = (
            aggregate_scores.get("exact_match,choice-label", 0.0) / processed
            if processed
            else 0.0
        )
        return (
            f"{self.task_name()}: scoring "
            f"accuracy={accuracy:.4f} "
            f"invalid={invalid_predictions}"
        )

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(
                generation_submission_mode=generation_submission_mode,
            ),
            "apply_chat_template": self.apply_chat_template,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            correct_choice = _correct_choice(doc)
            yield PreparedSample(
                index=index,
                doc=doc,
                target=f"{correct_choice.label}. {correct_choice.text}",
                request=self._build_request(doc),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        options = _choice_options(prepared_sample.doc)
        option_by_label = {option.label: option for option in options}
        correct_choice = _correct_choice(prepared_sample.doc)

        predicted_label = _extract_choice_label(output.text, set(option_by_label))
        if predicted_label == _INVALID_CHOICE:
            predicted_label = _extract_choice_label_from_text(output.text, options)
        predicted_text = (
            option_by_label[predicted_label].text
            if predicted_label in option_by_label
            else _INVALID_CHOICE
        )

        scores = {
            "exact_match,choice-label": float(predicted_label == correct_choice.label),
            "exact_match,choice-text": float(
                _normalize_choice_text(predicted_text) == _normalize_choice_text(correct_choice.text)
            ),
        }
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "choice-label": predicted_label,
                "choice-text": predicted_text,
            },
            scores=scores,
            metadata=self._sample_metadata(prepared_sample.doc),
        )

    def invalid_prediction_count(self, sample: SampleResult) -> int:
        return int(sample.extracted["choice-label"] == _INVALID_CHOICE)

    def _build_request(self, doc: dict[str, Any]) -> GenerationRequest:
        prompt = _prompt_text(doc)
        if self.apply_chat_template:
            return GenerationRequest(
                messages=[{"role": "user", "content": prompt}],
                stop=list(_STOP_STRINGS),
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
            )
        return GenerationRequest(
            prompt=prompt,
            stop=list(_STOP_STRINGS),
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
        )

    def _sample_metadata(self, doc: dict[str, Any]) -> dict[str, Any]:
        correct_choice = _correct_choice(doc)
        return {
            "id": doc.get("id"),
            "answer_label": correct_choice.label,
            "choices": [
                {
                    "label": option.label,
                    "text": option.text,
                }
                for option in _choice_options(doc)
            ],
        }


# Convenience constructor mirroring the public suite factory style.
def arc_challenge(**kwargs: Any) -> ARCChallenge:
    return ARCChallenge(**kwargs)
