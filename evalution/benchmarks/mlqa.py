# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import string
import sys
from typing import Any
import unicodedata

from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult

# Keep MLQA language coverage explicit because both dataset selection and answer normalization depend on it.
MLQA_LANGUAGES = ("ar", "de", "en", "es", "hi", "vi", "zh")
MLQA_TASKS = tuple(
    f"mlqa_{context_language}_{question_language}"
    for context_language in MLQA_LANGUAGES
    for question_language in MLQA_LANGUAGES
)
_STOP_STRINGS = ("\n", "\nQuestion:", "\nContext:")
# Mirror MLQA's language-aware punctuation stripping once at import so per-sample scoring stays cheap.
_MLQA_PUNCT = {
    chr(index)
    for index in range(sys.maxunicode + 1)
    if unicodedata.category(chr(index)).startswith("P")
}.union(string.punctuation)
_WHITESPACE_LANGS = frozenset({"ar", "de", "en", "es", "hi", "vi"})
_MIXED_SEGMENTATION_LANGS = frozenset({"zh"})


def _mlqa_prompt(*, context: str, question: str) -> str:
    return f"Context: {context.strip()}\n\nQuestion: {question.strip()}\n\nAnswer:"


def _mlqa_answer_texts(doc: dict[str, Any]) -> list[str]:
    answers = doc["answers"]
    if isinstance(answers, dict):
        values = answers.get("text", [])
    else:
        values = answers
    deduped: list[str] = []
    for answer in values:
        text = str(answer).strip()
        if text and text not in deduped:
            deduped.append(text)
    if not deduped:
        raise ValueError("mlqa requires at least one non-empty answer")
    return deduped


def _whitespace_tokenize(text: str) -> list[str]:
    return text.split()


def _mixed_segmentation(text: str) -> list[str]:
    segments: list[str] = []
    buffered = ""
    for character in text:
        if re.search(r"[\u4e00-\u9fa5]", character) or character in _MLQA_PUNCT:
            if buffered:
                segments.extend(_whitespace_tokenize(buffered))
                buffered = ""
            segments.append(character)
            continue
        buffered += character
    if buffered:
        segments.extend(_whitespace_tokenize(buffered))
    return segments


def _normalize_mlqa_answer(text: str, language: str) -> str:
    lowered = text.lower()
    stripped_punctuation = "".join(character for character in lowered if character not in _MLQA_PUNCT)
    if language == "en":
        stripped_articles = re.sub(r"\b(a|an|the)\b", " ", stripped_punctuation)
    elif language == "es":
        stripped_articles = re.sub(r"\b(un|una|unos|unas|el|la|los|las)\b", " ", stripped_punctuation)
    elif language == "hi":
        stripped_articles = stripped_punctuation
    elif language == "vi":
        stripped_articles = re.sub(r"\b(của|là|cái|chiếc|những)\b", " ", stripped_punctuation)
    elif language == "de":
        stripped_articles = re.sub(
            r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b",
            " ",
            stripped_punctuation,
        )
    elif language == "ar":
        stripped_articles = re.sub(r"(?:^|\s)ال", " ", stripped_punctuation)
    elif language == "zh":
        stripped_articles = stripped_punctuation
    else:
        raise ValueError(f"unsupported mlqa language for normalization: {language!r}")

    if language in _WHITESPACE_LANGS:
        tokens = _whitespace_tokenize(stripped_articles)
    elif language in _MIXED_SEGMENTATION_LANGS:
        tokens = _mixed_segmentation(stripped_articles)
    else:
        raise ValueError(f"unsupported mlqa tokenization language: {language!r}")
    return " ".join(token for token in tokens if token.strip())


def _mlqa_exact_match(prediction: str, answer: str, language: str) -> float:
    return float(_normalize_mlqa_answer(prediction, language) == _normalize_mlqa_answer(answer, language))


def _mlqa_f1(prediction: str, answer: str, language: str) -> float:
    prediction_tokens = _normalize_mlqa_answer(prediction, language).split()
    answer_tokens = _normalize_mlqa_answer(answer, language).split()
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(prediction_tokens), 1)
    recall = overlap / max(len(answer_tokens), 1)
    return (2 * precision * recall) / (precision + recall)


def _best_mlqa_scores(prediction: str, answers: list[str], language: str) -> tuple[float, float, int]:
    best_exact = 0.0
    best_f1 = 0.0
    best_index = 0
    for index, answer in enumerate(answers):
        exact = _mlqa_exact_match(prediction, answer, language)
        f1_score = _mlqa_f1(prediction, answer, language)
        if exact > best_exact or (exact == best_exact and f1_score > best_f1):
            best_exact = exact
            best_f1 = f1_score
            best_index = index
    return best_exact, best_f1, best_index


@dataclass(slots=True)
class MLQA(BaseTestSuite):
    # Recreate MLQA's language-paired extractive QA evaluation with context-language normalization.
    dataset_path: str = "facebook/mlqa"
    dataset_name: str | None = "mlqa.en.en"
    split: str = "test"
    context_language: str = "en"
    question_language: str = "en"
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.context_language not in MLQA_LANGUAGES:
            raise ValueError(f"unsupported mlqa context language: {self.context_language!r}")
        if self.question_language not in MLQA_LANGUAGES:
            raise ValueError(f"unsupported mlqa question language: {self.question_language!r}")
        expected_dataset_name = f"mlqa.{self.context_language}.{self.question_language}"
        if self.dataset_name in {None, expected_dataset_name}:
            self.dataset_name = expected_dataset_name
            return
        raise ValueError("mlqa dataset_name must match the configured language pair")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return f"mlqa_{self.context_language}_{self.question_language}"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_mlqa_exact_match_f1",
            "primary_metric": "f1",
            "context_language": self.context_language,
            "question_language": self.question_language,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            answers = _mlqa_answer_texts(doc)
            yield PreparedSample(
                index=index,
                doc=doc,
                target=answers[0],
                request=GenerationRequest(
                    prompt=_mlqa_prompt(
                        context=str(doc["context"]),
                        question=str(doc["question"]),
                    ),
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
        answers = _mlqa_answer_texts(prepared_sample.doc)
        exact, f1_score, best_index = _best_mlqa_scores(
            output.text,
            answers,
            self.context_language,
        )
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-normalized": _normalize_mlqa_answer(output.text, self.context_language),
                "best_answer_index": str(best_index),
                "best_answer": answers[best_index],
            },
            scores={
                "em": exact,
                "f1": f1_score,
            },
            metadata={
                "id": str(prepared_sample.doc.get("id", "")),
                "context_language": self.context_language,
                "question_language": self.question_language,
                "context": str(prepared_sample.doc["context"]),
                "question": str(prepared_sample.doc["question"]),
                "answer_texts": answers,
            },
        )


def mlqa(*, context_language: str, question_language: str, **kwargs: Any) -> MLQA:
    # Lock YAML and Python construction to one canonical dataset config string per language pair.
    kwargs.setdefault("dataset_name", f"mlqa.{context_language}.{question_language}")
    return MLQA(
        context_language=context_language,
        question_language=question_language,
        **kwargs,
    )


def _make_mlqa_factory(context_language: str, question_language: str) -> Any:
    # Emit one stable factory per context/question language pair for registry discovery.
    def factory(**kwargs: Any) -> MLQA:
        return mlqa(
            context_language=context_language,
            question_language=question_language,
            **kwargs,
        )

    factory.__name__ = f"mlqa_{context_language}_{question_language}"
    return factory


for _context_language in MLQA_LANGUAGES:
    for _question_language in MLQA_LANGUAGES:
        globals()[f"mlqa_{_context_language}_{_question_language}"] = _make_mlqa_factory(
            _context_language,
            _question_language,
        )

del _context_language
del _question_language
