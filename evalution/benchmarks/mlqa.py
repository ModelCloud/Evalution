# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
import string
import sys
from typing import Any
import unicodedata
from urllib.request import urlopen
import zipfile

from datasets import Dataset
import pcre

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
_MLQA_URL = "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"
_MLQA_SPLIT_DIRECTORY = {
    "test": "test",
    "validation": "dev",
    "dev": "dev",
}
_STOP_STRINGS = ("\n", "\nQuestion:", "\nContext:")
# Mirror MLQA's language-aware punctuation stripping once at import so per-sample scoring stays cheap.
_MLQA_PUNCT = {
    chr(index)
    for index in range(sys.maxunicode + 1)
    if unicodedata.category(chr(index)).startswith("P")
}.union(string.punctuation)
_WHITESPACE_LANGS = frozenset({"ar", "de", "en", "es", "hi", "vi"})
_MIXED_SEGMENTATION_LANGS = frozenset({"zh"})
# Keep language-specific normalizers compiled once because answer scoring hits them for every prediction/reference pair.
_MLQA_CJK_RE = pcre.compile("[\u4e00-\u9fff]")
_MLQA_ARTICLES_EN_RE = pcre.compile(r"\b(a|an|the)\b")
_MLQA_ARTICLES_ES_RE = pcre.compile(r"\b(un|una|unos|unas|el|la|los|las)\b")
_MLQA_ARTICLES_VI_RE = pcre.compile(r"\b(của|là|cái|chiếc|những)\b")
_MLQA_ARTICLES_DE_RE = pcre.compile(r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b")
_MLQA_ARTICLES_AR_RE = pcre.compile(r"(?:^|\s)ال")


def _mlqa_cache_dir(cache_dir: str | None) -> Path:
    """Implement MLQA cache dir for this module."""
    base_dir = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "evalution" / "datasets"
    target_dir = base_dir / "mlqa"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _download_mlqa_archive(cache_dir: str | None) -> Path:
    """Implement download MLQA archive for this module."""
    archive_path = _mlqa_cache_dir(cache_dir) / "MLQA_V1.zip"
    if archive_path.exists():
        return archive_path
    with urlopen(_MLQA_URL) as response, archive_path.open("wb") as output_file:
        output_file.write(response.read())
    return archive_path


def _flatten_mlqa_payload(payload: dict[str, Any]) -> Dataset:
    # Flatten the official SQuAD-style archive into Evalution's one-row-per-question schema.
    """Flatten MLQA payload. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    rows: list[dict[str, Any]] = []
    for article in payload["data"]:
        title = str(article.get("title", "")).strip()
        for paragraph in article["paragraphs"]:
            context = str(paragraph["context"])
            for qa in paragraph["qas"]:
                answer_texts: list[str] = []
                answer_starts: list[int] = []
                for answer in qa["answers"]:
                    text = str(answer["text"]).strip()
                    if not text:
                        continue
                    answer_texts.append(text)
                    answer_starts.append(int(answer["answer_start"]))
                if not answer_texts:
                    raise ValueError("mlqa requires at least one non-empty answer")
                rows.append(
                    {
                        "id": str(qa["id"]),
                        "title": title,
                        "context": context,
                        "question": str(qa["question"]).strip(),
                        "answers": {
                            "text": answer_texts,
                            "answer_start": answer_starts,
                        },
                    }
                )
    return Dataset.from_list(rows)


def _load_mlqa_dataset(
    dataset_path: str,
    dataset_name: str | None = None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool | None = None,
    context_language: str,
    question_language: str,
) -> Dataset:
    """Load MLQA dataset."""
    effective_stream = False if stream is None else stream
    if effective_stream:
        raise ValueError("mlqa does not support stream=True")
    expected_dataset_name = f"mlqa.{context_language}.{question_language}"
    if dataset_name not in {None, expected_dataset_name}:
        raise ValueError("mlqa dataset_name must match the configured language pair")
    split_directory = _MLQA_SPLIT_DIRECTORY.get(split)
    if split_directory is None:
        raise ValueError(f"unsupported mlqa split: {split!r}")
    archive_path = _download_mlqa_archive(cache_dir)
    member_name = f"MLQA_V1/{split_directory}/{split_directory}-context-{context_language}-question-{question_language}.json"
    with zipfile.ZipFile(archive_path) as archive:
        try:
            with archive.open(member_name) as handle:
                payload = json.load(handle)
        except KeyError as error:
            raise ValueError(f"mlqa archive missing member {member_name!r}") from error
    return _flatten_mlqa_payload(payload)


def _mlqa_prompt(*, context: str, question: str) -> str:
    """Implement MLQA prompt for this module."""
    return f"Context: {context.strip()}\n\nQuestion: {question.strip()}\n\nAnswer:"


def _mlqa_answer_texts(doc: dict[str, Any]) -> list[str]:
    """Implement MLQA answer texts for this module."""
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
    """Implement whitespace tokenize for this module."""
    return text.split()


def _mixed_segmentation(text: str) -> list[str]:
    """Implement mixed segmentation for this module."""
    segments: list[str] = []
    buffered = ""
    for character in text:
        if _MLQA_CJK_RE.search(character) or character in _MLQA_PUNCT:
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
    """Normalize MLQA answer. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    lowered = text.lower()
    stripped_punctuation = "".join(character for character in lowered if character not in _MLQA_PUNCT)
    if language == "en":
        stripped_articles = _MLQA_ARTICLES_EN_RE.sub(" ", stripped_punctuation)
    elif language == "es":
        stripped_articles = _MLQA_ARTICLES_ES_RE.sub(" ", stripped_punctuation)
    elif language == "hi":
        stripped_articles = stripped_punctuation
    elif language == "vi":
        stripped_articles = _MLQA_ARTICLES_VI_RE.sub(" ", stripped_punctuation)
    elif language == "de":
        stripped_articles = _MLQA_ARTICLES_DE_RE.sub(" ", stripped_punctuation)
    elif language == "ar":
        stripped_articles = _MLQA_ARTICLES_AR_RE.sub(" ", stripped_punctuation)
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
    """Implement MLQA exact match for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return float(_normalize_mlqa_answer(prediction, language) == _normalize_mlqa_answer(answer, language))


def _mlqa_f1(prediction: str, answer: str, language: str) -> float:
    """Implement MLQA F1 for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
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
    """Implement best MLQA scores for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
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
    """Implement the MLQA benchmark suite."""
    dataset_path: str = "facebook/mlqa"
    dataset_name: str | None = "mlqa.en.en"
    split: str = "test"
    context_language: str = "en"
    question_language: str = "en"
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Normalize and validate the dataclass configuration after initialization."""
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
        """Return the dataset loader bound to this suite."""
        return partial(
            _load_mlqa_dataset,
            context_language=self.context_language,
            question_language=self.question_language,
        )

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return f"mlqa_{self.context_language}_{self.question_language}"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_mlqa_exact_match_f1",
            "primary_metric": "f1",
            "context_language": self.context_language,
            "question_language": self.question_language,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
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
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
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
    """Implement MLQA for this module."""
    kwargs.setdefault("dataset_name", f"mlqa.{context_language}.{question_language}")
    return MLQA(
        context_language=context_language,
        question_language=question_language,
        **kwargs,
    )


def _make_mlqa_factory(context_language: str, question_language: str) -> Any:
    # Emit one stable factory per context/question language pair for registry discovery.
    """Make MLQA factory."""
    def factory(**kwargs: Any) -> MLQA:
        """Implement factory for this module."""
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
