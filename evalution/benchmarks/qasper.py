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
import re
import string
import tarfile
from typing import Any
from urllib.request import urlopen

from datasets import Dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.benchmarks.multiple_choice import BaseMultipleChoiceSuite, MultipleChoiceSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.classification import f1_for_label

# Preserve the two public QASPER evaluation modes because bool and freeform rows use different scorers.
QASPER_VARIANTS = ("bool", "freeform")
QASPER_TASKS = ("qasper_bool", "qasper_freeform")
_STOP_STRINGS = ("\n",)
_QASPER_URL_TRAIN_DEV = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
_QASPER_URL_TEST = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
_QASPER_DATA_FILES = {
    "train": "qasper-train-v0.3.json",
    "validation": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}


def _qasper_cache_dir(cache_dir: str | None) -> Path:
    base_dir = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "evalution" / "datasets"
    target_dir = base_dir / "qasper"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _download_qasper_archive(split: str, cache_dir: str | None) -> Path:
    archive_url = _QASPER_URL_TEST if split == "test" else _QASPER_URL_TRAIN_DEV
    archive_path = _qasper_cache_dir(cache_dir) / archive_url.rsplit("/", 1)[-1]
    if archive_path.exists():
        return archive_path
    with urlopen(archive_url) as response, archive_path.open("wb") as output_file:
        output_file.write(response.read())
    return archive_path


def _load_qasper_source_dataset(
    *,
    split: str,
    cache_dir: str | None = None,
) -> Dataset:
    data_file = _QASPER_DATA_FILES.get(split)
    if data_file is None:
        raise ValueError(f"unsupported qasper split: {split!r}")
    archive_path = _download_qasper_archive(split, cache_dir)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        member = next((item for item in archive.getmembers() if item.name.endswith(data_file)), None)
        if member is None:
            raise ValueError(f"qasper archive missing data file {data_file!r}")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"qasper archive member {data_file!r} could not be read")
        raw_dataset = json.load(handle)
    rows = []
    for paper_id, paper in raw_dataset.items():
        row = dict(paper)
        row["id"] = paper_id
        rows.append(row)
    return Dataset.from_list(rows)


def _qasper_prompt(*, title: str, abstract: str, question: str) -> str:
    return f"TITLE: {title}\nABSTRACT: {abstract}\n\nQ: {question}\n\nA:"


def _normalize_qasper_answer(text: str) -> str:
    lowered = text.lower()
    stripped_punctuation = "".join(character for character in lowered if character not in set(string.punctuation))
    stripped_articles = re.sub(r"\b(a|an|the)\b", " ", stripped_punctuation)
    return " ".join(stripped_articles.split())


def _qasper_abstractive_f1(prediction: str, answer: str) -> float:
    prediction_tokens = _normalize_qasper_answer(prediction).split()
    answer_tokens = _normalize_qasper_answer(answer).split()
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(prediction_tokens), 1)
    recall = overlap / max(len(answer_tokens), 1)
    return (2 * precision * recall) / (precision + recall)


def _categorize_qasper_answer(answer_blob: dict[str, Any]) -> tuple[str, str]:
    if answer_blob["unanswerable"]:
        return "unanswerable", "unanswerable"
    if answer_blob["yes_no"]:
        return "yes", "bool"
    if answer_blob["free_form_answer"]:
        return str(answer_blob["free_form_answer"]).strip(), "free form answer"
    if answer_blob["extractive_spans"]:
        spans = [str(span).strip() for span in answer_blob["extractive_spans"] if str(span).strip()]
        return ", ".join(spans), "extractive_spans"
    if answer_blob["yes_no"] is False:
        return "no", "bool"
    raise ValueError(f"unsupported qasper answer blob: {answer_blob!r}")


def _iter_qasper_annotations(doc: dict[str, Any]) -> Any:
    # Accept both the Hub-native list-of-question rows and the older script-style dict-of-columns rows.
    qas = doc["qas"]
    if isinstance(qas, list):
        for qa in qas:
            question = str(qa["question"]).strip()
            for annotation in qa.get("answers", []):
                answer_blob = annotation.get("answer", annotation)
                if not isinstance(answer_blob, dict):
                    raise TypeError(f"unsupported qasper answer annotation: {annotation!r}")
                yield question, answer_blob
        return
    if isinstance(qas, dict):
        for question, answer_list in zip(qas["question"], qas["answers"], strict=True):
            for annotation in answer_list["answer"]:
                if not isinstance(annotation, dict):
                    raise TypeError(f"unsupported qasper answer annotation: {annotation!r}")
                yield str(question).strip(), annotation
        return
    raise TypeError(f"unsupported qasper qas payload: {type(qas).__name__}")


def _flatten_qasper_dataset(
    dataset: list[dict[str, Any]] | Dataset,
    *,
    answer_type: str,
) -> Dataset:
    # Expand each paper/question/answer triple into flat benchmark rows for the shared evaluation pipeline.
    rows: list[dict[str, Any]] = []
    for doc in dataset:
        title = str(doc["title"]).strip()
        abstract = str(doc["abstract"]).strip()
        for question, answer_blob in _iter_qasper_annotations(doc):
            answer, parsed_answer_type = _categorize_qasper_answer(answer_blob)
            if parsed_answer_type != answer_type:
                continue
            rows.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "question": question,
                    "answer": answer,
                    "answer_type": parsed_answer_type,
                }
            )
    return Dataset.from_list(rows)


def _load_qasper_dataset(
    dataset_path: str,
    dataset_name: str | None = None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
    answer_type: str,
) -> Dataset:
    if stream:
        raise ValueError("qasper does not support stream=True")
    if dataset_name is not None:
        raise ValueError("qasper does not use dataset_name")
    dataset = _load_qasper_source_dataset(
        split=split,
        cache_dir=cache_dir,
    )
    return _flatten_qasper_dataset(dataset, answer_type=answer_type)


@dataclass(slots=True)
class QASPERBool(BaseMultipleChoiceSuite):
    # Score QASPER yes/no rows as a strict two-choice loglikelihood benchmark.
    dataset_path: str = "allenai/qasper"
    dataset_name: str | None = None
    split: str = "validation"
    stream: bool = False

    def dataset_loader(self) -> Any:
        return partial(_load_qasper_dataset, answer_type="bool")

    def task_name(self) -> str:
        return "qasper_bool"

    def result_metadata(self) -> dict[str, Any]:
        metadata = super().result_metadata()
        metadata["variant"] = "bool"
        return metadata

    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        answer = str(doc["answer"]).strip().lower()
        if answer not in {"yes", "no"}:
            raise ValueError(f"unexpected qasper bool answer: {answer!r}")
        return MultipleChoiceSample(
            index=index,
            prompt=_qasper_prompt(
                title=str(doc["title"]).strip(),
                abstract=str(doc["abstract"]).strip(),
                question=str(doc["question"]).strip(),
            ),
            choices=["no", "yes"],
            gold_index=1 if answer == "yes" else 0,
            metadata={
                "title": str(doc["title"]).strip(),
                "abstract": str(doc["abstract"]).strip(),
                "question": str(doc["question"]).strip(),
                "answer_type": "bool",
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
            "f1,ll_boolean": f1_for_label(gold_labels, raw_predictions, label=1),
            "f1,ll_avg_boolean": f1_for_label(gold_labels, normalized_predictions, label=1),
        }


@dataclass(slots=True)
class QASPERFreeform(BaseTestSuite):
    # Score QASPER abstractive rows with normalized token-overlap F1 against the flattened reference answer.
    dataset_path: str = "allenai/qasper"
    dataset_name: str | None = None
    split: str = "validation"
    stream: bool = False
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        return partial(_load_qasper_dataset, answer_type="free form answer")

    def task_name(self) -> str:
        return "qasper_freeform"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "variant": "freeform",
            "scoring_mode": "generated_qasper_abstractive_f1",
            "primary_metric": "f1",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        for index, doc in enumerate(docs):
            answer = str(doc["answer"]).strip()
            yield PreparedSample(
                index=index,
                doc=doc,
                target=answer,
                request=GenerationRequest(
                    prompt=_qasper_prompt(
                        title=str(doc["title"]).strip(),
                        abstract=str(doc["abstract"]).strip(),
                        question=str(doc["question"]).strip(),
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
        answer = str(prepared_sample.doc["answer"]).strip()
        f1_score = _qasper_abstractive_f1(output.text, answer)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-normalized": _normalize_qasper_answer(output.text),
                "answer-normalized": _normalize_qasper_answer(answer),
            },
            scores={"f1": f1_score},
            metadata={
                "title": str(prepared_sample.doc["title"]).strip(),
                "abstract": str(prepared_sample.doc["abstract"]).strip(),
                "question": str(prepared_sample.doc["question"]).strip(),
                "answer_type": "free form answer",
            },
        )


def qasper(*, variant: str = "bool", **kwargs: Any) -> QASPERBool | QASPERFreeform:
    # Keep one generic constructor for YAML and CLI while still exposing the concrete variants directly.
    if variant == "bool":
        return QASPERBool(**kwargs)
    if variant == "freeform":
        return QASPERFreeform(**kwargs)
    raise ValueError(f"unsupported qasper variant: {variant!r}")


def qasper_bool(**kwargs: Any) -> QASPERBool:
    return QASPERBool(**kwargs)


def qasper_freeform(**kwargs: Any) -> QASPERFreeform:
    return QASPERFreeform(**kwargs)
