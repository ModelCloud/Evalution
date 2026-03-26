# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from datasets import Dataset

from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
from evalution.engines.base import InferenceSession, LoglikelihoodRequest
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult
from evalution.scorers.multiple_choice import (
    ChoiceScore,
    build_choice_score,
    choice_logprobs,
    choice_logprobs_norm,
    multiple_choice_outcome,
)

_WSC273_XML_URL = "https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WSCollection.xml"
_WSC273_XML_NAME = "WSCollection.xml"
_UPPER_PRONOUNS = [
    "A",
    "An",
    "The",
    "She",
    "He",
    "It",
    "They",
    "My",
    "His",
    "Her",
    "Their",
]


def _cleanup_whitespace(text: str) -> str:
    return " ".join(text.split())


def _wsc273_cache_path(cache_dir: str | None) -> Path:
    if cache_dir is not None:
        base_dir = Path(cache_dir)
    else:
        base_dir = Path.home() / ".cache" / "evalution" / "downloads"
    return base_dir / _WSC273_XML_NAME


def _ensure_wsc273_xml(*, cache_dir: str | None) -> Path:
    xml_path = _wsc273_cache_path(cache_dir)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    if not xml_path.exists():
        urlretrieve(_WSC273_XML_URL, xml_path)
    return xml_path


def _normalize_wsc273_option(doc: dict[str, Any], option: str) -> str:
    normalized = option
    if str(doc["pronoun"]).lower() in {"my", "his", "her", "our", "their"}:
        normalized += "'s"
    pronoun = normalized.split()[0]
    start_of_sentence = int(doc["pronoun_loc"]) >= 2 and str(doc["text"])[int(doc["pronoun_loc"]) - 2] == "."
    if not start_of_sentence and pronoun in _UPPER_PRONOUNS:
        return normalized.replace(pronoun, pronoun.lower(), 1)
    return normalized


def _load_wsc273_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = True,
) -> Dataset:
    del stream
    if dataset_path != "winograd_wsc":
        raise ValueError(f"unsupported WSC dataset path: {dataset_path!r}")
    if dataset_name != "wsc273":
        raise ValueError(f"unsupported WSC dataset name: {dataset_name!r}")
    if split != "test":
        raise ValueError(f"unsupported WSC split: {split!r}")

    xml_path = _ensure_wsc273_xml(cache_dir=cache_dir)
    tree = ET.parse(xml_path)
    rows: list[dict[str, Any]] = []
    for index, schema in enumerate(tree.getroot()):
        if index >= 273:
            break
        text_root = schema.find("text")
        if text_root is None:
            continue
        text_left = _cleanup_whitespace(text_root.findtext("txt1", ""))
        text_right = _cleanup_whitespace(text_root.findtext("txt2", ""))
        pronoun = _cleanup_whitespace(text_root.findtext("pron", ""))
        text = " ".join(part for part in [text_left, pronoun, text_right] if part).strip().replace("  ", " ")
        options = [
            _cleanup_whitespace(option.text or "")
            for option in schema.find("answers").findall("answer")
        ]
        label_text = _cleanup_whitespace(schema.findtext("correctAnswer", ""))
        row = {
            "text": text,
            "pronoun": pronoun,
            "pronoun_loc": len(text_left) + 1 if text_left else 0,
            "options": options,
            "label": int("B" in label_text),
            "source": _cleanup_whitespace(schema.findtext("source", "")),
        }
        row["options"] = [_normalize_wsc273_option(row, option) for option in row["options"]]
        rows.append(row)
    return Dataset.from_list(rows)


@dataclass(slots=True)
class WSC273:
    dataset_path: str = "winograd_wsc"
    dataset_name: str | None = "wsc273"
    split: str = "test"
    stream: bool = True
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    def dataset_loader(self) -> Any:
        return _load_wsc273_dataset

    def task_name(self) -> str:
        return "wsc273"

    def result_metadata(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "stream": self.stream,
            "scoring_mode": "multiple_choice_loglikelihood",
            "prompt_variant": "partial_evaluation",
        }

    def evaluate(self, session: InferenceSession) -> TestResult:
        task_name = self.task_name()
        logger = get_logger()
        loaded_docs, _dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            stream=self.stream,
        )

        docs = limit_docs(loaded_docs, self.max_rows)
        if not isinstance(docs, list):
            docs = list(docs)

        total = doc_count(
            docs,
            loaded_docs=loaded_docs,
            max_rows=self.max_rows,
            split=self.split,
        )
        logger.info("%s: evaluating %d sample(s)", task_name, total)

        requests: list[LoglikelihoodRequest] = []
        request_to_choice: list[tuple[int, int]] = []
        sample_payloads: list[dict[str, Any]] = []
        for index, doc in enumerate(docs):
            text = str(doc["text"])
            pronoun = str(doc["pronoun"])
            pronoun_loc = int(doc["pronoun_loc"])
            suffix_start = pronoun_loc + len(pronoun)
            prefix = text[:pronoun_loc]
            suffix = text[suffix_start:]
            option_contexts = [f"{prefix}{option}" for option in doc["options"]]
            completed_choices = [f"{option_context}{suffix}" for option_context in option_contexts]
            sample_payloads.append(
                {
                    "index": index,
                    "prompt": text,
                    "target_suffix": suffix,
                    "gold_index": int(doc["label"]),
                    "completed_choices": completed_choices,
                    "metadata": {
                        "text": text,
                        "pronoun": pronoun,
                        "pronoun_loc": pronoun_loc,
                        "source": str(doc["source"]),
                        "choice_labels": ["A", "B"],
                        "choice_texts": list(doc["options"]),
                        "target_suffix": suffix,
                    },
                }
            )
            for choice_index, option_context in enumerate(option_contexts):
                requests.append(
                    LoglikelihoodRequest(
                        context=option_context,
                        continuation=suffix,
                    )
                )
                request_to_choice.append((index, choice_index))

        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(sample_payloads), total)

        sample_choice_scores: dict[int, list[ChoiceScore]] = defaultdict(list)
        for (sample_index, choice_index), output in zip(request_to_choice, outputs, strict=True):
            sample_choice_scores[sample_index].append(
                build_choice_score(
                    choice_index=choice_index,
                    logprob=output.logprob,
                    token_count=output.token_count,
                )
            )

        sample_results: list[SampleResult] = []
        raw_total = 0.0
        norm_total = 0.0
        for sample_payload in sample_payloads:
            choice_scores = sorted(
                sample_choice_scores[sample_payload["index"]],
                key=lambda item: item.index,
            )
            outcome = multiple_choice_outcome(choice_scores, sample_payload["gold_index"])
            raw_total += outcome.raw_accuracy
            norm_total += outcome.normalized_accuracy
            sample_results.append(
                SampleResult(
                    index=sample_payload["index"],
                    prompt=sample_payload["prompt"],
                    target=sample_payload["completed_choices"][sample_payload["gold_index"]],
                    prediction=sample_payload["completed_choices"][outcome.normalized_best_index],
                    extracted={
                        "gold_index": str(sample_payload["gold_index"]),
                        "predicted_index": str(outcome.raw_best_index),
                        "predicted_index_norm": str(outcome.normalized_best_index),
                    },
                    scores={
                        "acc,ll": outcome.raw_accuracy,
                        "acc,ll_avg": outcome.normalized_accuracy,
                    },
                    metadata={
                        **sample_payload["metadata"],
                        "choice_logprobs": choice_logprobs(choice_scores),
                        "choice_logprobs_norm": choice_logprobs_norm(choice_scores),
                    },
                )
            )

        denominator = max(len(sample_results), 1)
        return TestResult(
            name=task_name,
            metrics={
                "acc,ll": raw_total / denominator,
                "acc,ll_avg": norm_total / denominator,
            },
            samples=sample_results,
            metadata=self.result_metadata(),
        )


def wsc273(**kwargs: Any) -> WSC273:
    return WSC273(**kwargs)
