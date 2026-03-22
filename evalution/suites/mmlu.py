# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.engines.base import InferenceSession, LoglikelihoodRequest
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult
from evalution.suites.base import TestSuite
from evalution.suites.data import doc_count, limit_docs, load_suite_dataset

_MMLU_LABELS = ["A", "B", "C", "D"]


def _format_subject_title(subject: str) -> str:
    # Convert the dataset subject key into the human-readable title used in the benchmark prompt preamble.
    return " ".join(subject.split("_"))


def _format_mmlu_question(doc: dict[str, Any], *, include_answer: bool) -> str:
    # Render one MMLU question block with the canonical lettered options and either an empty or filled answer line.
    lines = [doc["question"].strip()]
    for label, choice in zip(_MMLU_LABELS, doc["choices"], strict=True):
        lines.append(f"{label}. {choice}")
    answer_text = _MMLU_LABELS[int(doc["answer"])] if include_answer else ""
    lines.append(f"Answer: {answer_text}".rstrip())
    return "\n".join(lines)


def _fewshot_prompt(subject: str, fewshot_docs: list[dict[str, Any]]) -> str:
    # Build the per-subject prompt prefix with the upstream description line and any selected dev examples.
    subject_title = _format_subject_title(subject)
    sections = [f"The following are multiple choice questions (with answers) about {subject_title}."]
    if fewshot_docs:
        sections.extend(_format_mmlu_question(doc, include_answer=True) for doc in fewshot_docs)
    return "\n\n".join(sections) + "\n\n"


@dataclass(slots=True)
class MMLU(TestSuite):
    # Evaluate the original MMLU multiple-choice setup with subject-aware few-shot examples from the dev split.
    dataset_path: str = "cais/mmlu"
    subject: str = "all"
    split: str = "validation"
    fewshot_split: str = "dev"
    num_fewshot: int = 5
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    streaming: bool = False

    # Return the dataset config name to load for the current subject selection.
    def dataset_name(self) -> str:
        return self.subject

    # Use the Hugging Face datasets loader for the maintained CAIS MMLU mirror.
    def dataset_loader(self) -> Any:
        return load_dataset

    # Return the stable suite name used by logs, YAML specs, and result payloads.
    def task_name(self) -> str:
        return "mmlu"

    # Report suite-level metadata that is stable across all evaluated rows.
    def result_metadata(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name(),
            "split": self.split,
            "fewshot_split": self.fewshot_split,
            "num_fewshot": self.num_fewshot,
            "streaming": self.streaming,
            "scoring_mode": "multiple_choice_loglikelihood",
        }

    # Execute MMLU with subject-matched few-shot prompts and label-only answer scoring.
    def evaluate(self, session: InferenceSession) -> TestResult:
        task_name = self.task_name()
        logger = get_logger()
        loaded_docs, _dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name(),
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
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

        fewshot_loaded_docs, _fewshot_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name(),
            split=self.fewshot_split,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
        )
        fewshot_docs = list(fewshot_loaded_docs)
        fewshot_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for doc in fewshot_docs:
            fewshot_by_subject[str(doc["subject"])].append(doc)

        prompts: list[str] = []
        requests: list[LoglikelihoodRequest] = []
        sample_docs: list[dict[str, Any]] = []
        for doc in docs:
            subject = str(doc["subject"])
            prompt = _fewshot_prompt(subject, fewshot_by_subject[subject][: self.num_fewshot])
            prompt += _format_mmlu_question(doc, include_answer=False)
            prompts.append(prompt)
            sample_docs.append(doc)
            for label in _MMLU_LABELS:
                requests.append(LoglikelihoodRequest(context=prompt, continuation=f" {label}"))

        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(sample_docs), total)

        sample_results: list[SampleResult] = []
        raw_total = 0.0
        norm_total = 0.0
        for index, doc in enumerate(sample_docs):
            start = index * len(_MMLU_LABELS)
            choice_outputs = outputs[start : start + len(_MMLU_LABELS)]
            choice_logprobs = [output.logprob for output in choice_outputs]
            choice_logprobs_norm = [
                output.logprob / max(output.token_count, 1)
                for output in choice_outputs
            ]
            raw_best = max(range(len(choice_logprobs)), key=choice_logprobs.__getitem__)
            norm_best = max(range(len(choice_logprobs_norm)), key=choice_logprobs_norm.__getitem__)
            gold_index = int(doc["answer"])
            raw_score = 1.0 if raw_best == gold_index else 0.0
            norm_score = 1.0 if norm_best == gold_index else 0.0
            raw_total += raw_score
            norm_total += norm_score
            sample_results.append(
                SampleResult(
                    index=index,
                    prompt=prompts[index],
                    target=_MMLU_LABELS[gold_index],
                    prediction=_MMLU_LABELS[norm_best],
                    extracted={
                        "gold_index": str(gold_index),
                        "predicted_index": str(raw_best),
                        "predicted_index_norm": str(norm_best),
                    },
                    scores={
                        "accuracy,loglikelihood": raw_score,
                        "accuracy,loglikelihood_norm": norm_score,
                    },
                    metadata={
                        "subject": str(doc["subject"]),
                        "choice_texts": list(doc["choices"]),
                        "choice_logprobs": choice_logprobs,
                        "choice_logprobs_norm": choice_logprobs_norm,
                    },
                )
            )

        denominator = max(len(sample_results), 1)
        metrics = {
            "accuracy,loglikelihood": raw_total / denominator,
            "accuracy,loglikelihood_norm": norm_total / denominator,
        }
        logger.info("%s: metrics=%s", task_name, metrics)
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=sample_results,
            metadata=self.result_metadata(),
        )


# Mirror the public suite factory style used by the rest of the package.
def mmlu(**kwargs: Any) -> MMLU:
    return MMLU(**kwargs)
