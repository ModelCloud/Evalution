# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.base import TestSuite
from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
from evalution.engines.base import InferenceSession, LoglikelihoodRequest
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult


def _webqs_prompt(question: str) -> str:
    """Implement webqs prompt for this module."""
    return f"Question: {question.strip()}\nAnswer:"


def _remove_prefix_answers(answers: Any) -> list[str]:
    """Implement remove prefix answers for this module."""
    normalized = sorted(
        str(answer).strip()
        for answer in (answers if isinstance(answers, (list, tuple)) else [answers])
        if str(answer).strip()
    )
    if not normalized:
        raise ValueError("webqs requires at least one non-empty accepted answer")

    retained = [normalized[0]]
    for answer in normalized[1:]:
        if not answer.startswith(retained[-1]):
            retained.append(answer)
    return retained


@dataclass(slots=True)
class WebQS(TestSuite):
    """Define the web qs helper class."""
    # Keep the class-level state explicit for this helper.
    dataset_path: str = "web_questions"
    dataset_name: str | None = None
    split: str = "test"
    stream: bool = True
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "webqs"

    def result_metadata(self) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "stream": self.stream,
            "scoring_mode": "accepted_alias_greedy_exact_match",
            "primary_metric": "em",
        }

    def continuation_for_alias(self, alias: str) -> str:
        """Implement continuation for alias for web qs."""
        return alias if alias[:1].isspace() else f" {alias}"

    def evaluate(self, session: InferenceSession) -> TestResult:
        """Evaluate evaluate. Keep the nested traversal explicit so ordering and metadata stay aligned."""
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

        sample_payloads: list[dict[str, Any]] = []
        requests: list[LoglikelihoodRequest] = []
        request_to_alias: list[tuple[int, int]] = []
        for index, doc in enumerate(docs):
            accepted_answers = _remove_prefix_answers(doc["answers"])
            prompt = _webqs_prompt(str(doc["question"]))
            sample_payloads.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "accepted_answers": accepted_answers,
                    "question": str(doc["question"]),
                    "url": str(doc["url"]),
                }
            )
            for alias_index, alias in enumerate(accepted_answers):
                requests.append(
                    LoglikelihoodRequest(
                        context=prompt,
                        continuation=self.continuation_for_alias(alias),
                    )
                )
                request_to_alias.append((index, alias_index))

        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(sample_payloads), total)

        sample_outputs: list[list[Any]] = [[] for _ in sample_payloads]
        for (sample_index, _alias_index), output in zip(request_to_alias, outputs, strict=True):
            sample_outputs[sample_index].append(output)

        sample_results: list[SampleResult] = []
        exact_match_total = 0.0
        for sample_payload, alias_outputs in zip(sample_payloads, sample_outputs, strict=True):
            accepted_answers = sample_payload["accepted_answers"]
            greedy_indices = [
                alias_index
                for alias_index, output in enumerate(alias_outputs)
                if output.is_greedy
            ]
            greedy_alias_index = greedy_indices[0] if greedy_indices else None
            highest_logprob_alias_index = max(
                range(len(alias_outputs)),
                key=lambda alias_index: alias_outputs[alias_index].logprob,
            )
            exact_match = 1.0 if greedy_alias_index is not None else 0.0
            exact_match_total += exact_match
            predicted_alias_index = (
                greedy_alias_index
                if greedy_alias_index is not None
                else highest_logprob_alias_index
            )
            sample_results.append(
                SampleResult(
                    index=sample_payload["index"],
                    prompt=sample_payload["prompt"],
                    target=accepted_answers[0],
                    prediction=accepted_answers[predicted_alias_index],
                    extracted={
                        "greedy_alias_index": (
                            str(greedy_alias_index)
                            if greedy_alias_index is not None
                            else "[none]"
                        ),
                        "highest_logprob_alias_index": str(highest_logprob_alias_index),
                    },
                    scores={"em": exact_match},
                    metadata={
                        "question": sample_payload["question"],
                        "url": sample_payload["url"],
                        "accepted_answers": accepted_answers,
                        "choice_texts": accepted_answers,
                        "choice_logprobs": [
                            output.logprob for output in alias_outputs
                        ],
                        "choice_greedy": [
                            output.is_greedy for output in alias_outputs
                        ],
                        "greedy_alias_indices": greedy_indices,
                    },
                )
            )

        denominator = max(len(sample_results), 1)
        return TestResult(
            name=task_name,
            metrics={"em": exact_match_total / denominator},
            samples=sample_results,
            metadata=self.result_metadata(),
        )


def webqs(**kwargs: Any) -> WebQS:
    """Implement webqs for this module."""
    return WebQS(**kwargs)
