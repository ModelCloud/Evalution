# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from evalution.engines.base import InferenceSession, LoglikelihoodRequest
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult
from evalution.scorers.multiple_choice import (
    ChoiceScore,
    build_choice_score,
    choice_labels,
    choice_logprobs,
    choice_logprobs_norm,
    label_permutation_metric_name,
    label_permutation_outcome,
    label_permutations_for_mode,
    multiple_choice_outcome,
    normalize_label_permutation_fraction,
)
from evalution.benchmarks.base import TestSuite
from evalution.benchmarks.data import (
    apply_order,
    doc_count,
    limit_docs,
    load_suite_dataset,
    normalize_order,
)


@dataclass(slots=True)
class MultipleChoiceSample:
    # Represent one multiple-choice question after prompt formatting and choice extraction.
    index: int
    prompt: str
    choices: list[str]
    gold_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BaseMultipleChoiceSuite(TestSuite, ABC):
    # Dataset location and execution options shared by multiple-choice benchmark families.
    dataset_path: str = ""
    dataset_name: str | None = None
    # Concrete benchmarks must set their own canonical split. Many Hugging Face tasks expose
    # validation-only public evaluation rows or use nonstandard split names, so a blanket test
    # default here would silently change benchmark semantics.
    split: str = "validation"
    order: str = "native"
    stream: bool = False
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    # Optional extra label-only scorer that averages over a subset of label permutations to reduce
    # fixed-label priors without replacing the benchmark-native score. Use any float in [0.0, 1.0].
    label_permutations: float = 0.0

    # Return the callable used to fetch the underlying dataset rows.
    @abstractmethod
    def dataset_loader(self) -> Any:
        raise NotImplementedError

    # Return the stable result name for the concrete suite instance.
    @abstractmethod
    def task_name(self) -> str:
        raise NotImplementedError

    # Convert one dataset row into the normalized prompt/choices form scored by the helper.
    @abstractmethod
    def build_sample(self, doc: dict[str, Any], *, index: int) -> MultipleChoiceSample:
        raise NotImplementedError

    # Control how each choice is appended to the shared prompt during scoring.
    def continuation_for_choice(self, choice: str) -> str:
        return choice if choice[:1].isspace() else f" {choice}"

    # Control how each option-label continuation is appended when the optional label-permutation scorer is enabled.
    def continuation_for_choice_label(self, label: str) -> str:
        return f" {label}"

    # Let suites override how prompt/choice payload length is measured for benchmark-level ordering.
    def order_length(self, sample: MultipleChoiceSample) -> int:
        return len(sample.prompt) + sum(len(self.continuation_for_choice(choice)) for choice in sample.choices)

    # Build the prompt used by the optional label-permutation scorer. The default keeps the original prompt
    # stem, lists labeled options, and asks for just the label. Suites with special semantics can override it.
    def label_prompt(
        self,
        sample: MultipleChoiceSample,
        *,
        choice_order: tuple[int, ...],
        labels: tuple[str, ...],
    ) -> str:
        prompt_stem = sample.prompt.rstrip()
        if prompt_stem.endswith("Answer:"):
            prompt_stem = prompt_stem[: -len("Answer:")].rstrip()
        lines = [prompt_stem, "Options:"]
        for label, choice_index in zip(labels, choice_order, strict=True):
            lines.append(f"{label}. {sample.choices[choice_index]}")
        lines.append("Answer:")
        return "\n".join(lines)

    # Report suite-level metadata that is stable across all samples in the run.
    def result_metadata(self) -> dict[str, Any]:
        metadata = {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "order": normalize_order(self.order),
            "stream": self.stream,
            "scoring_mode": "multiple_choice_loglikelihood",
        }
        metadata.update(self._label_permutation_metadata())
        return metadata

    # Allow concrete suites to publish extra aggregate metrics without reimplementing the shared scoring loop.
    def extra_metrics(
        self,
        *,
        samples: list[MultipleChoiceSample],
        raw_predictions: list[int],
        normalized_predictions: list[int],
    ) -> dict[str, float]:
        del samples, raw_predictions, normalized_predictions
        return {}

    def _resolved_label_permutations(self) -> float:
        return normalize_label_permutation_fraction(self.label_permutations)

    def _label_permutation_metadata(self) -> dict[str, Any]:
        resolved_mode = self._resolved_label_permutations()
        if resolved_mode == 0.0:
            return {}
        return {
            "extra_scoring_mode": "multiple_choice_label_permutation_average",
            "label_permutations": resolved_mode,
            "label_permutation_metric": label_permutation_metric_name(resolved_mode),
        }

    def _label_permutation_scores(
        self,
        session: InferenceSession,
        *,
        samples: list[MultipleChoiceSample],
    ) -> tuple[str | None, dict[int, Any]]:
        resolved_mode = self._resolved_label_permutations()
        if resolved_mode == 0.0:
            return None, {}

        requests: list[LoglikelihoodRequest] = []
        request_map: list[tuple[int, int, int]] = []
        sample_permutations: dict[int, list[tuple[int, ...]]] = {}
        for sample in samples:
            sample_labels = choice_labels(len(sample.choices))
            permutations = label_permutations_for_mode(len(sample.choices), resolved_mode)
            sample_permutations[sample.index] = permutations
            for permutation_index, permutation in enumerate(permutations):
                prompt = self.label_prompt(
                    sample,
                    choice_order=permutation,
                    labels=sample_labels,
                )
                for label_index, label in enumerate(sample_labels):
                    requests.append(
                        LoglikelihoodRequest(
                            context=prompt,
                            continuation=self.continuation_for_choice_label(label),
                        )
                    )
                    request_map.append((sample.index, permutation_index, label_index))

        # This second scoring pass is intentionally opt-in because every extra permutation multiplies
        # label-token scoring requests by `permutation_count * choice_count`.
        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        sample_scores: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for (sample_index, permutation_index, _label_index), output in zip(
            request_map,
            outputs,
            strict=True,
        ):
            sample_scores[sample_index][permutation_index].append(output.logprob)

        outcomes: dict[int, Any] = {}
        for sample in samples:
            permutations = sample_permutations[sample.index]
            permutation_label_logprobs = [
                sample_scores[sample.index][permutation_index]
                for permutation_index in range(len(permutations))
            ]
            outcomes[sample.index] = label_permutation_outcome(
                permutations=permutations,
                permutation_label_logprobs=permutation_label_logprobs,
                gold_index=sample.gold_index,
            )
        return label_permutation_metric_name(resolved_mode), outcomes

    # Execute the shared dataset loading, flattened choice scoring, and accuracy aggregation flow.
    def evaluate(self, session: InferenceSession) -> TestResult:
        task_name = self.task_name()
        resolved_order = normalize_order(self.order)
        logger = get_logger()
        loaded_docs, _dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=self.stream,
        )

        docs = limit_docs(loaded_docs, self.max_rows)
        if resolved_order != "native" and self.stream:
            raise ValueError("benchmark `stream=True` requires `order='native'`")
        if not isinstance(docs, list) or resolved_order != "native":
            docs = list(docs)

        total = doc_count(
            docs,
            loaded_docs=loaded_docs,
            max_rows=self.max_rows,
            split=self.split,
        )
        logger.info("%s: evaluating %d sample(s)", task_name, total)

        samples = [self.build_sample(doc, index=index) for index, doc in enumerate(docs)]
        samples = apply_order(
            samples,
            order=resolved_order,
            length_key=self.order_length,
        )
        requests: list[LoglikelihoodRequest] = []
        request_to_choice: list[tuple[int, int]] = []
        for sample in samples:
            for choice_index, choice in enumerate(sample.choices):
                requests.append(
                    LoglikelihoodRequest(
                        context=sample.prompt,
                        continuation=self.continuation_for_choice(choice),
                    )
                )
                request_to_choice.append((sample.index, choice_index))

        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(samples), total)

        label_metric_name, label_outcomes = self._label_permutation_scores(session, samples=samples)

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
        label_total = 0.0
        raw_predictions: list[int] = []
        normalized_predictions: list[int] = []
        for sample in samples:
            choice_scores = sorted(sample_choice_scores[sample.index], key=lambda item: item.index)
            outcome = multiple_choice_outcome(choice_scores, sample.gold_index)
            raw_total += outcome.raw_accuracy
            norm_total += outcome.normalized_accuracy
            raw_predictions.append(outcome.raw_best_index)
            normalized_predictions.append(outcome.normalized_best_index)
            sample_scores = {
                "acc,ll": outcome.raw_accuracy,
                "acc,ll_avg": outcome.normalized_accuracy,
            }
            sample_extracted = {
                "gold_index": str(sample.gold_index),
                "predicted_index": str(outcome.raw_best_index),
                "predicted_index_norm": str(outcome.normalized_best_index),
            }
            sample_metadata = {
                **sample.metadata,
                "choice_logprobs": choice_logprobs(choice_scores),
                "choice_logprobs_norm": choice_logprobs_norm(choice_scores),
            }
            if label_metric_name is not None:
                label_outcome = label_outcomes[sample.index]
                label_total += label_outcome.accuracy
                sample_scores[label_metric_name] = label_outcome.accuracy
                sample_extracted[f"predicted_index_{label_metric_name.split(',', maxsplit=1)[1]}"] = str(
                    label_outcome.predicted_index
                )
                sample_metadata[f"choice_logprobs_{label_metric_name.split(',', maxsplit=1)[1]}"] = list(
                    label_outcome.averaged_choice_logprobs
                )
                sample_metadata["label_permutation_count"] = label_outcome.permutation_count
            sample_results.append(
                SampleResult(
                    index=sample.index,
                    prompt=sample.prompt,
                    target=sample.choices[sample.gold_index],
                    prediction=sample.choices[outcome.normalized_best_index],
                    extracted=sample_extracted,
                    scores=sample_scores,
                    metadata=sample_metadata,
                )
            )

        denominator = max(len(sample_results), 1)
        metrics = {
            "acc,ll": raw_total / denominator,
            "acc,ll_avg": norm_total / denominator,
        }
        if label_metric_name is not None:
            metrics[label_metric_name] = label_total / denominator
        metrics.update(
            self.extra_metrics(
                samples=samples,
                raw_predictions=raw_predictions,
                normalized_predictions=normalized_predictions,
            )
        )
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=sample_results,
            metadata=self.result_metadata(),
        )
