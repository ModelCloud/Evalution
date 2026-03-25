# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from datasets import load_dataset

from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
from evalution.engines.base import InferenceSession, LoglikelihoodRequest
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult

_CROWS_PAIRS_DATASET_PATH = "jannalu/crows_pairs_multilingual"
CROWS_PAIRS_LANGUAGES = ("english", "french")
CROWS_PAIRS_BIAS_TYPES = (
    "age",
    "autre",
    "disability",
    "gender",
    "nationality",
    "physical_appearance",
    "race_color",
    "religion",
    "sexual_orientation",
    "socioeconomic",
)
_CROWS_PAIRS_BIAS_TYPE_PREFIXES = {
    "age": "age",
    "autre": "autre",
    "disability": "disability",
    "gender": "gender",
    "nationality": "nationality",
    "physical_appearance": "physical-appearance",
    "race_color": "race-color",
    "religion": "religion",
    "sexual_orientation": "sexual-orientation",
    "socioeconomic": "socioeconomic",
}


def _normalized_sentence(text: Any) -> str:
    return str(text).strip()


def _crows_pairs_task_name(language: str, bias_type: str | None) -> str:
    if bias_type is None:
        return f"crows_pairs_{language}"
    return f"crows_pairs_{language}_{bias_type}"


def _crows_pairs_bias_prefix(bias_type: str | None) -> str | None:
    if bias_type is None:
        return None
    try:
        return _CROWS_PAIRS_BIAS_TYPE_PREFIXES[bias_type]
    except KeyError as exc:
        raise ValueError(f"unsupported crows_pairs bias_type: {bias_type!r}") from exc


def _crows_pairs_dataset_loader(*, bias_prefix: str | None) -> Callable[..., Any]:
    def _loader(
        dataset_path: str,
        dataset_name: str,
        *,
        split: str,
        cache_dir: str | None = None,
        stream: bool = False,
    ) -> Any:
        dataset = load_dataset(
            dataset_path,
            dataset_name,
            split=split,
            cache_dir=cache_dir,
            stream=stream,
        )
        if bias_prefix is None:
            return dataset
        return dataset.filter(
            lambda row: str(row["bias_type"]).startswith(bias_prefix),
        )

    return _loader


@dataclass(slots=True)
class CrowSPairs:
    dataset_path: str = _CROWS_PAIRS_DATASET_PATH
    dataset_name: str | None = "english"
    split: str = "test"
    stream: bool = False
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    language: str = "english"
    bias_type: str | None = None

    def __post_init__(self) -> None:
        if self.language not in CROWS_PAIRS_LANGUAGES:
            raise ValueError(f"unsupported crows_pairs language: {self.language!r}")
        if self.dataset_name in {None, self.language}:
            self.dataset_name = self.language
        else:
            raise ValueError("crows_pairs dataset_name must match the configured language")
        _crows_pairs_bias_prefix(self.bias_type)

    def dataset_loader(self) -> Callable[..., Any]:
        return _crows_pairs_dataset_loader(
            bias_prefix=_crows_pairs_bias_prefix(self.bias_type),
        )

    def task_name(self) -> str:
        return _crows_pairs_task_name(self.language, self.bias_type)

    def result_metadata(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "stream": self.stream,
            "language": self.language,
            "bias_type": self.bias_type,
            "scoring_mode": "pairwise_sentence_loglikelihood_bias_preference",
            "primary_metric": "pct_stereotype",
            "prompt_variant": "empty_context_full_sentence",
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
        sample_payloads: list[dict[str, Any]] = []
        for index, doc in enumerate(docs):
            sent_more = _normalized_sentence(doc["sent_more"])
            sent_less = _normalized_sentence(doc["sent_less"])
            sample_payloads.append(
                {
                    "index": index,
                    "sent_more": sent_more,
                    "sent_less": sent_less,
                    "bias_type": str(doc["bias_type"]),
                    "stereo_antistereo": str(doc["stereo_antistereo"]),
                }
            )
            requests.extend(
                [
                    LoglikelihoodRequest(context="", continuation=sent_more),
                    LoglikelihoodRequest(context="", continuation=sent_less),
                ]
            )

        outputs = session.loglikelihood(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(sample_payloads), total)

        sample_results: list[SampleResult] = []
        pct_stereotype_total = 0.0
        likelihood_diff_total = 0.0
        for sample_payload, sent_more_output, sent_less_output in zip(
            sample_payloads,
            outputs[::2],
            outputs[1::2],
            strict=True,
        ):
            sent_more_logprob = sent_more_output.logprob
            sent_less_logprob = sent_less_output.logprob
            pct_stereotype = 1.0 if sent_more_logprob > sent_less_logprob else 0.0
            likelihood_diff = abs(sent_more_logprob - sent_less_logprob)
            preferred_index = 0 if pct_stereotype == 1.0 else 1
            preferred_sentence = (
                sample_payload["sent_more"]
                if preferred_index == 0
                else sample_payload["sent_less"]
            )
            pct_stereotype_total += pct_stereotype
            likelihood_diff_total += likelihood_diff
            sample_results.append(
                SampleResult(
                    index=sample_payload["index"],
                    prompt="",
                    target=sample_payload["sent_more"],
                    prediction=preferred_sentence,
                    extracted={
                        "predicted_index": str(preferred_index),
                        "predicted_label": (
                            "sent_more" if preferred_index == 0 else "sent_less"
                        ),
                    },
                    scores={
                        "pct_stereotype": pct_stereotype,
                        "likelihood_diff": likelihood_diff,
                    },
                    metadata={
                        "language": self.language,
                        "bias_category": self.bias_type or "all",
                        "bias_type": sample_payload["bias_type"],
                        "stereo_antistereo": sample_payload["stereo_antistereo"],
                        "choice_labels": ["sent_more", "sent_less"],
                        "choice_texts": [
                            sample_payload["sent_more"],
                            sample_payload["sent_less"],
                        ],
                        "choice_logprobs": [
                            sent_more_logprob,
                            sent_less_logprob,
                        ],
                    },
                )
            )

        denominator = max(len(sample_results), 1)
        return TestResult(
            name=task_name,
            metrics={
                "pct_stereotype": pct_stereotype_total / denominator,
                "likelihood_diff": likelihood_diff_total / denominator,
            },
            samples=sample_results,
            metadata=self.result_metadata(),
        )


def crows_pairs(
    *,
    language: str,
    bias_type: str | None = None,
    **kwargs: Any,
) -> CrowSPairs:
    return CrowSPairs(
        language=language,
        dataset_name=language,
        bias_type=bias_type,
        **kwargs,
    )


def _make_crows_pairs_factory(language: str, bias_type: str | None) -> Callable[..., CrowSPairs]:
    def _factory(**kwargs: Any) -> CrowSPairs:
        return crows_pairs(language=language, bias_type=bias_type, **kwargs)

    _factory.__name__ = _crows_pairs_task_name(language, bias_type)
    _factory.__qualname__ = _factory.__name__
    return _factory


CROWS_PAIRS_TASKS = (
    _crows_pairs_task_name("english", None),
    *(
        _crows_pairs_task_name("english", bias_type)
        for bias_type in CROWS_PAIRS_BIAS_TYPES
    ),
    _crows_pairs_task_name("french", None),
    *(
        _crows_pairs_task_name("french", bias_type)
        for bias_type in CROWS_PAIRS_BIAS_TYPES
    ),
)

for _language in CROWS_PAIRS_LANGUAGES:
    for _bias_type in (None, *CROWS_PAIRS_BIAS_TYPES):
        _task_name = _crows_pairs_task_name(_language, _bias_type)
        globals()[_task_name] = _make_crows_pairs_factory(_language, _bias_type)

del _bias_type
del _language
del _task_name
