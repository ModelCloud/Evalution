# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.benchmarks.subsets import normalize_subset_token
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult
from evalution.scorers.longbench import (
    longbench_classification_score,
    longbench_code_sim_score,
    longbench_count_score,
    longbench_qa_f1_score,
    longbench_qa_f1_zh_score,
    longbench_retrieval_score,
    longbench_retrieval_zh_score,
    longbench_rouge_score,
    longbench_rouge_zh_score,
)

# Keep the public factory names Python-safe while still mapping onto the upstream dataset configs.
LONG_BENCH_TASK_TO_DATASET = {
    "longbench_narrativeqa": "narrativeqa",
    "longbench_qasper": "qasper",
    "longbench_multifieldqa_en": "multifieldqa_en",
    "longbench_multifieldqa_zh": "multifieldqa_zh",
    "longbench_hotpotqa": "hotpotqa",
    "longbench_2wikimqa": "2wikimqa",
    "longbench_musique": "musique",
    "longbench_dureader": "dureader",
    "longbench_gov_report": "gov_report",
    "longbench_qmsum": "qmsum",
    "longbench_multi_news": "multi_news",
    "longbench_vcsum": "vcsum",
    "longbench_trec": "trec",
    "longbench_triviaqa": "triviaqa",
    "longbench_samsum": "samsum",
    "longbench_lsht": "lsht",
    "longbench_passage_count": "passage_count",
    "longbench_passage_retrieval_en": "passage_retrieval_en",
    "longbench_passage_retrieval_zh": "passage_retrieval_zh",
    "longbench_lcc": "lcc",
    "longbench_repobench_p": "repobench-p",
    "longbench_qasper_e": "qasper_e",
    "longbench_multifieldqa_en_e": "multifieldqa_en_e",
    "longbench_hotpotqa_e": "hotpotqa_e",
    "longbench_2wikimqa_e": "2wikimqa_e",
    "longbench_gov_report_e": "gov_report_e",
    "longbench_multi_news_e": "multi_news_e",
    "longbench_trec_e": "trec_e",
    "longbench_triviaqa_e": "triviaqa_e",
    "longbench_samsum_e": "samsum_e",
    "longbench_passage_count_e": "passage_count_e",
    "longbench_passage_retrieval_en_e": "passage_retrieval_en_e",
    "longbench_lcc_e": "lcc_e",
    "longbench_repobench_p_e": "repobench-p_e",
}
LONG_BENCH_TASKS = tuple(LONG_BENCH_TASK_TO_DATASET)
_LONG_BENCH_ALIAS_TO_TASK = {
    normalize_subset_token(alias): task_name
    for task_name, dataset_name in LONG_BENCH_TASK_TO_DATASET.items()
    for alias in (
        task_name,
        task_name.removeprefix("longbench_"),
        dataset_name,
        f"longbench_{dataset_name}",
    )
}
_LONG_BENCH_DEFAULT_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}


@dataclass(frozen=True, slots=True)
class _LongBenchMetricSpec:
    # Keep per-task metric behavior explicit because LongBench mixes QA, summarization, retrieval, and code scoring.
    metric_name: str
    scoring_mode: str
    stop_strings: tuple[str, ...] = ()
    single_line_prediction: bool = False


# Mirror the author task categories, but keep the runtime scorer registry local and dependency-light.
_LONG_BENCH_METRICS = {
    "narrativeqa": _LongBenchMetricSpec("qa_f1_score", "generated_longbench_qa_f1"),
    "qasper": _LongBenchMetricSpec("qa_f1_score", "generated_longbench_qa_f1"),
    "multifieldqa_en": _LongBenchMetricSpec("qa_f1_score", "generated_longbench_qa_f1"),
    "multifieldqa_zh": _LongBenchMetricSpec("qa_f1_zh_score", "generated_longbench_qa_f1_zh"),
    "hotpotqa": _LongBenchMetricSpec("qa_f1_score", "generated_longbench_qa_f1"),
    "2wikimqa": _LongBenchMetricSpec("qa_f1_score", "generated_longbench_qa_f1"),
    "musique": _LongBenchMetricSpec("qa_f1_score", "generated_longbench_qa_f1"),
    "dureader": _LongBenchMetricSpec("rouge_zh_score", "generated_longbench_rouge_zh"),
    "gov_report": _LongBenchMetricSpec("rouge_score", "generated_longbench_rouge"),
    "qmsum": _LongBenchMetricSpec("rouge_score", "generated_longbench_rouge"),
    "multi_news": _LongBenchMetricSpec("rouge_score", "generated_longbench_rouge"),
    "vcsum": _LongBenchMetricSpec("rouge_zh_score", "generated_longbench_rouge_zh"),
    "trec": _LongBenchMetricSpec(
        "classification_score",
        "generated_longbench_classification",
        stop_strings=("\n",),
        single_line_prediction=True,
    ),
    "triviaqa": _LongBenchMetricSpec(
        "qa_f1_score",
        "generated_longbench_qa_f1",
        stop_strings=("\n",),
        single_line_prediction=True,
    ),
    "samsum": _LongBenchMetricSpec(
        "rouge_score",
        "generated_longbench_rouge",
        stop_strings=("\n",),
        single_line_prediction=True,
    ),
    "lsht": _LongBenchMetricSpec(
        "classification_score",
        "generated_longbench_classification",
        stop_strings=("\n",),
        single_line_prediction=True,
    ),
    "passage_count": _LongBenchMetricSpec("count_score", "generated_longbench_count"),
    "passage_retrieval_en": _LongBenchMetricSpec("retrieval_score", "generated_longbench_retrieval"),
    "passage_retrieval_zh": _LongBenchMetricSpec("retrieval_zh_score", "generated_longbench_retrieval_zh"),
    "lcc": _LongBenchMetricSpec("code_sim_score", "generated_longbench_code_sim"),
    "repobench-p": _LongBenchMetricSpec("code_sim_score", "generated_longbench_code_sim"),
}


def _longbench_task_name(value: str) -> str:
    task_name = _LONG_BENCH_ALIAS_TO_TASK.get(normalize_subset_token(value))
    if task_name is None:
        raise ValueError(f"unsupported longbench subset: {value!r}")
    return task_name


def _longbench_task_root(dataset_name: str) -> str:
    return dataset_name[:-2] if dataset_name.endswith("_e") else dataset_name


def _longbench_answers(doc: dict[str, Any]) -> list[str]:
    raw_answers = doc.get("answers", [])
    values = raw_answers if isinstance(raw_answers, list) else [raw_answers]
    deduped: list[str] = []
    for answer in values:
        text = str(answer)
        if text.strip() and text not in deduped:
            deduped.append(text)
    if not deduped:
        raise ValueError("longbench rows must contain at least one non-empty answer")
    return deduped


def _longbench_all_classes(doc: dict[str, Any]) -> list[str]:
    raw_classes = doc.get("all_classes", [])
    if not isinstance(raw_classes, list):
        return []
    return [str(value) for value in raw_classes if str(value)]


def _longbench_prompt(doc: dict[str, Any]) -> str:
    context = str(doc.get("context", ""))
    question = str(doc.get("question", doc.get("input", "")))
    answer_prefix = str(doc.get("answer_prefix", ""))
    prompt = f"{context}{question}"
    if answer_prefix and not question.rstrip().endswith(answer_prefix.rstrip()):
        prompt += answer_prefix
    return prompt


def _longbench_max_new_tokens(
    doc: dict[str, Any],
    *,
    task_root: str,
    override: int | None,
) -> int:
    if override is not None:
        return override
    raw_value = doc.get("max_new_tokens")
    if raw_value is not None:
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            pass
    return _LONG_BENCH_DEFAULT_MAX_NEW_TOKENS[task_root]


def _trim_prediction(prediction: str, *, single_line: bool) -> str:
    if not single_line:
        return prediction
    return prediction.lstrip("\n").split("\n")[0]


def _best_longbench_score(
    prediction: str,
    references: list[str],
    *,
    metric_name: str,
    all_classes: list[str],
) -> tuple[float, int]:
    best_score = 0.0
    best_index = 0
    for index, reference in enumerate(references):
        if metric_name == "qa_f1_score":
            score = longbench_qa_f1_score(prediction, reference)
        elif metric_name == "qa_f1_zh_score":
            score = longbench_qa_f1_zh_score(prediction, reference)
        elif metric_name == "rouge_score":
            score = longbench_rouge_score(prediction, reference)
        elif metric_name == "rouge_zh_score":
            score = longbench_rouge_zh_score(prediction, reference)
        elif metric_name == "classification_score":
            score = longbench_classification_score(
                prediction,
                reference,
                all_classes=all_classes,
            )
        elif metric_name == "retrieval_score":
            score = longbench_retrieval_score(prediction, reference)
        elif metric_name == "retrieval_zh_score":
            score = longbench_retrieval_zh_score(prediction, reference)
        elif metric_name == "count_score":
            score = longbench_count_score(prediction, reference)
        elif metric_name == "code_sim_score":
            score = longbench_code_sim_score(prediction, reference)
        else:
            raise ValueError(f"unsupported longbench metric: {metric_name!r}")
        if score > best_score or (score == best_score and index == 0):
            best_score = score
            best_index = index
    return best_score, best_index


@dataclass(slots=True)
class LongBench(BaseTestSuite):
    # Evaluate one normalized LongBench or LongBench-E subset using the author task-specific metric.
    dataset_path: str = "Xnhyacinth/LongBench"
    dataset_name: str | None = "qasper"
    split: str = "test"
    subset: str = "qasper"
    max_new_tokens: int | None = None
    do_sample: bool = False
    temperature: float = 0.0

    def __post_init__(self) -> None:
        task_name = _longbench_task_name(self.subset)
        dataset_name = LONG_BENCH_TASK_TO_DATASET[task_name]
        self.subset = task_name
        if self.dataset_name in {None, dataset_name}:
            self.dataset_name = dataset_name
            return
        raise ValueError("longbench dataset_name must match the configured subset")

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return self.subset

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        if self.dataset_name is None:
            raise ValueError("longbench dataset_name cannot be None")
        task_root = _longbench_task_root(self.dataset_name)
        metric_spec = _LONG_BENCH_METRICS[task_root]
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "subset": self.subset,
            "task_root": task_root,
            "variant": "e" if self.dataset_name.endswith("_e") else "base",
            "scoring_mode": metric_spec.scoring_mode,
            "primary_metric": "score",
            "metric_name": metric_spec.metric_name,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        if self.dataset_name is None:
            raise ValueError("longbench dataset_name cannot be None")
        task_root = _longbench_task_root(self.dataset_name)
        metric_spec = _LONG_BENCH_METRICS[task_root]
        for index, doc in enumerate(docs):
            yield PreparedSample(
                index=index,
                doc=doc,
                target=_longbench_answers(doc)[0],
                request=GenerationRequest(
                    prompt=_longbench_prompt(doc),
                    stop=list(metric_spec.stop_strings),
                    max_new_tokens=_longbench_max_new_tokens(
                        doc,
                        task_root=task_root,
                        override=self.max_new_tokens,
                    ),
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        if self.dataset_name is None:
            raise ValueError("longbench dataset_name cannot be None")
        task_root = _longbench_task_root(self.dataset_name)
        metric_spec = _LONG_BENCH_METRICS[task_root]
        references = _longbench_answers(prepared_sample.doc)
        all_classes = _longbench_all_classes(prepared_sample.doc)
        prediction_scored = _trim_prediction(
            output.text,
            single_line=metric_spec.single_line_prediction,
        )
        score, best_index = _best_longbench_score(
            prediction_scored,
            references,
            metric_name=metric_spec.metric_name,
            all_classes=all_classes,
        )
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-scored": prediction_scored,
                "best_reference_index": str(best_index),
                "best_reference": references[best_index],
            },
            scores={
                "score": score,
                metric_spec.metric_name: score,
            },
            metadata={
                "dataset": str(prepared_sample.doc.get("dataset", self.dataset_name)),
                "task": str(prepared_sample.doc.get("task", task_root)),
                "language": str(prepared_sample.doc.get("language", "")),
                "length": str(prepared_sample.doc.get("length", "")),
                "answers": references,
                "all_classes": all_classes,
            },
        )


def longbench(*, subset: str = "qasper", **kwargs: Any) -> LongBench:
    task_name = _longbench_task_name(subset)
    kwargs.setdefault("dataset_name", LONG_BENCH_TASK_TO_DATASET[task_name])
    return LongBench(subset=task_name, **kwargs)


def _make_longbench_factory(task_name: str) -> Any:
    def factory(**kwargs: Any) -> LongBench:
        return longbench(subset=task_name, **kwargs)

    factory.__name__ = task_name
    return factory


for _task_name in LONG_BENCH_TASKS:
    globals()[_task_name] = _make_longbench_factory(_task_name)

del _task_name
