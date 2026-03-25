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
from evalution.scorers.multiple_choice import (
    build_choice_scores,
    choice_logprobs,
    choice_logprobs_norm,
    multiple_choice_outcome,
)
from evalution.benchmarks.base import TestSuite
from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
from evalution.benchmarks.subsets import ResolvedSubsets, SubsetTree, normalize_subset_token

_MMLU_LABELS = ["A", "B", "C", "D"]
_MMLU_SUBSET_TREE = {
    "stem": {
        "abstract_algebra": "abstract_algebra",
        "anatomy": "anatomy",
        "astronomy": "astronomy",
        "college_biology": "college_biology",
        "college_chemistry": "college_chemistry",
        "college_computer_science": "college_computer_science",
        "college_mathematics": "college_mathematics",
        "college_physics": "college_physics",
        "computer_security": "computer_security",
        "conceptual_physics": "conceptual_physics",
        "electrical_engineering": "electrical_engineering",
        "elementary_mathematics": "elementary_mathematics",
        "high_school_biology": "high_school_biology",
        "high_school_chemistry": "high_school_chemistry",
        "high_school_computer_science": "high_school_computer_science",
        "high_school_mathematics": "high_school_mathematics",
        "high_school_physics": "high_school_physics",
        "high_school_statistics": "high_school_statistics",
        "machine_learning": "machine_learning",
    },
    "humanities": {
        "formal_logic": "formal_logic",
        "high_school_european_history": "high_school_european_history",
        "high_school_us_history": "high_school_us_history",
        "high_school_world_history": "high_school_world_history",
        "international_law": "international_law",
        "jurisprudence": "jurisprudence",
        "logical_fallacies": "logical_fallacies",
        "moral_disputes": "moral_disputes",
        "moral_scenarios": "moral_scenarios",
        "philosophy": "philosophy",
        "prehistory": "prehistory",
        "professional_law": "professional_law",
        "world_religions": "world_religions",
    },
    "social_sciences": {
        "econometrics": "econometrics",
        "high_school_geography": "high_school_geography",
        "high_school_government_and_politics": "high_school_government_and_politics",
        "high_school_macroeconomics": "high_school_macroeconomics",
        "high_school_microeconomics": "high_school_microeconomics",
        "high_school_psychology": "high_school_psychology",
        "human_sexuality": "human_sexuality",
        "professional_psychology": "professional_psychology",
        "public_relations": "public_relations",
        "security_studies": "security_studies",
        "sociology": "sociology",
        "us_foreign_policy": "us_foreign_policy",
    },
    "other": {
        "business_ethics": "business_ethics",
        "clinical_knowledge": "clinical_knowledge",
        "college_medicine": "college_medicine",
        "global_facts": "global_facts",
        "human_aging": "human_aging",
        "management": "management",
        "marketing": "marketing",
        "medical_genetics": "medical_genetics",
        "miscellaneous": "miscellaneous",
        "nutrition": "nutrition",
        "professional_accounting": "professional_accounting",
        "professional_medicine": "professional_medicine",
        "virology": "virology",
    },
}
_MMLU_SUBSETS = SubsetTree(_MMLU_SUBSET_TREE)


def _format_subject_title(subject: str) -> str:
    return " ".join(subject.split("_"))


def _format_mmlu_question(doc: dict[str, Any], *, include_answer: bool) -> str:
    lines = [doc["question"].strip()]
    for label, choice in zip(_MMLU_LABELS, doc["choices"], strict=True):
        lines.append(f"{label}. {choice}")
    answer_text = _MMLU_LABELS[int(doc["answer"])] if include_answer else ""
    lines.append(f"Answer: {answer_text}".rstrip())
    return "\n".join(lines)


def _fewshot_prompt(subject: str, fewshot_docs: list[dict[str, Any]]) -> str:
    subject_title = _format_subject_title(subject)
    sections = [f"The following are multiple choice questions (with answers) about {subject_title}."]
    if fewshot_docs:
        sections.extend(_format_mmlu_question(doc, include_answer=True) for doc in fewshot_docs)
    return "\n\n".join(sections) + "\n\n"


@dataclass(slots=True)
class MMLU(TestSuite):
    dataset_path: str = "cais/mmlu"
    subsets: str | list[str] = "all"
    # Default to the benchmark-reporting split. Callers can still override `split=` explicitly
    # for development or cross-framework alignment checks.
    split: str = "test"
    fewshot_split: str = "dev"
    num_fewshot: int = 5
    stream: bool = False
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    def dataset_name(self) -> str:
        resolved_subsets = self._resolved_subsets()
        if resolved_subsets.selection_mode == "single" and resolved_subsets.kinds[0] == "leaf":
            return resolved_subsets.leaf_values[0]
        return "all"

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        resolved_subsets = self._resolved_subsets()
        if resolved_subsets.selection_mode == "single" and resolved_subsets.kinds[0] == "all":
            return "mmlu"
        suffix = "__".join(canonical.replace(".", "_") for canonical in resolved_subsets.canonicals)
        return f"mmlu_{suffix}"

    def result_metadata(self) -> dict[str, Any]:
        resolved_subsets = self._resolved_subsets()
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name(),
            "subsets": list(resolved_subsets.canonicals),
            "subset_paths": [list(path) for path in resolved_subsets.paths],
            "subset_kinds": list(resolved_subsets.kinds),
            "selection_mode": resolved_subsets.selection_mode,
            "split": self.split,
            "fewshot_split": self.fewshot_split,
            "num_fewshot": self.num_fewshot,
            "stream": self.stream,
            "scoring_mode": "multiple_choice_loglikelihood",
        }

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
            stream=self.stream,
        )
        docs = self._select_docs(list(loaded_docs))
        docs = limit_docs(docs, self.max_rows)
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
            stream=self.stream,
        )
        fewshot_docs = self._select_docs(list(fewshot_loaded_docs))
        fewshot_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for doc in fewshot_docs:
            fewshot_by_subject[normalize_subset_token(doc["subject"])].append(doc)

        prompts: list[str] = []
        requests: list[LoglikelihoodRequest] = []
        sample_docs: list[dict[str, Any]] = []
        for doc in docs:
            subject = str(doc["subject"])
            subject_key = normalize_subset_token(subject)
            prompt = _fewshot_prompt(subject, fewshot_by_subject[subject_key][: self.num_fewshot])
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
            choice_scores = build_choice_scores(
                (
                    choice_index,
                    output.logprob,
                    output.token_count,
                )
                for choice_index, output in enumerate(choice_outputs)
            )
            gold_index = int(doc["answer"])
            outcome = multiple_choice_outcome(choice_scores, gold_index)
            raw_total += outcome.raw_accuracy
            norm_total += outcome.normalized_accuracy
            leaf_subset = _MMLU_SUBSETS.leaf_subset(doc["subject"])
            sample_results.append(
                SampleResult(
                    index=index,
                    prompt=prompts[index],
                    target=_MMLU_LABELS[gold_index],
                    prediction=_MMLU_LABELS[outcome.normalized_best_index],
                    extracted={
                        "gold_index": str(gold_index),
                        "predicted_index": str(outcome.raw_best_index),
                        "predicted_index_norm": str(outcome.normalized_best_index),
                    },
                    scores={
                        "acc,ll": outcome.raw_accuracy,
                        "acc,ll_avg": outcome.normalized_accuracy,
                    },
                    metadata={
                        "subset": leaf_subset,
                        "subset_path": leaf_subset.split("."),
                        "subset_kind": "leaf",
                        "subset_value": str(doc["subject"]),
                        "choice_texts": list(doc["choices"]),
                        "choice_logprobs": choice_logprobs(choice_scores),
                        "choice_logprobs_norm": choice_logprobs_norm(choice_scores),
                    },
                )
            )

        denominator = max(len(sample_results), 1)
        metrics = {
            "acc,ll": raw_total / denominator,
            "acc,ll_avg": norm_total / denominator,
        }
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=sample_results,
            metadata=self.result_metadata(),
        )

    def _resolved_subsets(self) -> ResolvedSubsets:
        return _MMLU_SUBSETS.resolve_many(self.subsets)

    def _selected_subjects(self) -> set[str] | None:
        resolved_subsets = self._resolved_subsets()
        if resolved_subsets.selection_mode == "single" and resolved_subsets.kinds[0] == "all":
            return None
        return {normalize_subset_token(subject) for subject in resolved_subsets.leaf_values}

    def _select_docs(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected_subjects = self._selected_subjects()
        if selected_subjects is None:
            return docs
        selected_docs = [
            doc
            for doc in docs
            if normalize_subset_token(doc.get("subject", "")) in selected_subjects
        ]
        if selected_docs:
            return selected_docs
        raise ValueError(f"MMLU subsets {self.subsets!r} are not present in the dataset")


def mmlu(**kwargs: Any) -> MMLU:
    return MMLU(**kwargs)
