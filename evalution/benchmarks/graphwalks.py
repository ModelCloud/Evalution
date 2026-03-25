# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List

from datasets import load_dataset

import pcre

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult


_FINAL_ANSWER_RE = pcre.compile(r"final answer:\s*\[(.*)\]", pcre.IGNORECASE)


def _split_nodes(text: str) -> List[str]:
    if not text:
        return []
    return [token.strip().strip("'\"") for token in text.split(",") if token.strip()]


def _extract_final_answer(line: str) -> tuple[List[str], bool]:
    match = _FINAL_ANSWER_RE.search(line)
    if not match:
        return [], True
    content = match.group(1).strip()
    return _split_nodes(content), False


def _parse_final_answer(text: str, *, flexible: bool = False) -> tuple[List[str], bool]:
    if not text:
        return [], True
    lines = [line for line in text.rstrip().splitlines()] or [""]
    if flexible:
        for line in reversed(lines):
            nodes, error = _extract_final_answer(line)
            if not error:
                return nodes, False
        return [], True
    for line in reversed(lines):
        trimmed = line.strip()
        if trimmed:
            return _extract_final_answer(trimmed)
    return [], True


def _set_f1(predicted: list[str], reference: list[str]) -> float:
    predicted_set = set(predicted)
    reference_set = set(reference)
    if not reference_set and not predicted_set:
        return 1.0
    if not reference_set or not predicted_set:
        return 0.0
    intersection = predicted_set & reference_set
    precision = len(intersection) / len(predicted_set)
    recall = len(intersection) / len(reference_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass(slots=True)
class GraphWalks(BaseTestSuite):
    dataset_path: str = "openai/graphwalks"
    dataset_name: str | None = None
    split: str = "train"
    data_file: str = "graphwalks_128k_and_shorter.parquet"
    generation_stop: tuple[str, ...] = ("</s>", "<|im_end|>", "<|endoftext|>")
    max_new_tokens: int = 16384
    do_sample: bool = False
    temperature: float = 0.0
    task_variant: str = "graphwalks_128k"

    def dataset_loader(self) -> Callable[..., Any]:
        def loader(path: str, *, split: str, cache_dir: str | None, stream: bool) -> Any:
            return load_dataset(
                path,
                data_files=self.data_file,
                split=split,
                cache_dir=cache_dir,
                stream=stream,
            )

        return loader

    def task_name(self) -> str:
        return self.task_variant

    def result_metadata(self, *, generation_submission_mode: str) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "graphwalks_set_f1",
            "data_file": self.data_file,
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Iterable[dict[str, Any]]) -> Iterable[PreparedSample]:
        for index, doc in enumerate(docs):
            prompt = str(doc["prompt"]).strip()
            yield PreparedSample(
                index=index,
                doc=doc,
                target=self._format_target(doc.get("answer_nodes", [])),
                request=GenerationRequest(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    stop=list(self.generation_stop),
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def _format_target(self, answer_nodes: list[str]) -> str:
        entries = [str(node) for node in answer_nodes]
        return f"[{', '.join(entries)}]"

    def score_sample(self, prepared_sample: PreparedSample, output: GenerationOutput) -> SampleResult:
        predicted = output.text
        strict_nodes, strict_error = _parse_final_answer(predicted, flexible=False)
        flexible_nodes, flexible_error = _parse_final_answer(predicted, flexible=True)
        reference_nodes = prepared_sample.doc.get("answer_nodes", [])
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=predicted,
            extracted={
                "prediction_nodes_strict": strict_nodes,
                "prediction_nodes_flexible": flexible_nodes,
            },
            scores={
                "f1": _set_f1(strict_nodes, reference_nodes) if not strict_error else 0.0,
                "flexible_f1": _set_f1(flexible_nodes, reference_nodes) if not flexible_error else 0.0,
            },
            metadata={
                "problem_type": prepared_sample.doc.get("problem_type"),
                "prompt_chars": int(prepared_sample.doc.get("prompt_chars", 0)),
                "prediction_lines": len(predicted.splitlines()),
                "strict_parse_error": strict_error,
                "flexible_parse_error": flexible_error,
            },
        )


def graphwalks_128k(**kwargs: Any) -> GraphWalks:
    kwargs.setdefault("stream", False)
    return GraphWalks(**kwargs)


def graphwalks_1M(**kwargs: Any) -> GraphWalks:
    kwargs.setdefault("stream", False)
    return GraphWalks(
        data_file="graphwalks_256k_to_1mil.parquet",
        task_variant="graphwalks_1M",
        **kwargs,
    )
