# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
import json
from zipfile import ZipFile

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

mathqa_module = importlib.import_module("evalution.benchmarks.mathqa")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 5
        assert requests[0].context == "Question: What is 2 plus 3?\nAnswer:"
        assert requests[0].continuation == " 4"
        assert requests[4].continuation == " 7"
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
        ]


def test_mathqa_scores_five_way_multiple_choice_accuracy(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "Problem": "What is 2 plus 3?",
                "Rationale": "2 + 3 = 5",
                "options": "a ) 4, b ) 5, c ) 6, d ) 3, e ) 7",
                "correct": "b",
                "annotated_formula": "add(2,3)",
                "linear_formula": "add(n0,n1)",
                "category": "arithmetic",
            }
        ]
    )
    monkeypatch.setattr(mathqa_module, "_load_mathqa_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.mathqa(max_rows=1, batch_size=8).evaluate(FakeSession())

    assert result.name == "mathqa"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "math_qa"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "test"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == "Question: What is 2 plus 3?\nAnswer:"
    assert sample.target == "5"
    assert sample.prediction == "5"
    assert sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D", "E"]
    assert sample.metadata["choice_texts"] == ["4", "5", "6", "3", "7"]


def test_mathqa_can_emit_label_permutation_metric(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "Problem": "What is 2 plus 3?",
                "Rationale": "2 + 3 = 5",
                "options": "a ) 4, b ) 5, c ) 6, d ) 3, e ) 7",
                "correct": "b",
                "annotated_formula": "add(2,3)",
                "linear_formula": "add(n0,n1)",
                "category": "arithmetic",
            }
        ]
    )
    monkeypatch.setattr(mathqa_module, "_load_mathqa_dataset", lambda *args, **kwargs: dataset)

    class LabelPermutationSession:
        def __init__(self) -> None:
            self.calls = 0

        def loglikelihood(self, requests, *, batch_size=None):
            assert batch_size == 8
            self.calls += 1
            if self.calls == 1:
                assert len(requests) == 5
                return [
                    LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
                    LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-2.0, is_greedy=False, token_count=1),
                ]

            assert len(requests) == 150
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. 5" in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.1,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.benchmarks.mathqa(
        max_rows=1,
        batch_size=8,
        label_permutations=0.25,
    ).evaluate(LabelPermutationSession())

    assert result.metrics == {
        "acc,ll": 0.0,
        "acc,ll_avg": 0.0,
        "acc,label_perm:0.25": 1.0,
    }
    assert result.metadata["label_permutations"] == 0.25
    assert result.metadata["label_permutation_metric"] == "acc,label_perm:0.25"
    assert result.samples[0].extracted["predicted_index_label_perm:0.25"] == "1"
    assert result.samples[0].metadata["label_permutation_count"] == 30


def test_mathqa_loader_reads_raw_zip_member(tmp_path, monkeypatch) -> None:
    archive_path = tmp_path / "MathQA.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "dev.json",
            json.dumps(
                [
                    {
                        "Problem": "What is 2 plus 3?",
                        "Rationale": "2 + 3 = 5",
                        "options": "a ) 4, b ) 5, c ) 6, d ) 3, e ) 7",
                        "correct": "b",
                        "annotated_formula": "add(2,3)",
                        "linear_formula": "add(n0,n1)",
                        "category": "arithmetic",
                    }
                ]
            ),
        )

    monkeypatch.setattr(
        mathqa_module,
        "_ensure_mathqa_archive",
        lambda *, cache_dir: archive_path,
    )

    dataset = mathqa_module._load_mathqa_dataset(
        "math_qa",
        split="validation",
        cache_dir=None,
    )

    assert len(dataset) == 1
    assert dataset[0] == {
        "Problem": "What is 2 plus 3?",
        "Rationale": "2 + 3 = 5",
        "options": "a ) 4, b ) 5, c ) 6, d ) 3, e ) 7",
        "correct": "b",
        "annotated_formula": "add(2,3)",
        "linear_formula": "add(n0,n1)",
        "category": "arithmetic",
    }
