# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib
import json
from zipfile import ZipFile

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
siqa_module = importlib.import_module("evalution.benchmarks.siqa")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 3
        assert requests[0].context == (
            "Q: Jordan wanted to tell Tracy a secret. Why did Jordan lean toward Tracy?\nA:"
        )
        assert requests[0].continuation == " To make sure no one else could hear"
        assert requests[1].continuation == " To ask Tracy to move away"
        assert requests[2].continuation == " To borrow Tracy's phone"
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=9),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=7),
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=6),
        ]


def test_siqa_scores_three_way_multiple_choice_accuracy(monkeypatch) -> None:
    """Verify SIQA scores three way multiple choice accuracy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "context": "Jordan wanted to tell Tracy a secret.",
                "question": "Why did Jordan lean toward Tracy?",
                "answerA": "To make sure no one else could hear",
                "answerB": "To ask Tracy to move away",
                "answerC": "To borrow Tracy's phone",
                "label": "1",
            }
        ]
    )
    monkeypatch.setattr(siqa_module, "_load_social_iqa_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.siqa(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "siqa"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "allenai/social_i_qa"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "validation"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.prompt == (
        "Q: Jordan wanted to tell Tracy a secret. Why did Jordan lean toward Tracy?\nA:"
    )
    assert sample.target == "To make sure no one else could hear"
    assert sample.prediction == "To make sure no one else could hear"
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }
    assert sample.metadata["choice_texts"] == [
        "To make sure no one else could hear",
        "To ask Tracy to move away",
        "To borrow Tracy's phone",
    ]


def test_siqa_can_emit_label_permutation_metric(monkeypatch) -> None:
    """Verify SIQA can emit label permutation metric. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "context": "Jordan wanted to tell Tracy a secret.",
                "question": "Why did Jordan lean toward Tracy?",
                "answerA": "To make sure no one else could hear",
                "answerB": "To ask Tracy to move away",
                "answerC": "To borrow Tracy's phone",
                "label": "1",
            }
        ]
    )
    monkeypatch.setattr(siqa_module, "_load_social_iqa_dataset", lambda *args, **kwargs: dataset)

    class LabelPermutationSession:
        """Define the label permutation session helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.calls = 0

        def loglikelihood(self, requests, *, batch_size=None):
            """Implement loglikelihood for label permutation session."""
            assert batch_size == 6
            self.calls += 1
            if self.calls == 1:
                assert len(requests) == 3
                return [
                    LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=7),
                    LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=7),
                    LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=6),
                ]

            assert len(requests) == 9
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. To make sure no one else could hear" in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.0,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.benchmarks.siqa(
        max_rows=1,
        batch_size=6,
        label_permutations=0.5,
    ).evaluate(LabelPermutationSession())

    assert result.metrics == {
        "acc,ll": 0.0,
        "acc,ll_avg": 0.0,
        "acc,label_perm:0.5": 1.0,
    }
    assert result.metadata["label_permutations"] == 0.5
    assert result.metadata["label_permutation_metric"] == "acc,label_perm:0.5"
    assert result.samples[0].extracted["predicted_index_label_perm:0.5"] == "0"
    assert result.samples[0].metadata["label_permutation_count"] == 3


def test_siqa_loader_reads_raw_zip_members(tmp_path, monkeypatch) -> None:
    """Verify SIQA loader reads raw zip members."""
    archive_path = tmp_path / "socialiqa-train-dev.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "socialiqa-train-dev/dev.jsonl",
            json.dumps(
                {
                    "context": "Jordan wanted to tell Tracy a secret.",
                    "question": "Why did Jordan lean toward Tracy?",
                    "answerA": "To make sure no one else could hear",
                    "answerB": "To ask Tracy to move away",
                    "answerC": "To borrow Tracy's phone",
                }
            )
            + "\n",
        )
        archive.writestr("socialiqa-train-dev/dev-labels.lst", "1\n")

    monkeypatch.setattr(
        siqa_module,
        "_ensure_social_iqa_archive",
        lambda *, cache_dir: archive_path,
    )

    dataset = siqa_module._load_social_iqa_dataset(
        "allenai/social_i_qa",
        split="validation",
        cache_dir=None,
    )

    assert len(dataset) == 1
    assert dataset[0] == {
        "context": "Jordan wanted to tell Tracy a secret.",
        "question": "Why did Jordan lean toward Tracy?",
        "answerA": "To make sure no one else could hear",
        "answerB": "To ask Tracy to move away",
        "answerC": "To borrow Tracy's phone",
        "label": "1",
    }
