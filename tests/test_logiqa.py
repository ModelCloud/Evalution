# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
logiqa_module = importlib.import_module("evalution.benchmarks.logiqa")


class FakeSession:
    # Return deterministic per-choice scores so the suite can be tested without a real model.
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 4
        assert requests[0].context == (
            "Passage: The library closes at sunset.  Tonight sunset is at 8 pm.\n"
            "Question: When should Mei leave if she wants one extra hour to read?\n"
            "Choices:\n"
            "A. 7 pm\n"
            "B. 8 pm\n"
            "C. 9 pm\n"
            "D. Noon\n"
            "Answer:"
        )
        assert requests[0].continuation == " 7 pm"
        assert requests[3].continuation == " Noon"
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=2),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=2),
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=2),
            LoglikelihoodOutput(logprob=-2.3, is_greedy=False, token_count=1),
        ]


def test_logiqa_scores_four_way_multiple_choice_accuracy(monkeypatch) -> None:
    """Verify logiqa scores four way multiple choice accuracy. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "label": "a",
                "context": "The library closes at sunset.  Tonight sunset is at 8 pm.",
                "question": "When should Mei leave if she wants one extra hour to read?",
                "options": ["7 pm", "8 pm", "9 pm", "Noon"],
            }
        ]
    )
    monkeypatch.setattr(logiqa_module, "_load_logiqa_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.logiqa(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "logiqa"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "EleutherAI/logiqa"
    assert result.metadata["dataset_name"] == "logiqa"
    assert result.metadata["split"] == "validation"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.target == "7 pm"
    assert sample.prediction == "7 pm"
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_texts"] == ["7 pm", "8 pm", "9 pm", "Noon"]


def test_logiqa_can_emit_label_permutation_metric(monkeypatch) -> None:
    """Verify logiqa can emit label permutation metric. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "label": "a",
                "context": "The library closes at sunset.",
                "question": "When should Mei leave if she wants one extra hour to read?",
                "options": ["7 pm", "8 pm", "9 pm", "Noon"],
            }
        ]
    )
    monkeypatch.setattr(logiqa_module, "_load_logiqa_dataset", lambda *args, **kwargs: dataset)

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
                assert len(requests) == 4
                return [
                    LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=2),
                    LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=2),
                    LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=2),
                    LoglikelihoodOutput(logprob=-2.3, is_greedy=False, token_count=1),
                ]

            assert len(requests) == 24
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. 7 pm" in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.2,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.benchmarks.logiqa(
        max_rows=1,
        batch_size=6,
        label_permutations=0.25,
    ).evaluate(LabelPermutationSession())

    assert result.metrics == {
        "acc,ll": 0.0,
        "acc,ll_avg": 0.0,
        "acc,label_perm:0.25": 1.0,
    }
    assert result.metadata["label_permutations"] == 0.25
    assert result.metadata["label_permutation_metric"] == "acc,label_perm:0.25"
    assert result.samples[0].extracted["predicted_index_label_perm:0.25"] == "0"
    assert result.samples[0].metadata["label_permutation_count"] == 6


def test_logiqa_loader_reads_raw_source_file(tmp_path, monkeypatch) -> None:
    """Verify logiqa loader reads raw source file."""
    split_path = tmp_path / "Eval.txt"
    split_path.write_text(
        "a\n"
        "The library closes at sunset.\n"
        "When should Mei leave if she wants one extra hour to read?\n"
        "A. 7 pm\n"
        "B. 8 pm\n"
        "C. 9 pm\n"
        "D. Noon\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        logiqa_module,
        "_ensure_logiqa_split_file",
        lambda split, *, cache_dir: split_path,
    )

    dataset = logiqa_module._load_logiqa_dataset(
        "EleutherAI/logiqa",
        "logiqa",
        split="validation",
    )

    assert len(dataset) == 1
    assert dataset[0] == {
        "label": "a",
        "context": "The library closes at sunset.",
        "question": "When should Mei leave if she wants one extra hour to read?",
        "options": ["7 pm", "8 pm", "9 pm", "Noon"],
    }
