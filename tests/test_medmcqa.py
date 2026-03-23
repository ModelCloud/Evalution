# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

medmcqa_module = importlib.import_module("evalution.benchmarks.medmcqa")


class FakeSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 4
        assert requests[0].context == (
            "Question: Which of the following is not true for myelinated nerve fibers:\n"
            "Choices:\n"
            "A. Impulse through myelinated fibers is slower than non-myelinated fibers\n"
            "B. Membrane currents are generated at nodes of Ranvier\n"
            "C. Saltatory conduction of impulses is seen\n"
            "D. Local anesthesia is effective only when the nerve is not covered by myelin sheath\n"
            "Answer:"
        )
        assert requests[0].continuation == " A"
        assert requests[3].continuation == " D"
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.6, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=1),
        ]


def test_medmcqa_scores_four_way_labeled_multiple_choice(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "45258d3d-b974-44dd-a161-c3fccbdadd88",
                "question": "Which of the following is not true for myelinated nerve fibers:",
                "opa": "Impulse through myelinated fibers is slower than non-myelinated fibers",
                "opb": "Membrane currents are generated at nodes of Ranvier",
                "opc": "Saltatory conduction of impulses is seen",
                "opd": "Local anesthesia is effective only when the nerve is not covered by myelin sheath",
                "cop": 0,
                "choice_type": "multi",
                "exp": None,
                "subject_name": "Physiology",
                "topic_name": None,
            }
        ]
    )
    monkeypatch.setattr(medmcqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.medmcqa(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "medmcqa"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "openlifescienceai/medmcqa"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "validation"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.target == "A"
    assert sample.prediction == "A"
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }
    assert sample.metadata["id"] == "45258d3d-b974-44dd-a161-c3fccbdadd88"
    assert sample.metadata["choice_type"] == "multi"
    assert sample.metadata["subject_name"] == "Physiology"
    assert sample.metadata["topic_name"] is None
    assert sample.metadata["explanation"] is None
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_texts"] == [
        "Impulse through myelinated fibers is slower than non-myelinated fibers",
        "Membrane currents are generated at nodes of Ranvier",
        "Saltatory conduction of impulses is seen",
        "Local anesthesia is effective only when the nerve is not covered by myelin sheath",
    ]


def test_medmcqa_can_emit_label_permutation_metric(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": "45258d3d-b974-44dd-a161-c3fccbdadd88",
                "question": "Which of the following is not true for myelinated nerve fibers:",
                "opa": "Impulse through myelinated fibers is slower than non-myelinated fibers",
                "opb": "Membrane currents are generated at nodes of Ranvier",
                "opc": "Saltatory conduction of impulses is seen",
                "opd": "Local anesthesia is effective only when the nerve is not covered by myelin sheath",
                "cop": 0,
                "choice_type": "multi",
                "exp": None,
                "subject_name": "Physiology",
                "topic_name": None,
            }
        ]
    )
    monkeypatch.setattr(medmcqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    class LabelPermutationSession:
        def __init__(self) -> None:
            self.calls = 0

        def loglikelihood(self, requests, *, batch_size=None):
            assert batch_size == 6
            self.calls += 1
            if self.calls == 1:
                assert len(requests) == 4
                return [
                    LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
                    LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=1),
                ]

            assert len(requests) == 24
            gold_text = "Impulse through myelinated fibers is slower than non-myelinated fibers"
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. {gold_text}" in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.4,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.benchmarks.medmcqa(
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
