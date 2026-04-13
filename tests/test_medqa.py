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
medqa_module = importlib.import_module("evalution.benchmarks.medqa")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 4
        assert requests[0].context == (
            "Question: A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\n"
            "A. Disclose the error to the patient and put it in the operative report\n"
            "B. Tell the attending that he cannot fail to disclose this mistake\n"
            "C. Report the physician to the ethics committee\n"
            "D. Refuse to dictate the operative report\n"
            "Answer:"
        )
        assert requests[0].continuation == " A"
        assert requests[1].continuation == " B"
        return [
            LoglikelihoodOutput(logprob=-1.6, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=1),
        ]


def test_medqa_scores_four_way_labeled_multiple_choice(monkeypatch) -> None:
    """Verify medqa scores four way labeled multiple choice. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "id": "test-00000",
                "sent1": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
                "sent2": "",
                "ending0": "Disclose the error to the patient and put it in the operative report",
                "ending1": "Tell the attending that he cannot fail to disclose this mistake",
                "ending2": "Report the physician to the ethics committee",
                "ending3": "Refuse to dictate the operative report",
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(medqa_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.medqa_4options(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "medqa_4options"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "GBaker/MedQA-USMLE-4-options-hf"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "test"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.target == "B"
    assert sample.prediction == "B"
    assert sample.extracted == {
        "gold_index": "1",
        "predicted_index": "1",
        "predicted_index_norm": "1",
    }
    assert sample.metadata["id"] == "test-00000"
    assert sample.metadata["question_suffix"] == ""
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["choice_texts"] == [
        "Disclose the error to the patient and put it in the operative report",
        "Tell the attending that he cannot fail to disclose this mistake",
        "Report the physician to the ethics committee",
        "Refuse to dictate the operative report",
    ]


def test_medqa_can_emit_label_permutation_metric(monkeypatch) -> None:
    """Verify medqa can emit label permutation metric. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "id": "test-00000",
                "sent1": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
                "sent2": "",
                "ending0": "Disclose the error to the patient and put it in the operative report",
                "ending1": "Tell the attending that he cannot fail to disclose this mistake",
                "ending2": "Report the physician to the ethics committee",
                "ending3": "Refuse to dictate the operative report",
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(medqa_module, "load_dataset", lambda *args, **kwargs: dataset)

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
                    LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
                    LoglikelihoodOutput(logprob=-1.5, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.7, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=1),
                ]

            assert len(requests) == 24
            gold_text = "Tell the attending that he cannot fail to disclose this mistake"
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

    result = evalution.benchmarks.medqa_4options(
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
    assert result.samples[0].extracted["predicted_index_label_perm:0.25"] == "1"
    assert result.samples[0].metadata["label_permutation_count"] == 6
