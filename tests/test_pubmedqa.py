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
pubmedqa_module = importlib.import_module("evalution.benchmarks.pubmedqa")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 3
        assert requests[0].context == (
            "Abstract: Programmed cell death (PCD) is the regulated death of cells within an organism.\n"
            "The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis.\n"
            "Question: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?\n"
            "Answer:"
        )
        assert requests[0].continuation == " yes"
        assert requests[1].continuation == " no"
        assert requests[2].continuation == " maybe"
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.6, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.9, is_greedy=False, token_count=1),
        ]


def test_pubmedqa_scores_three_way_multiple_choice(monkeypatch) -> None:
    """Verify pubmedqa scores three way multiple choice. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "PUBID": "21645374",
                "QUESTION": "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
                "CONTEXTS": [
                    "Programmed cell death (PCD) is the regulated death of cells within an organism.",
                    "The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis.",
                ],
                "LABELS": ["BACKGROUND", "RESULTS"],
                "MESHES": ["Alismataceae", "Apoptosis"],
                "reasoning_required_pred": "yes",
                "reasoning_free_pred": "yes",
                "LONG_ANSWER": "Overall, our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD in the lace plant.",
                "final_decision": "yes",
            }
        ]
    )
    monkeypatch.setattr(pubmedqa_module, "_load_pubmedqa_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.pubmedqa(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert result.name == "pubmedqa"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert result.metadata["dataset_path"] == "bigbio/pubmed_qa"
    assert result.metadata["dataset_name"] == "pubmed_qa_labeled_fold0_source"
    assert result.metadata["split"] == "test"
    assert result.metadata["scoring_mode"] == "multiple_choice_loglikelihood"
    assert len(result.samples) == 1

    sample = result.samples[0]
    assert sample.target == "yes"
    assert sample.prediction == "yes"
    assert sample.extracted == {
        "gold_index": "0",
        "predicted_index": "0",
        "predicted_index_norm": "0",
    }
    assert sample.metadata["pubid"] == "21645374"
    assert sample.metadata["context_labels"] == ["BACKGROUND", "RESULTS"]
    assert sample.metadata["meshes"] == ["Alismataceae", "Apoptosis"]
    assert sample.metadata["reasoning_required_pred"] == "yes"
    assert sample.metadata["reasoning_free_pred"] == "yes"
    assert sample.metadata["choice_labels"] == ["yes", "no", "maybe"]
    assert sample.metadata["choice_texts"] == ["yes", "no", "maybe"]


def test_pubmedqa_can_emit_label_permutation_metric(monkeypatch) -> None:
    """Verify pubmedqa can emit label permutation metric. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "PUBID": "21645374",
                "QUESTION": "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
                "CONTEXTS": [
                    "Programmed cell death (PCD) is the regulated death of cells within an organism.",
                    "The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis.",
                ],
                "LABELS": ["BACKGROUND", "RESULTS"],
                "MESHES": ["Alismataceae", "Apoptosis"],
                "reasoning_required_pred": "yes",
                "reasoning_free_pred": "yes",
                "LONG_ANSWER": "Overall, our findings implicate the mitochondria as playing a critical and early role in developmentally regulated PCD in the lace plant.",
                "final_decision": "yes",
            }
        ]
    )
    monkeypatch.setattr(pubmedqa_module, "_load_pubmedqa_dataset", lambda *args, **kwargs: dataset)

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
                    LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
                    LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
                    LoglikelihoodOutput(logprob=-1.9, is_greedy=False, token_count=1),
                ]

            assert len(requests) == 9
            outputs = []
            for request in requests:
                label = request.continuation.strip()
                is_gold_label = f"{label}. yes" in request.context
                outputs.append(
                    LoglikelihoodOutput(
                        logprob=-0.1 if is_gold_label else -1.0,
                        is_greedy=is_gold_label,
                        token_count=1,
                    )
                )
            return outputs

    result = evalution.benchmarks.pubmedqa(
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


def test_pubmedqa_loader_reads_raw_zip_members(tmp_path, monkeypatch) -> None:
    """Verify pubmedqa loader reads raw zip members."""
    archive_path = tmp_path / "pqal.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "pqal_test_set.json",
            json.dumps(
                {
                    "12377809": {
                        "QUESTION": "Is anorectal endosonography valuable in dyschesia?",
                        "CONTEXTS": [
                            "Dyschesia can be provoked by inappropriate defecation movements.",
                            "Twenty consecutive patients with a medical history of dyschesia and a control group of 20 healthy subjects underwent linear anorectal endosonography.",
                            "The anal sphincter became paradoxically shorter and/or thicker during straining in patients versus control subjects.",
                        ],
                        "LABELS": ["AIMS", "METHODS", "RESULTS"],
                        "MESHES": ["Anal Canal", "Constipation"],
                        "YEAR": "2002",
                        "reasoning_required_pred": "yes",
                        "reasoning_free_pred": "yes",
                        "final_decision": "yes",
                        "LONG_ANSWER": "Linear anorectal endosonography highlighted the value of this technique in the diagnosis of pelvic floor dyssynergia.",
                    }
                }
            ),
        )
    monkeypatch.setattr(pubmedqa_module, "_ensure_pubmedqa_archive", lambda *, cache_dir: archive_path)

    dataset = pubmedqa_module._load_pubmedqa_dataset(
        "bigbio/pubmed_qa",
        dataset_name="pubmed_qa_labeled_fold0_source",
        split="test",
        cache_dir=None,
    )

    assert len(dataset) == 1
    assert dataset[0] == {
        "PUBID": "12377809",
        "QUESTION": "Is anorectal endosonography valuable in dyschesia?",
        "CONTEXTS": [
            "Dyschesia can be provoked by inappropriate defecation movements.",
            "Twenty consecutive patients with a medical history of dyschesia and a control group of 20 healthy subjects underwent linear anorectal endosonography.",
            "The anal sphincter became paradoxically shorter and/or thicker during straining in patients versus control subjects.",
        ],
        "LABELS": ["AIMS", "METHODS", "RESULTS"],
        "MESHES": ["Anal Canal", "Constipation"],
        "YEAR": "2002",
        "reasoning_required_pred": "yes",
        "reasoning_free_pred": "yes",
        "final_decision": "yes",
        "LONG_ANSWER": "Linear anorectal endosonography highlighted the value of this technique in the diagnosis of pelvic floor dyssynergia.",
    }
