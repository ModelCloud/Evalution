# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
copal_id_module = importlib.import_module("evalution.benchmarks.copal_id")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 2
        if requests[0].context == "Pria itu memangku tasnya saat menaiki angkutan umum karena":
            assert requests[0].continuation == " pria itu berjaga-jaga agar tasnya tidak dicuri."
            assert requests[1].continuation == " tasnya empuk."
            return [
                LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=8),
                LoglikelihoodOutput(logprob=-4.0, is_greedy=False, token_count=4),
            ]
        assert requests[0].context == "Cowok itu mangku tasnya pas lagi naek angkot karena"
        assert requests[0].continuation == " cowoknya jaga-jaga biar tasnya gak diambil"
        assert requests[1].continuation == " tasnya empukk"
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=8),
            LoglikelihoodOutput(logprob=-3.5, is_greedy=False, token_count=4),
        ]


def test_copal_id_scores_standard_and_colloquial_rows(monkeypatch) -> None:
    """Verify copal id scores standard and colloquial rows. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    standard_dataset = Dataset.from_list(
        [
            {
                "premise": "Pria itu memangku tasnya saat menaiki angkutan umum.",
                "choice1": "Pria itu berjaga-jaga agar tasnya tidak dicuri.",
                "choice2": "Tasnya empuk.",
                "question": "cause",
                "idx": 0,
                "label": 0,
                "Terminology": 0,
                "Culture": 1,
                "Language": 0,
            }
        ]
    )
    colloquial_dataset = Dataset.from_list(
        [
            {
                "premise": "Cowok itu mangku tasnya pas lagi naek angkot",
                "choice1": "Cowoknya jaga-jaga biar tasnya gak diambil",
                "choice2": "Tasnya empukk",
                "question": "cause",
                "idx": 0,
                "label": 0,
                "Terminology": 0,
                "Culture": 1,
                "Language": 0,
            }
        ]
    )

    def fake_load_dataset(dataset_path, dataset_name, *, split, **kwargs):
        """Support the surrounding tests with fake load dataset."""
        assert dataset_path == "haryoaw/COPAL"
        assert dataset_name == "id"
        if split == "test":
            return standard_dataset
        if split == "test_colloquial":
            return colloquial_dataset
        raise AssertionError(f"unexpected split: {split}")

    monkeypatch.setattr(copal_id_module, "load_dataset", fake_load_dataset)

    standard_result = evalution.benchmarks.copal_id_standard(max_rows=1, batch_size=6).evaluate(FakeSession())
    colloquial_result = evalution.benchmarks.copal_id_colloquial(max_rows=1, batch_size=6).evaluate(FakeSession())

    assert standard_result.name == "copal_id_standard"
    assert standard_result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
    }
    assert standard_result.metadata["dataset_path"] == "haryoaw/COPAL"
    assert standard_result.metadata["dataset_name"] == "id"
    assert standard_result.metadata["split"] == "test"
    assert standard_result.samples[0].prompt == "Pria itu memangku tasnya saat menaiki angkutan umum karena"
    assert standard_result.samples[0].target == "pria itu berjaga-jaga agar tasnya tidak dicuri."
    assert standard_result.samples[0].metadata["variant"] == "standard"
    assert standard_result.samples[0].metadata["culture"] is True

    assert colloquial_result.name == "copal_id_colloquial"
    assert colloquial_result.metadata["split"] == "test_colloquial"
    assert colloquial_result.samples[0].prompt == "Cowok itu mangku tasnya pas lagi naek angkot karena"
    assert colloquial_result.samples[0].target == "cowoknya jaga-jaga biar tasnya gak diambil"
    assert colloquial_result.samples[0].metadata["variant"] == "colloquial"


def test_copal_id_helpers_format_connectors_choices_and_prompt() -> None:
    """Verify copal id helpers format connectors choices and prompt."""
    assert copal_id_module._copal_id_connector("cause") == "karena"
    assert copal_id_module._copal_id_connector("effect") == "maka"
    assert copal_id_module._copal_id_choice_text("Pilihan Satu") == "pilihan Satu"
    assert copal_id_module._copal_id_prompt("Kalimat dengan titik.", "effect") == "Kalimat dengan titik maka"
    assert copal_id_module._copal_id_prompt("Kalimat tanpa titik", "cause") == "Kalimat tanpa titik karena"


def test_copal_id_rejects_unknown_variant_and_mismatched_split() -> None:
    """Verify copal id rejects unknown variant and mismatched split."""
    with pytest.raises(ValueError, match="unsupported copal_id variant"):
        evalution.benchmarks.copal_id(variant="unknown")

    with pytest.raises(ValueError, match="split must match the configured variant"):
        copal_id_module.COPALID(variant="standard", split="test_colloquial")
