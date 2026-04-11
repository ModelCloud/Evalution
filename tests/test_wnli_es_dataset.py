# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from evalution.datasets import wnli_es as wnli_es_dataset


def _write_csv(path: Path, rows: str) -> str:
    """Write CSV."""
    path.write_text(rows, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_wnli_es_dataset_reads_pinned_csv(monkeypatch, tmp_path: Path) -> None:
    """Verify load WNLI es dataset reads pinned CSV."""
    train_sha = _write_csv(
        tmp_path / "wnli-train-es.csv",
        "index,sentence1,sentence2,label\n0,Uno,Dos,1\n",
    )
    validation_sha = _write_csv(
        tmp_path / "wnli-dev-es.csv",
        "index,sentence1,sentence2,label\n1,Tres,Cuatro,0\n",
    )
    test_sha = _write_csv(
        tmp_path / "wnli-test-shuffled-es.csv",
        "index,sentence1,sentence2,label\n2,Cinco,Seis,-1\n",
    )
    monkeypatch.setattr(
        wnli_es_dataset,
        "WNLI_ES_FILE_SPECS",
        {
            "train": ("wnli-train-es.csv", train_sha),
            "validation": ("wnli-dev-es.csv", validation_sha),
            "test": ("wnli-test-shuffled-es.csv", test_sha),
        },
    )
    wnli_es_dataset._load_rows_cached.cache_clear()

    def fake_download(repo_id: str, filename: str, *, repo_type: str, cache_dir: str | None = None) -> str:
        """Support the surrounding tests with fake download."""
        assert repo_id == "PlanTL-GOB-ES/wnli-es"
        assert repo_type == "dataset"
        del cache_dir
        return str(tmp_path / filename)

    monkeypatch.setattr(wnli_es_dataset, "hf_hub_download", fake_download)

    docs = wnli_es_dataset.load_wnli_es_dataset(
        "PlanTL-GOB-ES/wnli-es",
        split="validation",
        cache_dir=str(tmp_path / "cache"),
    )

    assert docs == [
        {
            "index": 1,
            "sentence1": "Tres",
            "sentence2": "Cuatro",
            "label": 0,
        }
    ]


def test_load_wnli_es_dataset_rejects_checksum_mismatch(monkeypatch, tmp_path: Path) -> None:
    """Verify load WNLI es dataset rejects checksum mismatch."""
    csv_path = tmp_path / "wnli-dev-es.csv"
    csv_path.write_text("index,sentence1,sentence2,label\n0,A,B,1\n", encoding="utf-8")
    monkeypatch.setattr(
        wnli_es_dataset,
        "WNLI_ES_FILE_SPECS",
        {
            "train": ("wnli-train-es.csv", "0" * 64),
            "validation": ("wnli-dev-es.csv", "f" * 64),
            "test": ("wnli-test-shuffled-es.csv", "0" * 64),
        },
    )
    wnli_es_dataset._load_rows_cached.cache_clear()
    monkeypatch.setattr(
        wnli_es_dataset,
        "hf_hub_download",
        lambda repo_id, filename, *, repo_type, cache_dir=None: str(csv_path),
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        wnli_es_dataset.load_wnli_es_dataset("PlanTL-GOB-ES/wnli-es", split="validation")
