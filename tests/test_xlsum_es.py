# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import hashlib
import importlib
from io import BytesIO
from pathlib import Path
import tarfile

import pytest

import evalution
from evalution.datasets import xlsum as xlsum_dataset
from evalution.engines.base import GenerationOutput
from evalution.scorers.summary_rouge import summary_rouge_scores

xlsum_es_module = importlib.import_module("evalution.benchmarks.xlsum_es")


def _build_xlsum_archive(path: Path) -> str:
    train_rows = (
        '{"id":"train-1","url":"https://example.com/train","title":"Titulo  de  prueba","summary":"Resumen  de  prueba","text":"Texto  de  prueba"}\n'
    )
    val_rows = (
        '{"id":"val-1","url":"https://example.com/val","title":"Noticia  principal","summary":"Resumen  breve","text":"Texto  largo  con  espacios"}\n'
    )
    test_rows = (
        '{"id":"test-1","url":"https://example.com/test","title":"Otra  noticia","summary":"Resumen  final","text":"Texto  final  con  dobles  espacios"}\n'
    )

    with tarfile.open(path, mode="w:bz2") as archive:
        for member_name, contents in (
            ("./spanish_train.jsonl", train_rows),
            ("./spanish_val.jsonl", val_rows),
            ("./spanish_test.jsonl", test_rows),
        ):
            encoded = contents.encode("utf-8")
            info = tarfile.TarInfo(name=member_name)
            info.size = len(encoded)
            archive.addfile(info, BytesIO(encoded))

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_xlsum_dataset_reads_vendored_archive(monkeypatch, tmp_path: Path) -> None:
    archive_path = tmp_path / "spanish_XLSum_v2.0.tar.bz2"
    archive_sha256 = _build_xlsum_archive(archive_path)
    monkeypatch.setattr(
        xlsum_dataset,
        "XLSUM_ARCHIVES",
        {
            "spanish": {
                "filename": "data/spanish_XLSum_v2.0.tar.bz2",
                "sha256": archive_sha256,
                "split_files": {
                    "train": "spanish_train.jsonl",
                    "validation": "spanish_val.jsonl",
                    "test": "spanish_test.jsonl",
                },
            }
        },
    )
    xlsum_dataset._load_rows_cached.cache_clear()

    def fake_download(repo_id: str, filename: str, *, repo_type: str, cache_dir: str | None = None) -> str:
        assert repo_id == "csebuetnlp/xlsum"
        assert filename == "data/spanish_XLSum_v2.0.tar.bz2"
        assert repo_type == "dataset"
        del cache_dir
        return str(archive_path)

    monkeypatch.setattr(xlsum_dataset, "hf_hub_download", fake_download)

    docs = xlsum_dataset.load_xlsum_dataset(
        "csebuetnlp/xlsum",
        "spanish",
        split="validation",
        cache_dir=str(tmp_path / "cache"),
    )

    assert docs == [
        {
            "id": "val-1",
            "url": "https://example.com/val",
            "title": "Noticia principal",
            "summary": "Resumen breve",
            "text": "Texto largo con espacios",
        }
    ]


def test_safe_extract_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.bz2"
    with tarfile.open(archive_path, mode="w:bz2") as archive:
        encoded = b"owned\n"
        info = tarfile.TarInfo(name="../escape.jsonl")
        info.size = len(encoded)
        archive.addfile(info, BytesIO(encoded))

    with pytest.raises(ValueError, match="unsafe path traversal"):
        xlsum_dataset._safe_extract_archive(archive_path, tmp_path / "extract")


def test_xlsum_es_scores_generated_summary_rouge(monkeypatch) -> None:
    docs = [
        {
            "id": "row-1",
            "url": "https://example.com/row-1",
            "title": "Helicópteros mentales",
            "summary": "Pilotar un helicóptero con la mente ya es posible.",
            "text": "El sistema transforma los patrones eléctricos del pensamiento en movimientos reales.",
        }
    ]
    monkeypatch.setattr(xlsum_es_module, "load_xlsum_dataset", lambda *args, **kwargs: list(docs))

    class FakeSession:
        def generate(self, requests, *, batch_size=None):
            assert batch_size == 1
            assert len(requests) == 1
            assert requests[0].prompt == (
                "Texto: El sistema transforma los patrones eléctricos del pensamiento en movimientos reales.\n\n"
                "Resumen:"
            )
            assert requests[0].max_new_tokens == 128
            return [
                GenerationOutput(
                    prompt=requests[0].prompt,
                    text="Pilotar un helicóptero con la mente ya es posible.",
                )
            ]

    result = evalution.benchmarks.xlsum_es(max_rows=1, batch_size=4).evaluate(FakeSession())

    assert result.name == "xlsum_es"
    assert result.metrics == summary_rouge_scores(
        "Pilotar un helicóptero con la mente ya es posible.",
        "Pilotar un helicóptero con la mente ya es posible.",
    )
    assert result.metadata["dataset_path"] == "csebuetnlp/xlsum"
    assert result.metadata["dataset_name"] == "spanish"
    assert result.metadata["split"] == "test"
    assert result.metadata["scoring_mode"] == "generated_summary_rouge"
    assert result.metadata["primary_metric"] == "rougeLsum"
    assert result.samples[0].metadata["id"] == "row-1"
    assert result.samples[0].metadata["title"] == "Helicópteros mentales"


def test_xlsum_es_prompt_formats_article() -> None:
    assert xlsum_es_module._xlsum_es_prompt("  Texto con espacios.  ") == (
        "Texto: Texto con espacios.\n\nResumen:"
    )
