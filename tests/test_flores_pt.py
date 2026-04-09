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
import sacrebleu

import evalution
from evalution.datasets import flores200 as flores200_dataset
from evalution.engines.base import GenerationOutput

flores_pt_module = importlib.import_module("evalution.benchmarks.flores_pt")


def _build_flores_archive(path: Path) -> str:
    metadata_devtest = (
        "URL\tdomain\ttopic\thas_image\thas_hyperlink\n"
        "https://example.com/1\tnews\tmedicine\tno\tyes\n"
        "https://example.com/2\tnews\ttravel\tyes\tno\n"
    )
    metadata_dev = (
        "URL\tdomain\ttopic\thas_image\thas_hyperlink\n"
        "https://example.com/dev\tnews\tmedicine\tno\tno\n"
    )
    eng_devtest = (
        "The team reported a careful improvement in outcomes after the pilot study.\n"
        "Researchers expect the final report to include a broader international sample.\n"
    )
    por_devtest = (
        "A equipe relatou uma melhora cuidadosa nos resultados após o estudo piloto.\n"
        "Os pesquisadores esperam que o relatório final inclua uma amostra internacional mais ampla.\n"
    )
    eng_dev = "The pilot report remains preliminary.\n"
    por_dev = "O relatório piloto continua preliminar.\n"

    with tarfile.open(path, mode="w:gz") as archive:
        for member_name, contents in (
            ("./flores200_dataset/metadata_devtest.tsv", metadata_devtest),
            ("./flores200_dataset/metadata_dev.tsv", metadata_dev),
            ("./flores200_dataset/devtest/eng_Latn.devtest", eng_devtest),
            ("./flores200_dataset/devtest/por_Latn.devtest", por_devtest),
            ("./flores200_dataset/dev/eng_Latn.dev", eng_dev),
            ("./flores200_dataset/dev/por_Latn.dev", por_dev),
        ):
            encoded = contents.encode("utf-8")
            info = tarfile.TarInfo(name=member_name)
            info.size = len(encoded)
            archive.addfile(info, BytesIO(encoded))

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_flores200_pair_reads_vendored_archive(monkeypatch, tmp_path: Path) -> None:
    archive_path = tmp_path / "flores200_dataset.tar.gz"
    archive_sha256 = _build_flores_archive(archive_path)
    monkeypatch.setattr(flores200_dataset, "FLORES200_ARCHIVE_URL", archive_path.as_uri())
    monkeypatch.setattr(flores200_dataset, "FLORES200_ARCHIVE_SHA256", archive_sha256)
    flores200_dataset._load_pair_docs_cached.cache_clear()

    docs = flores200_dataset.load_flores200_pair(
        "facebook/flores",
        "all",
        split="test",
        cache_dir=str(tmp_path / "cache"),
        source_language="eng_Latn",
        target_language="por_Latn",
    )

    assert len(docs) == 2
    assert docs[0]["id"] == 1
    assert docs[0]["domain"] == "news"
    assert docs[0]["has_image"] is False
    assert docs[0]["has_hyperlink"] is True
    assert docs[0]["sentence_eng_Latn"].startswith("The team reported")
    assert docs[0]["sentence_por_Latn"].startswith("A equipe relatou")


def test_safe_extract_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        encoded = b"owned\n"
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = len(encoded)
        archive.addfile(info, BytesIO(encoded))

    with pytest.raises(ValueError, match="unsafe path traversal"):
        flores200_dataset._safe_extract_archive(archive_path, tmp_path / "extract")


def test_flores_pt_scores_translation_metrics(monkeypatch) -> None:
    docs = [
        {
            "id": 1,
            "URL": "https://example.com/1",
            "domain": "news",
            "topic": "medicine",
            "has_image": False,
            "has_hyperlink": True,
            "sentence_eng_Latn": "The team reported a careful improvement in outcomes after the pilot study.",
            "sentence_por_Latn": "A equipe relatou uma melhora cuidadosa nos resultados após o estudo piloto.",
        },
        {
            "id": 2,
            "URL": "https://example.com/2",
            "domain": "news",
            "topic": "travel",
            "has_image": True,
            "has_hyperlink": False,
            "sentence_eng_Latn": "Researchers expect the final report to include a broader international sample.",
            "sentence_por_Latn": "Os pesquisadores esperam que o relatório final inclua uma amostra internacional mais ampla.",
        },
    ]
    monkeypatch.setattr(flores_pt_module, "load_flores200_pair", lambda *args, **kwargs: list(docs))

    class FakeSession:
        def generate(self, requests, *, batch_size=None):
            assert batch_size == 2
            assert len(requests) == 2
            assert requests[0].prompt == (
                "English sentence: The team reported a careful improvement in outcomes after the pilot study.\n"
                "Portuguese sentence:"
            )
            assert requests[0].stop == ["\n"]
            return [
                GenerationOutput(
                    prompt=requests[0].prompt,
                    text="A equipe relatou uma melhora cuidadosa nos resultados após o estudo piloto.",
                ),
                GenerationOutput(
                    prompt=requests[1].prompt,
                    text="Os pesquisadores esperam que o relatório final apresente uma amostra global menor.",
                ),
            ]

    result = evalution.benchmarks.flores_pt(direction="en-pt", max_rows=2, batch_size=2).evaluate(FakeSession())

    references = [doc["sentence_por_Latn"] for doc in docs]
    predictions = [
        "A equipe relatou uma melhora cuidadosa nos resultados após o estudo piloto.",
        "Os pesquisadores esperam que o relatório final apresente uma amostra global menor.",
    ]
    assert result.name == "flores_pt_en_pt"
    assert result.metrics == {
        "bleu": pytest.approx(sacrebleu.corpus_bleu(predictions, [references]).score),
        "chrf": pytest.approx(sacrebleu.corpus_chrf(predictions, [references]).score),
        "ter": pytest.approx(sacrebleu.corpus_ter(predictions, [references]).score),
    }
    assert result.metadata["dataset_path"] == "facebook/flores"
    assert result.metadata["dataset_name"] == "all"
    assert result.metadata["direction"] == "en-pt"
    assert result.metadata["upstream_task"] == "portuguese_bench_flores_en-pt"
    assert result.samples[0].metadata["source_language"] == "en"
    assert result.samples[0].metadata["target_language"] == "pt"
    assert result.samples[0].target.startswith("A equipe relatou")


def test_flores_pt_dispatcher_and_validation() -> None:
    suite = evalution.benchmarks.flores_pt(direction="pt-en", max_rows=1)
    alias_suite = evalution.benchmarks.flores_pt_pt_en(max_rows=1)

    assert suite.task_name() == "flores_pt_pt_en"
    assert suite.direction == "pt-en"
    assert alias_suite.task_name() == "flores_pt_pt_en"
    assert alias_suite.direction == "pt-en"

    with pytest.raises(ValueError, match="unsupported flores_pt direction"):
        evalution.benchmarks.flores_pt(direction="ru-pt")
