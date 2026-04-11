# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import zipfile

import pytest

import evalution
from evalution.datasets import mediqa_qa as mediqa_qa_dataset
from evalution.engines.base import GenerationOutput

# Keep shared test fixtures and expectations explicit at module scope.
mediqa_qa2019_module = importlib.import_module("evalution.benchmarks.mediqa_qa2019")


def _build_minimal_mediqa_archive(path: Path) -> str:
    """Build minimal mediqa archive."""
    xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<MEDIQA2019-Task3-QA-ValidationSet>
  <Question QID="2">
    <QuestionText>about uveitis. IS THE UVEITIS, AN AUTOIMMUNE DISEASE?</QuestionText>
    <AnswerList>
      <Answer AID="2_Answer1" SystemRank="1" ReferenceRank="3" ReferenceScore="4">
        <AnswerURL>https://example.com/uveitis-1</AnswerURL>
        <AnswerText>Uveitis can be associated with autoimmune disease.</AnswerText>
      </Answer>
      <Answer AID="2_Answer2" SystemRank="2" ReferenceRank="1" ReferenceScore="4">
        <AnswerURL>https://example.com/uveitis-2</AnswerURL>
        <AnswerText>Some cases of uveitis are autoimmune, but not all are.</AnswerText>
      </Answer>
    </AnswerList>
  </Question>
</MEDIQA2019-Task3-QA-ValidationSet>
"""
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr(
            "MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-ValidationSet.xml",
            xml_payload,
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_mediqa_qa_dataset_reads_pinned_zip(monkeypatch, tmp_path: Path) -> None:
    """Verify load mediqa QA dataset reads pinned zip."""
    archive_path = tmp_path / "mediqa.zip"
    archive_sha256 = _build_minimal_mediqa_archive(archive_path)
    monkeypatch.setattr(mediqa_qa_dataset, "MEDIQA_QA_SOURCE_SHA256", archive_sha256)
    monkeypatch.setattr(
        mediqa_qa_dataset,
        "MEDIQA_QA_SPLIT_FILES",
        {
            "validation": "MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-ValidationSet.xml",
        },
    )
    mediqa_qa_dataset._load_rows_cached.cache_clear()
    monkeypatch.setattr(mediqa_qa_dataset, "_download_source_zip", lambda cache_dir: archive_path)

    docs = mediqa_qa_dataset.load_mediqa_qa_dataset(
        "bigbio/mediqa_qa",
        "mediqa_qa_source",
        split="validation",
        cache_dir=str(tmp_path / "cache"),
    )

    assert docs == [
        {
            "QUESTION": {
                "QID": "2",
                "QuestionText": "about uveitis. IS THE UVEITIS, AN AUTOIMMUNE DISEASE?",
                "AnswerList": [
                    {
                        "Answer": {
                            "AID": "2_Answer1",
                            "SystemRank": 1,
                            "ReferenceRank": 3,
                            "ReferenceScore": 4,
                            "AnswerURL": "https://example.com/uveitis-1",
                            "AnswerText": "Uveitis can be associated with autoimmune disease.",
                        }
                    },
                    {
                        "Answer": {
                            "AID": "2_Answer2",
                            "SystemRank": 2,
                            "ReferenceRank": 1,
                            "ReferenceScore": 4,
                            "AnswerURL": "https://example.com/uveitis-2",
                            "AnswerText": "Some cases of uveitis are autoimmune, but not all are.",
                        }
                    },
                ],
            }
        }
    ]


def test_load_mediqa_qa_dataset_rejects_checksum_mismatch(monkeypatch, tmp_path: Path) -> None:
    """Verify load mediqa QA dataset rejects checksum mismatch."""
    archive_path = tmp_path / "mediqa.zip"
    _build_minimal_mediqa_archive(archive_path)
    monkeypatch.setattr(mediqa_qa_dataset, "MEDIQA_QA_SOURCE_SHA256", "0" * 64)
    monkeypatch.setattr(
        mediqa_qa_dataset,
        "MEDIQA_QA_SPLIT_FILES",
        {
            "validation": "MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-ValidationSet.xml",
        },
    )
    mediqa_qa_dataset._load_rows_cached.cache_clear()
    monkeypatch.setattr(mediqa_qa_dataset, "_download_source_zip", lambda cache_dir: archive_path)

    with pytest.raises(ValueError, match="checksum mismatch"):
        mediqa_qa_dataset.load_mediqa_qa_dataset("bigbio/mediqa_qa", "mediqa_qa_source", split="validation")


def test_mediqa_qa2019_scores_medical_answer_generation(monkeypatch) -> None:
    """Verify mediqa qa2019 scores medical answer generation. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    docs = [
        {
            "QUESTION": {
                "QID": "2",
                "QuestionText": "about uveitis. IS THE UVEITIS, AN AUTOIMMUNE DISEASE?",
                "AnswerList": [
                    {
                        "Answer": {
                            "AID": "2_Answer1",
                            "SystemRank": 1,
                            "ReferenceRank": 3,
                            "ReferenceScore": 4,
                            "AnswerURL": "https://example.com/uveitis-1",
                            "AnswerText": "Uveitis can be associated with autoimmune disease.",
                        }
                    }
                ],
            }
        }
    ]
    monkeypatch.setattr(
        mediqa_qa2019_module,
        "load_mediqa_qa_dataset",
        lambda *args, **kwargs: list(docs),
    )

    class FakeSession:
        """Provide the fake session helper used by the surrounding tests."""
        def generate(self, requests, *, batch_size=None):
            """Generate generate."""
            assert batch_size == 1
            assert len(requests) == 1
            assert requests[0].prompt == (
                "Instructions: The following text is a question asked by a patient. "
                "Answer how a doctor would, while trying to be as informative and helpful as possible.\n\n"
                "Question: about uveitis. IS THE UVEITIS, AN AUTOIMMUNE DISEASE?\n\n"
                "Answer:"
            )
            assert requests[0].stop == ["\n\n"]
            return [
                GenerationOutput(
                    prompt=requests[0].prompt,
                    text="Uveitis can be associated with autoimmune disease.",
                )
            ]

    result = evalution.benchmarks.mediqa_qa2019(max_rows=1, batch_size=4).evaluate(FakeSession())

    assert result.name == "mediqa_qa2019"
    assert result.metrics == {
        "bleu": pytest.approx(1.0),
        "rouge1": pytest.approx(1.0),
        "rouge2": pytest.approx(1.0),
        "rougeL": pytest.approx(1.0),
    }
    assert result.metadata["dataset_path"] == "bigbio/mediqa_qa"
    assert result.metadata["dataset_name"] == "mediqa_qa_source"
    assert result.metadata["primary_metric"] == "rouge1"
    assert result.metadata["omitted_upstream_metrics"] == ["bert_score", "bleurt"]
    assert result.samples[0].metadata == {
        "qid": "2",
        "answer_count": 1,
        "first_answer_aid": "2_Answer1",
        "first_answer_reference_rank": 3,
        "first_answer_reference_score": 4,
    }


def test_mediqa_qa2019_empty_text_scores_zero() -> None:
    """Verify mediqa qa2019 empty text scores zero. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    assert mediqa_qa2019_module._mediqa_qa2019_answer_scores("", "reference") == {
        "bleu": 0.0,
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
    }
    assert mediqa_qa2019_module._mediqa_qa2019_answer_scores("prediction", "") == {
        "bleu": 0.0,
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
    }
