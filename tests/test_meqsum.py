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
from evalution.datasets import meqsum as meqsum_dataset
from evalution.engines.base import GenerationOutput

meqsum_module = importlib.import_module("evalution.benchmarks.meqsum")


def _build_minimal_meqsum_xlsx(path: Path) -> str:
    files = {
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>
""",
        "_rels/.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
""",
        "xl/workbook.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="QS" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
""",
        "xl/_rels/workbook.xml.rels": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>
""",
        "xl/sharedStrings.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="9" uniqueCount="9">
  <si><t>File</t></si>
  <si><t>CHQ</t></si>
  <si><t>Summary</t></si>
  <si><t>row-1.txt</t></si>
  <si><t>SUBJECT: aspirin
MESSAGE: I need to know who manufactures aspirin.</t></si>
  <si><t>Who manufactures aspirin?</t></si>
  <si><t>row-2.txt</t></si>
  <si><t>Where can I get tested for William's syndrome?</t></si>
  <si><t>Where can I get tested for William's syndrome?</t></si>
</sst>
""",
        "xl/worksheets/sheet1.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1">
      <c r="A1" t="s"><v>0</v></c>
      <c r="B1" t="s"><v>1</v></c>
      <c r="C1" t="s"><v>2</v></c>
    </row>
    <row r="2">
      <c r="A2" t="s"><v>3</v></c>
      <c r="B2" t="s"><v>4</v></c>
      <c r="C2" t="s"><v>5</v></c>
    </row>
    <row r="3">
      <c r="A3" t="s"><v>6</v></c>
      <c r="B3" t="s"><v>7</v></c>
      <c r="C3" t="s"><v>8</v></c>
    </row>
  </sheetData>
</worksheet>
""",
    }
    with zipfile.ZipFile(path, mode="w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_meqsum_dataset_reads_pinned_xlsx(monkeypatch, tmp_path: Path) -> None:
    xlsx_path = tmp_path / "meqsum.xlsx"
    xlsx_sha256 = _build_minimal_meqsum_xlsx(xlsx_path)
    monkeypatch.setattr(meqsum_dataset, "MEQSUM_SOURCE_SHA256", xlsx_sha256)
    meqsum_dataset._load_rows_cached.cache_clear()
    monkeypatch.setattr(meqsum_dataset, "_download_source_xlsx", lambda cache_dir: xlsx_path)

    docs = meqsum_dataset.load_meqsum_dataset(
        "bigbio/meqsum",
        "meqsum_source",
        split="test",
        cache_dir=str(tmp_path / "cache"),
    )

    assert docs == [
        {
            "File": "row-1.txt",
            "CHQ": "SUBJECT: aspirin\nMESSAGE: I need to know who manufactures aspirin.",
            "Summary": "Who manufactures aspirin?",
        },
        {
            "File": "row-2.txt",
            "CHQ": "Where can I get tested for William's syndrome?",
            "Summary": "Where can I get tested for William's syndrome?",
        },
    ]


def test_load_meqsum_dataset_rejects_checksum_mismatch(monkeypatch, tmp_path: Path) -> None:
    xlsx_path = tmp_path / "meqsum.xlsx"
    _build_minimal_meqsum_xlsx(xlsx_path)
    monkeypatch.setattr(meqsum_dataset, "MEQSUM_SOURCE_SHA256", "0" * 64)
    meqsum_dataset._load_rows_cached.cache_clear()
    monkeypatch.setattr(meqsum_dataset, "_download_source_xlsx", lambda cache_dir: xlsx_path)

    with pytest.raises(ValueError, match="checksum mismatch"):
        meqsum_dataset.load_meqsum_dataset("bigbio/meqsum", "meqsum_source", split="train")


def test_meqsum_scores_medical_question_summary(monkeypatch) -> None:
    docs = [
        {
            "File": "row-1.txt",
            "CHQ": "SUBJECT: aspirin\nMESSAGE: I need to know who manufactures aspirin.",
            "Summary": "Who manufactures aspirin?",
        }
    ]
    monkeypatch.setattr(meqsum_module, "load_meqsum_dataset", lambda *args, **kwargs: list(docs))

    class FakeSession:
        def generate(self, requests, *, batch_size=None):
            assert batch_size == 1
            assert len(requests) == 1
            assert requests[0].prompt == (
                "Instructions: The following text is contains a medical question. "
                "Extract and summarize the question.\n\n"
                "I need to know who manufactures aspirin."
            )
            assert requests[0].stop == ["\n\n"]
            return [
                GenerationOutput(
                    prompt=requests[0].prompt,
                    text="Who manufactures aspirin?",
                )
            ]

    result = evalution.benchmarks.meqsum(max_rows=1, batch_size=4).evaluate(FakeSession())

    assert result.name == "meqsum"
    assert result.metrics == {
        "bleu": pytest.approx(1.0),
        "rouge1": pytest.approx(1.0),
        "rouge2": pytest.approx(1.0),
        "rougeL": pytest.approx(1.0),
    }
    assert result.metadata["dataset_path"] == "bigbio/meqsum"
    assert result.metadata["dataset_name"] == "meqsum_source"
    assert result.metadata["primary_metric"] == "rouge1"
    assert result.metadata["omitted_upstream_metrics"] == ["bert_score", "bleurt"]
    assert result.samples[0].metadata["file"] == "row-1.txt"


def test_meqsum_question_text_prefers_message_body() -> None:
    assert meqsum_module._meqsum_question_text("SUBJECT: aspirin\nMESSAGE: Hello there") == "Hello there"
    assert meqsum_module._meqsum_question_text("Standalone question") == "Standalone question"
