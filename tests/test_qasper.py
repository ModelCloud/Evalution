# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
import io
import json
import tarfile

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput, LoglikelihoodOutput

qasper_module = importlib.import_module("evalution.benchmarks.qasper")


class BoolSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 2
        assert requests[0].context == "TITLE: Paper\nABSTRACT: Summary\n\nQ: Is the answer yes?\n\nA:"
        assert [request.continuation for request in requests] == [" no", " yes"]
        return [
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
        ]


class FreeformSession:
    def generate_continuous(self, requests, *, batch_size=None):
        assert batch_size == 4
        request_items = list(requests)
        assert len(request_items) == 1
        assert request_items[0][1].prompt == "TITLE: Paper\nABSTRACT: Summary\n\nQ: What is the answer?\n\nA:"
        for item_id, request in request_items:
            yield item_id, GenerationOutput(prompt=request.prompt, text="A concise answer")


def test_qasper_bool_scores_boolean_rows(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "title": "Paper",
                "abstract": "Summary",
                "question": "Is the answer yes?",
                "answer": "yes",
            }
        ]
    )
    monkeypatch.setattr(qasper_module, "_load_qasper_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.qasper_bool(max_rows=1, batch_size=8).evaluate(BoolSession())

    assert result.name == "qasper_bool"
    assert result.metrics == {
        "acc,ll": 1.0,
        "acc,ll_avg": 1.0,
        "f1,ll_boolean": 1.0,
        "f1,ll_avg_boolean": 1.0,
    }
    assert result.metadata["dataset_path"] == "allenai/qasper"
    assert result.metadata["split"] == "validation"
    assert result.metadata["variant"] == "bool"
    assert result.samples[0].metadata["answer_type"] == "bool"


def test_qasper_freeform_scores_abstractive_rows(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "title": "Paper",
                "abstract": "Summary",
                "question": "What is the answer?",
                "answer": "A concise answer",
            }
        ]
    )
    monkeypatch.setattr(qasper_module, "_load_qasper_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.qasper_freeform(max_rows=1, batch_size=4).evaluate(FreeformSession())

    assert result.name == "qasper_freeform"
    assert result.metrics == {"f1": 1.0}
    assert result.metadata["variant"] == "freeform"
    assert result.metadata["scoring_mode"] == "generated_qasper_abstractive_f1"
    assert result.samples[0].metadata["answer_type"] == "free form answer"


def test_qasper_loads_public_tarballs_without_dataset_scripts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    archive_path = tmp_path / "qasper-train-dev-v0.3.tgz"
    payload = {
        "paper-1": {
            "title": "Paper",
            "abstract": "Summary",
            "qas": [
                {
                    "question": "Is the answer yes?",
                    "answers": [
                        {
                            "annotation_id": "ann-1",
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": [],
                                "yes_no": True,
                                "free_form_answer": "",
                                "evidence": [],
                                "highlighted_evidence": [],
                            },
                            "worker_id": "worker-1",
                        }
                    ],
                }
            ],
        }
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("qasper-dev-v0.3.json")
        info.size = len(data_bytes)
        archive.addfile(info, io.BytesIO(data_bytes))
    monkeypatch.setattr(qasper_module, "_download_qasper_archive", lambda split, cache_dir: archive_path)

    dataset = qasper_module._load_qasper_dataset(
        "allenai/qasper",
        split="validation",
        answer_type="bool",
    )

    assert list(dataset) == [
        {
            "title": "Paper",
            "abstract": "Summary",
            "question": "Is the answer yes?",
            "answer": "yes",
            "answer_type": "bool",
        }
    ]


def test_qasper_flatten_supports_legacy_script_style_rows() -> None:
    dataset = Dataset.from_list(
        [
            {
                "title": "Paper",
                "abstract": "Summary",
                "qas": {
                    "question": ["What is the answer?"],
                    "answers": [
                        {
                            "answer": [
                                {
                                    "unanswerable": False,
                                    "extractive_spans": [],
                                    "yes_no": None,
                                    "free_form_answer": "A concise answer",
                                    "evidence": [],
                                    "highlighted_evidence": [],
                                }
                            ]
                        }
                    ],
                },
            }
        ]
    )

    flattened = qasper_module._flatten_qasper_dataset(dataset, answer_type="free form answer")

    assert list(flattened) == [
        {
            "title": "Paper",
            "abstract": "Summary",
            "question": "What is the answer?",
            "answer": "A concise answer",
            "answer_type": "free form answer",
        }
    ]


def test_qasper_dispatcher_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="unsupported qasper variant"):
        evalution.benchmarks.qasper(variant="extractive")
