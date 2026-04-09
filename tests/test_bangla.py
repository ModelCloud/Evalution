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

bangla_module = importlib.import_module("evalution.benchmarks.bangla")


class FakeBoolQSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 2
        assert requests[0].context == (
            "Passage:\nবাংলাদেশের জাতীয় ফুল শাপলা।\n\nQuestion:\nবাংলাদেশের জাতীয় ফুল কি শাপলা?\n\nAnswer:"
        )
        assert [request.continuation for request in requests] == [" yes", " no"]
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=1),
        ]


class FakeCommonsenseQASession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 5
        assert requests[0].context == (
            "বৃষ্টি হলে মানুষ সাধারণত কী ব্যবহার করে?\n"
            "A. ছাতা\n"
            "B. চশমা\n"
            "C. হাতুড়ি\n"
            "D. জুতা\n"
            "E. চামচ\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D", " E"]
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
        ]


class FakeOpenBookQASession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 4
        assert requests[0].context == (
            "সূর্য থেকে আমরা কী পাই?\n"
            "A. আলো\n"
            "B. মাটি\n"
            "C. বৃষ্টি\n"
            "D. পাথর\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
        ]


class FakePIQASession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 2
        assert requests[0].context == (
            "ঠান্ডা পানি দ্রুত গরম করতে চাই\n"
            "A. পানিটা চুলায় গরম করো\n"
            "B. পানিটা ফ্রিজে রাখো\n"
            "Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B"]
        return [
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
        ]


class FakeMMLUSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 4
        assert requests[0].context == (
            "পৃথিবী সূর্যের চারদিকে কত সময়ে একবার ঘোরে? "
            "A. এক দিন B. এক মাস C. এক বছর D. এক ঘণ্টা Answer:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
        ]


def test_bangla_boolqa_scores_yes_no_reading_comprehension(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "passage": "বাংলাদেশের জাতীয় ফুল শাপলা।",
                "question": "বাংলাদেশের জাতীয় ফুল কি শাপলা?",
                "answer": True,
                "answer_bn": "হ্যাঁ",
            }
        ]
    )
    monkeypatch.setattr(bangla_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.bangla_boolqa(max_rows=1, batch_size=6).evaluate(FakeBoolQSession())

    assert result.name == "bangla_boolqa"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "hishab/boolq_bn"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "validation"
    assert result.metadata["subset"] == "boolqa"
    sample = result.samples[0]
    assert sample.target == "yes"
    assert sample.prediction == "yes"
    assert sample.metadata["passage"] == "বাংলাদেশের জাতীয় ফুল শাপলা।"


def test_bangla_commonsenseqa_scores_label_multiple_choice(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question_stem": "বৃষ্টি হলে মানুষ সাধারণত কী ব্যবহার করে?",
                "choices": {
                    "label": ["A", "B", "C", "D", "E"],
                    "text": ["ছাতা", "চশমা", "হাতুড়ি", "জুতা", "চামচ"],
                },
                "answerKey": "A",
            }
        ]
    )
    monkeypatch.setattr(bangla_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.bangla_commonsenseqa(max_rows=1, batch_size=6).evaluate(
        FakeCommonsenseQASession()
    )

    assert result.name == "bangla_commonsenseqa"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "hishab/commonsenseqa-bn"
    assert result.metadata["subset"] == "commonsenseqa"
    sample = result.samples[0]
    assert sample.target == "A"
    assert sample.prediction == "A"
    assert sample.metadata["raw_choices"] == ["ছাতা", "চশমা", "হাতুড়ি", "জুতা", "চামচ"]


def test_bangla_openbookqa_scores_label_multiple_choice(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question_stem": "সূর্য থেকে আমরা কী পাই?",
                "choices": {
                    "label": ["A", "B", "C", "D"],
                    "text": ["আলো", "মাটি", "বৃষ্টি", "পাথর"],
                },
                "answerKey": "A",
            }
        ]
    )
    monkeypatch.setattr(bangla_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.bangla_openbookqa(max_rows=1, batch_size=6).evaluate(
        FakeOpenBookQASession()
    )

    assert result.name == "bangla_openbookqa"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "hishab/openbookqa-bn"
    assert result.metadata["split"] == "test"
    assert result.metadata["subset"] == "openbookqa"
    sample = result.samples[0]
    assert sample.target == "A"
    assert sample.prediction == "A"


def test_bangla_piqa_scores_binary_label_multiple_choice(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "goal": "ঠান্ডা পানি দ্রুত গরম করতে চাই",
                "sol1": "পানিটা চুলায় গরম করো",
                "sol2": "পানিটা ফ্রিজে রাখো",
                "label": 0,
            }
        ]
    )
    monkeypatch.setattr(bangla_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.bangla_piqa(max_rows=1, batch_size=6).evaluate(FakePIQASession())

    assert result.name == "bangla_piqa"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "hishab/piqa-bn"
    assert result.metadata["subset"] == "piqa"
    sample = result.samples[0]
    assert sample.target == "A"
    assert sample.prediction == "A"
    assert sample.metadata["raw_choices"] == ["পানিটা চুলায় গরম করো", "পানিটা ফ্রিজে রাখো"]


def test_bangla_mmlu_scores_four_way_label_multiple_choice(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "question": "পৃথিবী সূর্যের চারদিকে কত সময়ে একবার ঘোরে?",
                "options": ["এক দিন", "এক মাস", "এক বছর", "এক ঘণ্টা"],
                "answer": "C",
                "subject": "astronomy",
            }
        ]
    )
    monkeypatch.setattr(bangla_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.bangla_mmlu(max_rows=1, batch_size=6).evaluate(FakeMMLUSession())

    assert result.name == "bangla_mmlu"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "hishab/titulm-bangla-mmlu"
    assert result.metadata["dataset_name"] == "all"
    assert result.metadata["split"] == "test"
    assert result.metadata["subset"] == "mmlu"
    sample = result.samples[0]
    assert sample.target == "C"
    assert sample.prediction == "C"
    assert sample.metadata["subject"] == "astronomy"


def test_bangla_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="unsupported bangla subset"):
        evalution.benchmarks.bangla(subset="unknown")


def test_bangla_rejects_dataset_name_mismatch() -> None:
    with pytest.raises(ValueError, match="dataset_name must match"):
        evalution.benchmarks.bangla(subset="mmlu", dataset_name="history")


def test_bangla_label_prompt_reorders_choice_texts() -> None:
    suite = evalution.benchmarks.bangla_commonsenseqa()
    sample = bangla_module.MultipleChoiceSample(
        index=0,
        prompt="ignored",
        choices=["A", "B", "C", "D", "E"],
        gold_index=0,
        metadata={
            "subset": "commonsenseqa",
            "question": "বৃষ্টি হলে মানুষ সাধারণত কী ব্যবহার করে?",
            "raw_choices": ["ছাতা", "চশমা", "হাতুড়ি", "জুতা", "চামচ"],
        },
    )

    prompt = suite.label_prompt(
        sample,
        choice_order=(1, 0, 2, 3, 4),
        labels=("A", "B", "C", "D", "E"),
    )

    assert prompt == (
        "বৃষ্টি হলে মানুষ সাধারণত কী ব্যবহার করে?\n"
        "A. চশমা\n"
        "B. ছাতা\n"
        "C. হাতুড়ি\n"
        "D. জুতা\n"
        "E. চামচ\n"
        "Answer:"
    )
