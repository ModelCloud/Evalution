# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

kobest_module = importlib.import_module("evalution.benchmarks.kobest")


class FakeBoolQSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 2
        assert requests[0].context == (
            "지문: 구한말, 통영 안뒤산 기슭 간창골에 김봉제 형제가 살았다.\n"
            "질문: 봉룡은 숙정을 죽였는가?\n"
            "답변:"
        )
        assert [request.continuation for request in requests] == [" 아니오", " 예"]
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
        ]


class FakeCOPASession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 2
        assert requests[0].context == "전쟁이 시작되었다 그래서"
        assert [request.continuation for request in requests] == [
            " 병사들이 집으로 돌아왔다.",
            " 병사들이 전투에 파견되었다.",
        ]
        return [
            LoglikelihoodOutput(logprob=-1.8, is_greedy=False, token_count=4),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=4),
        ]


class FakeHellaSwagSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 4
        assert requests[0].context == (
            "모자를 쓴 투수가 타자에게 온 힘을 다해 공을 던진다. "
            "공이 타자에게 빠른 속도로 다가온다. 타자가 공을 배트로 친다."
        )
        assert [request.continuation for request in requests] == [
            " 외야수가 떨어지는 공을 글러브로 잡는다.",
            " 외야수가 공이 떨어질 위치에 자리를 잡는다.",
            " 심판이 아웃을 외친다.",
            " 외야수가 공을 따라 뛰기 시작한다.",
        ]
        return [
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=5),
            LoglikelihoodOutput(logprob=-1.6, is_greedy=False, token_count=6),
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=3),
            LoglikelihoodOutput(logprob=-0.3, is_greedy=True, token_count=5),
        ]


class FakeSentiNegSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 2
        assert requests[0].context == "문장: 택배사 정말 마음에 듬\n질문: 이 문장의 감성은 무엇입니까?\n답변:"
        assert [request.continuation for request in requests] == [" 부정", " 긍정"]
        return [
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.2, is_greedy=True, token_count=1),
        ]


class FakeWiCSession:
    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 8
        assert len(requests) == 2
        assert requests[0].context == (
            "문장 1: 백제는 의자왕의 방탕과 [실정]으로 패망했다고 보는 견해가 있다.\n"
            "문장 2: 지금 회사는 자금난으로 공장을 운영할 수 없는 [실정]이다.\n"
            "질문: 두 문장에서 '실정'의 의미가 같습니까?\n"
            "답변:"
        )
        assert [request.continuation for request in requests] == [" 아니오", " 예"]
        return [
            LoglikelihoodOutput(logprob=-0.3, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.0, is_greedy=False, token_count=1),
        ]


def test_kobest_boolq_scores_yes_no_reading_comprehension(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "paragraph": "구한말, 통영 안뒤산 기슭 간창골에 김봉제 형제가 살았다.",
                "question": "봉룡은 숙정을 죽였는가",
                "label": 0,
            }
        ]
    )
    monkeypatch.setattr(kobest_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.kobest_boolq(max_rows=1, batch_size=8).evaluate(FakeBoolQSession())

    assert result.name == "kobest_boolq"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "skt/kobest_v1"
    assert result.metadata["dataset_name"] == "boolq"
    assert result.metadata["split"] == "test"
    assert result.metadata["subset"] == "boolq"
    sample = result.samples[0]
    assert sample.target == "아니오"
    assert sample.prediction == "아니오"
    assert sample.metadata["question"] == "봉룡은 숙정을 죽였는가"


def test_kobest_copa_scores_korean_causal_choice(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "premise": "전쟁이 시작되었다.",
                "question": "결과",
                "alternative_1": "병사들이 집으로 돌아왔다.",
                "alternative_2": "병사들이 전투에 파견되었다.",
                "label": 1,
            }
        ]
    )
    monkeypatch.setattr(kobest_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.kobest_copa(max_rows=1, batch_size=8).evaluate(FakeCOPASession())

    assert result.name == "kobest_copa"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_name"] == "copa"
    sample = result.samples[0]
    assert sample.target == "병사들이 전투에 파견되었다."
    assert sample.prediction == "병사들이 전투에 파견되었다."
    assert sample.metadata["question"] == "결과"


def test_kobest_hellaswag_scores_four_way_story_completion(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "context": (
                    "모자를 쓴 투수가 타자에게 온 힘을 다해 공을 던진다. "
                    "공이 타자에게 빠른 속도로 다가온다. 타자가 공을 배트로 친다."
                ),
                "ending_1": "외야수가 떨어지는 공을 글러브로 잡는다.",
                "ending_2": "외야수가 공이 떨어질 위치에 자리를 잡는다.",
                "ending_3": "심판이 아웃을 외친다.",
                "ending_4": "외야수가 공을 따라 뛰기 시작한다.",
                "label": 3,
            }
        ]
    )
    monkeypatch.setattr(kobest_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.kobest_hellaswag(max_rows=1, batch_size=8).evaluate(FakeHellaSwagSession())

    assert result.name == "kobest_hellaswag"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    sample = result.samples[0]
    assert sample.target == "외야수가 공을 따라 뛰기 시작한다."
    assert sample.prediction == "외야수가 공을 따라 뛰기 시작한다."
    assert len(sample.metadata["raw_choices"]) == 4


def test_kobest_sentineg_scores_sentence_sentiment(monkeypatch) -> None:
    dataset = Dataset.from_list([{"sentence": "택배사 정말 마음에 듬", "label": 1}])
    monkeypatch.setattr(kobest_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.kobest_sentineg(max_rows=1, batch_size=8).evaluate(FakeSentiNegSession())

    assert result.name == "kobest_sentineg"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    sample = result.samples[0]
    assert sample.target == "긍정"
    assert sample.prediction == "긍정"
    assert sample.metadata["sentence"] == "택배사 정말 마음에 듬"


def test_kobest_wic_scores_word_sense_similarity(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "word": "실정",
                "context_1": "백제는 의자왕의 방탕과 [실정]으로 패망했다고 보는 견해가 있다.",
                "context_2": "지금 회사는 자금난으로 공장을 운영할 수 없는 [실정]이다.",
                "label": 0,
            }
        ]
    )
    monkeypatch.setattr(kobest_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.kobest_wic(max_rows=1, batch_size=8).evaluate(FakeWiCSession())

    assert result.name == "kobest_wic"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    sample = result.samples[0]
    assert sample.target == "아니오"
    assert sample.prediction == "아니오"
    assert sample.metadata["word"] == "실정"


def test_kobest_rejects_unknown_subset() -> None:
    with pytest.raises(ValueError, match="unsupported kobest subset"):
        evalution.benchmarks.kobest(subset="unknown_subset")


def test_kobest_rejects_dataset_name_mismatch() -> None:
    with pytest.raises(ValueError, match="dataset_name must match"):
        evalution.benchmarks.kobest(subset="boolq", dataset_name="wic")
