# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
click_module = importlib.import_module("evalution.benchmarks.click")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 4
        assert len(requests) == 4
        assert requests[0].context == (
            "주어진 질문을 천천히 읽고, 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n"
            "질문: 언어 문제\n"
            "보기:\n"
            "A:첫째, B: 둘째, C: 셋째, D: 넷째\n"
            "정답:"
        )
        assert [request.continuation for request in requests] == [" A", " B", " C", " D"]
        return [
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
        ]


def test_click_lang_text_filters_and_scores_label_choices(monkeypatch) -> None:
    """Verify click lang text filters and scores label choices. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "id": "TK_5",
                "paragraph": "",
                "question": "언어 문제",
                "choices": ["첫째", "둘째", "셋째", "넷째"],
                "answer": "첫째",
            },
            {
                "id": "KIIP_economy_1",
                "paragraph": "",
                "question": "문화 문제",
                "choices": ["가", "나", "다", "라"],
                "answer": "가",
            },
        ]
    )
    monkeypatch.setattr(click_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.click_lang_text(max_rows=4, batch_size=4).evaluate(FakeSession())

    assert result.name == "click_lang_text"
    assert result.metrics == {"acc,ll": 1.0, "acc,ll_avg": 1.0}
    assert result.metadata["dataset_path"] == "EunsuKim/CLIcK"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "train"
    assert result.metadata["subset"] == "click_lang_text"
    assert len(result.samples) == 1
    sample = result.samples[0]
    assert sample.target == "A"
    assert sample.prediction == "A"
    assert sample.metadata["id"] == "TK_5"
    assert sample.metadata["choice_labels"] == ["A", "B", "C", "D"]
    assert sample.metadata["raw_choices"] == ["첫째", "둘째", "셋째", "넷째"]


def test_click_build_sample_supports_five_choice_csat_items() -> None:
    """Verify click build sample supports five choice csat items."""
    suite = evalution.benchmarks.click_lang_text()
    sample = suite.build_sample(
        {
            "id": "CSAT_korean_22_5",
            "paragraph": "지문",
            "question": "정답은?",
            "choices": ["가", "나", "다", "라", "마"],
            "answer": "마",
        },
        index=0,
    )

    assert sample.prompt.startswith("주어진 맥락을 천천히 읽고")
    assert sample.prompt.endswith("\n정답:")
    assert sample.choices == ["A", "B", "C", "D", "E"]
    assert sample.gold_index == 4


def test_click_prompt_matches_upstream_shape() -> None:
    """Verify click prompt matches upstream shape."""
    assert click_module._click_prompt(
        {
            "paragraph": "배경",
            "question": "무엇인가?",
            "choices": ["가", "나", "다", "라"],
        }
    ) == (
        "주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n"
        "맥락: 배경\n"
        "질문: 무엇인가?\n"
        "보기:\n"
        "A:가, B: 나, C: 다, D: 라\n"
        "정답:"
    )
