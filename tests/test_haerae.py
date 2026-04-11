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
haerae_module = importlib.import_module("evalution.benchmarks.haerae")


def _haerae_row(*, query: str, answer: str, options: list[str]) -> dict[str, object]:
    """Support the surrounding tests with haerae row."""
    return {
        "query": query,
        "options": str(options),
        "answer": answer,
    }


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self, expected_text: str):
        """Initialize this object."""
        self.expected_text = expected_text

    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 2
        assert len(requests) == 5
        assert requests[0].context == self.expected_text
        assert [request.continuation for request in requests] == [
            " (A)",
            " (B)",
            " (C)",
            " (D)",
            " (E)",
        ]
        return [
            LoglikelihoodOutput(logprob=-1.1, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-0.1, is_greedy=True, token_count=1),
            LoglikelihoodOutput(logprob=-1.2, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.3, is_greedy=False, token_count=1),
            LoglikelihoodOutput(logprob=-1.4, is_greedy=False, token_count=1),
        ]


def test_haerae_general_knowledge_scores_label_choices(monkeypatch) -> None:
    """Verify haerae general knowledge scores label choices. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    suite = evalution.benchmarks.haerae_general_knowledge(batch_size=2, max_rows=1, stream=False)
    expected_prompt = (
        "다음 질문을 읽고 정답으로 가장 알맞은 것을 고르시요.\n\n"
        "### 질문: \n언택트의 순화어 알맞은 것은?\n\n"
        "### 선택지: \n(A) 비전속\n(B) 부대기\n(C) 기대주\n(D) 상대방\n(E) 비대면\n\n"
        "### 정답:"
    )
    dataset = Dataset.from_list(
        [
            _haerae_row(
                query=expected_prompt,
                answer="(B)",
                options=["비전속", "부대기", "기대주", "상대방", "비대면"],
            )
        ]
    )
    monkeypatch.setattr(haerae_module, "load_dataset", lambda *args, **kwargs: dataset)
    session = FakeSession(expected_text=expected_prompt)
    result = suite.evaluate(session)
    assert result.name == "haerae_general_knowledge"
    assert result.metrics["acc,ll"] == 1.0
    assert result.metrics["acc,ll_avg"] == 1.0
    sample = result.samples[0]
    assert sample.metadata["subset"] == "general_knowledge"
    assert sample.metadata["dataset_name"] == "general_knowledge"
    assert sample.metadata["answer"] == "(B)"
    assert sample.metadata["raw_choices"] == ["비전속", "부대기", "기대주", "상대방", "비대면"]


def test_haerae_group_loader_round_robins_subsets(monkeypatch) -> None:
    """Verify haerae group loader round robins subsets."""
    datasets = {
        "general_knowledge": Dataset.from_list([_haerae_row(query="gk1", answer="(A)", options=["a", "b", "c", "d", "e"])]),
        "history": Dataset.from_list([
            _haerae_row(query="hi1", answer="(A)", options=["a", "b", "c", "d", "e"]),
            _haerae_row(query="hi2", answer="(A)", options=["a", "b", "c", "d", "e"]),
        ]),
        "loan_words": Dataset.from_list([_haerae_row(query="lw1", answer="(A)", options=["a", "b", "c", "d", "e"])]),
        "rare_words": Dataset.from_list([_haerae_row(query="rw1", answer="(A)", options=["a", "b", "c", "d", "e"])]),
        "standard_nomenclature": Dataset.from_list([_haerae_row(query="sn1", answer="(A)", options=["a", "b", "c", "d", "e"])]),
    }

    def fake_load_dataset(path, name, split, cache_dir=None):
        """Support the surrounding tests with fake load dataset."""
        assert path == "HAERAE-HUB/HAE_RAE_BENCH"
        assert split == "test"
        return datasets[name]

    monkeypatch.setattr(haerae_module, "load_dataset", fake_load_dataset)
    suite = evalution.benchmarks.haerae(max_rows=6, stream=False)
    loaded = suite.dataset_loader()(suite.dataset_path, suite.dataset_name, split=suite.split)
    assert loaded[0]["query"] == "gk1"
    assert loaded[1]["query"] == "hi1"
    assert loaded[2]["query"] == "lw1"
    assert loaded[3]["query"] == "rw1"
    assert loaded[4]["query"] == "sn1"
    assert loaded[5]["query"] == "hi2"
