# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput
from evalution.scorers.bleu import smoothed_corpus_bleu_4

code_x_glue_module = importlib.import_module("evalution.benchmarks.code_x_glue")


class FakeSession:
    def generate(self, requests, *, batch_size=None):
        assert batch_size == 2
        assert len(requests) == 2
        assert requests[0].prompt == "func add ( a int , b int ) int { return a + b }"
        assert requests[1].prompt == "func sub ( a int , b int ) int { return a - b }"
        assert requests[0].stop == ["</s>"]
        assert requests[0].max_new_tokens == 128
        assert requests[0].num_beams == 10
        return [
            GenerationOutput(
                prompt=requests[0].prompt,
                text="returns the sum of two integers",
            ),
            GenerationOutput(
                prompt=requests[1].prompt,
                text="subtracts two numbers with overflow checks",
            ),
        ]


def test_code_x_glue_scores_corpus_bleu(monkeypatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": 0,
                "repo": "demo/repo",
                "path": "math.go",
                "func_name": "add",
                "original_string": "",
                "language": "go",
                "code": "",
                "code_tokens": ["func", "add", "(", "a", "int", ",", "b", "int", ")", "int", "{", "return", "a", "+", "b", "}"],
                "docstring": "// returns the sum of two integers",
                "docstring_tokens": ["returns", "the", "sum", "of", "two", "integers"],
                "sha": "sha-add",
                "url": "https://example.com/add",
            },
            {
                "id": 1,
                "repo": "demo/repo",
                "path": "math.go",
                "func_name": "sub",
                "original_string": "",
                "language": "go",
                "code": "",
                "code_tokens": ["func", "sub", "(", "a", "int", ",", "b", "int", ")", "int", "{", "return", "a", "-", "b", "}"],
                "docstring": "// returns the difference of two integers",
                "docstring_tokens": ["returns", "the", "difference", "of", "two", "integers"],
                "sha": "sha-sub",
                "url": "https://example.com/sub",
            },
        ]
    )
    monkeypatch.setattr(code_x_glue_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = evalution.benchmarks.code2text_go(max_rows=2, batch_size=2).evaluate(FakeSession())

    assert result.name == "code2text_go"
    assert result.metrics == {
        "bleu4": smoothed_corpus_bleu_4(
            [
                "returns the sum of two integers",
                "returns the difference of two integers",
            ],
            [
                "returns the sum of two integers",
                "subtracts two numbers with overflow checks",
            ],
        )
    }
    assert result.metadata == {
        "dataset_path": "CM/codexglue_code2text_go",
        "dataset_name": None,
        "split": "test",
        "stream": True,
        "generation_submission_mode": "fixed_batches",
        "scoring_mode": "generated_docstring_corpus_bleu4",
        "primary_metric": "bleu4",
        "language": "go",
        "num_beams": 10,
    }

    first_sample = result.samples[0]
    assert first_sample.target == "returns the sum of two integers"
    assert first_sample.prediction == "returns the sum of two integers"
    assert first_sample.extracted == {
        "prediction-stripped": "returns the sum of two integers",
        "reference-stripped": "returns the sum of two integers",
    }
    assert first_sample.scores == {}
    assert first_sample.metadata["repo"] == "demo/repo"
    assert first_sample.metadata["path"] == "math.go"
    assert first_sample.metadata["func_name"] == "add"
    assert first_sample.metadata["language"] == "go"
    assert first_sample.metadata["code_token_count"] == 16
    assert first_sample.metadata["docstring_token_count"] == 6


def test_code_x_glue_normalizes_code_and_docstring_tokens() -> None:
    assert code_x_glue_module._normalized_code_text(
        ["func", "add", "(", "a", ",", "b", ")", "{", "\n", "return", "a", "+", "b", "}"]
    ) == "func add ( a , b ) { return a + b }"
    assert code_x_glue_module._normalized_docstring_text(
        ["returns", "the", "sum", "\n", "of", "two", "integers"]
    ) == "returns the sum of two integers"


def test_smoothed_corpus_bleu_4_matches_perfect_prediction() -> None:
    assert smoothed_corpus_bleu_4(
        ["returns the sum of two integers"],
        ["returns the sum of two integers"],
    ) == 100.0
