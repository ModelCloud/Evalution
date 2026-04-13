# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset
import pcre

import evalution
from evalution.engines.base import GenerationOutput
from evalution.scorers.gsm8k import INVALID_ANSWER
from evalution.scorers.gsm8k import extract_format_insensitive_numeric_answer
from evalution.scorers.gsm8k import extract_gsm8k_reference_answer
from evalution.benchmarks import base as base_suite_module

# Keep shared test fixtures and expectations explicit at module scope.
gsm8k_module = importlib.import_module("evalution.benchmarks.gsm8k")

_REFERENCE_ANS_RE = pcre.compile(r"#### (\-?[0-9\.\,]+)")


def _official_gsm8k_extract_answer(completion: str) -> str:
    """Support the surrounding tests with official GSM8K extract answer."""
    match = _REFERENCE_ANS_RE.search(completion)
    if match is None:
        return INVALID_ANSWER
    return match.group(1).strip().replace(",", "")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self, responses: list[str]) -> None:
        """Initialize this object."""
        self.responses = responses
        self.requests = []

    def generate(self, requests, *, batch_size=None):
        """Generate generate."""
        del batch_size
        self.requests.extend(requests)
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]

    def close(self) -> None:
        """Release the resources owned by this object."""
        return None


def test_gsm8k_reference_parser_matches_openai_release_cases() -> None:
    """Verify GSM8K reference parser matches openai release cases."""
    cases = [
        ("Work...\n#### 1,234", "1234"),
        ("Reasoning first\n#### -7.5", "-7.5"),
        ("The answer is 42.", INVALID_ANSWER),
        ("No final marker here", INVALID_ANSWER),
    ]

    for completion, expected in cases:
        assert _official_gsm8k_extract_answer(completion) == expected
        assert extract_gsm8k_reference_answer(completion) == expected


def test_gsm8k_numeric_parser_is_format_insensitive() -> None:
    """Verify GSM8K numeric parser is format insensitive."""
    assert extract_format_insensitive_numeric_answer("The answer is 42.") == "42"
    assert extract_format_insensitive_numeric_answer("Answer: 1,234") == "1234"
    assert extract_format_insensitive_numeric_answer("Reasoning...\n\\boxed{18}") == "18"
    assert extract_format_insensitive_numeric_answer("No number at all") == INVALID_ANSWER


def test_gsm8k_suite_scores_numeric_primary(monkeypatch) -> None:
    """Verify GSM8K suite scores numeric primary. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
            }
        ]
    )
    monkeypatch.setattr(gsm8k_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k(max_rows=1)
    session = FakeSession(["The answer is 42."])
    result = suite.evaluate(session)

    assert result.name == "gsm8k_cot"
    assert set(result.metrics) == {"acc,num"}
    assert result.metrics["acc,num"] == 1.0
    assert result.metadata["dataset_path"] == "openai/gsm8k"
    assert result.metadata["variant"] == "cot"
    assert result.metadata["scoring_mode"] == "numeric_format_insensitive"
    assert result.samples[0].extracted["numeric-extract"] == "42"
    assert set(result.samples[0].extracted) == {"numeric-extract"}


def test_gsm8k_suite_scores_hash_formatted_answers(monkeypatch) -> None:
    """Verify GSM8K suite scores hash formatted answers. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
            }
        ]
    )
    monkeypatch.setattr(gsm8k_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k(max_rows=1)
    session = FakeSession(["Reasoning first\n#### 42"])
    result = suite.evaluate(session)

    assert result.metrics["acc,num"] == 1.0


def test_gsm8k_base_variant_omits_platinum_specific_cleaning_metadata(monkeypatch) -> None:
    """Verify GSM8K base variant omits platinum specific cleaning metadata."""
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
            }
        ]
    )
    monkeypatch.setattr(gsm8k_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = evalution.benchmarks.gsm8k(
        variant="base",
        max_rows=1,
    )
    session = FakeSession(["42"])
    result = suite.evaluate(session)

    assert result.name == "gsm8k"
    assert result.samples[0].metadata == {}


def test_gsm8k_logs_executed_rows_and_warns_on_short_generation(monkeypatch) -> None:
    """Verify GSM8K logs executed rows and warns on short generation."""
    dataset = Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
            },
            {
                "question": "What is 40 plus 3?",
                "answer": "40 + 3 = 43\n#### 43",
            },
        ]
    )
    monkeypatch.setattr(gsm8k_module, "load_dataset", lambda *args, **kwargs: dataset)

    class FakeLogger:
        """Provide the fake logger helper used by the surrounding tests."""
        def __init__(self) -> None:
            """Initialize this object."""
            self.info_messages: list[str] = []
            self.warning_messages: list[str] = []

        def info(self, message: str, *args) -> None:
            """Implement info for fake logger."""
            self.info_messages.append(message % args if args else message)

        def warning(self, message: str, *args) -> None:
            """Implement warning for fake logger."""
            self.warning_messages.append(message % args if args else message)

    class ShortOutputSession:
        """Define the short output session helper used by the surrounding tests."""
        def generate(self, requests, *, batch_size=None):
            """Generate generate."""
            del batch_size
            request = requests[0]
            return [
                GenerationOutput(
                    prompt=request.prompt if request.prompt is not None else str(request.messages),
                    text="The answer is 42.",
                )
            ]

        def close(self) -> None:
            """Release the resources owned by this object."""
            return None

    fake_logger = FakeLogger()
    monkeypatch.setattr(base_suite_module, "get_logger", lambda: fake_logger)

    suite = evalution.benchmarks.gsm8k(max_rows=2, batch_size=2)
    result = suite.evaluate(ShortOutputSession())

    assert len(result.samples) == 1
    assert any("gsm8k_cot: executed 1/2 sample(s)" in message for message in fake_logger.info_messages)
    assert any(
        "gsm8k_cot: only executed 1/2 sample(s); generation returned fewer outputs than expected"
        in message
        for message in fake_logger.warning_messages
    )
