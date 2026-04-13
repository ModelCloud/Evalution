# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib
import math

import pytest
from datasets import Dataset

import evalution
from evalution.engines.base import LoglikelihoodOutput

# Keep shared test fixtures and expectations explicit at module scope.
lambada_module = importlib.import_module("evalution.benchmarks.lambada")


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(
        self,
        *,
        expected_prompt: str,
        expected_continuation: str,
        output: LoglikelihoodOutput,
    ) -> None:
        """Initialize this object."""
        self.expected_prompt = expected_prompt
        self.expected_continuation = expected_continuation
        self.output = output

    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        assert batch_size == 6
        assert len(requests) == 1
        assert requests[0].context == self.expected_prompt
        assert requests[0].continuation == self.expected_continuation
        return [self.output]


@pytest.mark.parametrize(
    ("factory_name", "dataset_name", "doc", "expected_prompt", "expected_target"),
    [
        (
            "lambada_openai_mt_de",
            "de",
            {"text": "Stattdessen starrte ich geradeaus, als wäre ich tief interessiert, was an der Vorderseite des Flugzeugs passierte"},
            "Stattdessen starrte ich geradeaus, als wäre ich tief interessiert, was an der Vorderseite des Flugzeugs",
            " passierte",
        ),
        (
            "lambada_openai_mt_en",
            "en",
            {"text": "In my palm is a clear stone, and inside it is a small ivory statuette guardia"},
            "In my palm is a clear stone, and inside it is a small ivory statuette",
            " guardia",
        ),
        (
            "lambada_openai_mt_es",
            "es",
            {"text": "Escuchó a Rhinna hablar y entonces Tom comprendió todo"},
            "Escuchó a Rhinna hablar y entonces Tom comprendió",
            " todo",
        ),
        (
            "lambada_openai_mt_fr",
            "fr",
            {"text": "Dans ma paume est une pierre claire et à l'intérieur repose une statuette"},
            "Dans ma paume est une pierre claire et à l'intérieur repose une",
            " statuette",
        ),
        (
            "lambada_openai_mt_it",
            "it",
            {"text": "Solo una fonte che so che sarebbe probabile trovare domani"},
            "Solo una fonte che so che sarebbe probabile trovare",
            " domani",
        ),
    ],
)
def test_lambada_multilingual_scores_single_continuation_loglikelihood(
    monkeypatch,
    factory_name: str,
    dataset_name: str,
    doc: dict[str, str],
    expected_prompt: str,
    expected_target: str,
) -> None:
    """Verify lambada multilingual scores single continuation loglikelihood. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list([doc])
    monkeypatch.setattr(lambada_module, "load_dataset", lambda *args, **kwargs: dataset)

    result = getattr(evalution.benchmarks, factory_name)(max_rows=1, batch_size=6).evaluate(
        FakeSession(
            expected_prompt=expected_prompt,
            expected_continuation=expected_target,
            output=LoglikelihoodOutput(
                logprob=-0.10536051565782628,
                is_greedy=True,
                token_count=1,
            ),
        )
    )

    assert result.name == factory_name
    assert result.metrics == pytest.approx(
        {
            "acc,ll": 1.0,
            "ppl,ll": math.exp(0.10536051565782628),
        }
    )
    assert result.metadata == {
        "dataset_path": "EleutherAI/lambada_openai",
        "dataset_name": dataset_name,
        "split": "test",
        "stream": False,
        "scoring_mode": "single_continuation_loglikelihood",
    }

    sample = result.samples[0]
    assert sample.prompt == expected_prompt
    assert sample.target == expected_target
    assert sample.prediction == expected_target
    assert sample.extracted == {
        "greedy_match": "1",
        "token_count": "1",
    }
    assert sample.metadata["text"] == doc["text"]
    assert sample.metadata["target_token"] == expected_target.strip()
