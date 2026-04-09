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

lambada_module = importlib.import_module("evalution.benchmarks.lambada")


class FakeSession:
    def __init__(
        self,
        *,
        expected_prompt: str,
        expected_continuation: str,
        output: LoglikelihoodOutput,
    ) -> None:
        self.expected_prompt = expected_prompt
        self.expected_continuation = expected_continuation
        self.output = output

    def loglikelihood(self, requests, *, batch_size=None):
        assert batch_size == 6
        assert len(requests) == 1
        assert requests[0].context == self.expected_prompt
        assert requests[0].continuation == self.expected_continuation
        return [self.output]


@pytest.mark.parametrize(
    ("factory_name", "dataset_name", "doc", "expected_prompt", "expected_target"),
    [
        (
            "lambada_openai_mt_stablelm_de",
            "de",
            {"text": "Stattdessen starrte er geradeaus, als sei er tief interessiert, was an der Vorderseite des Flugzeugs passierte"},
            "Stattdessen starrte er geradeaus, als sei er tief interessiert, was an der Vorderseite des Flugzeugs",
            " passierte",
        ),
        (
            "lambada_openai_mt_stablelm_en",
            "en",
            {"text": "In my palm is a clear stone, and inside it is a small ivory statuette guardia"},
            "In my palm is a clear stone, and inside it is a small ivory statuette",
            " guardia",
        ),
        (
            "lambada_openai_mt_stablelm_es",
            "es",
            {"text": "Escuchó a Rhinna hablar y entonces Tom comprendió todo"},
            "Escuchó a Rhinna hablar y entonces Tom comprendió",
            " todo",
        ),
        (
            "lambada_openai_mt_stablelm_fr",
            "fr",
            {"text": "Dans ma paume est une pierre claire et à l'intérieur repose une statuette"},
            "Dans ma paume est une pierre claire et à l'intérieur repose une",
            " statuette",
        ),
        (
            "lambada_openai_mt_stablelm_it",
            "it",
            {"text": "Solo una fonte che so che sarebbe probabile trovare domani"},
            "Solo una fonte che so che sarebbe probabile trovare",
            " domani",
        ),
        (
            "lambada_openai_mt_stablelm_nl",
            "nl",
            {"text": "Dus keek hij rechtuit naar voren, alsof hij diep geïnteresseerd was in wat er gebeurde aan de voorkant van het vliegtuig"},
            "Dus keek hij rechtuit naar voren, alsof hij diep geïnteresseerd was in wat er gebeurde aan de voorkant van het",
            " vliegtuig",
        ),
        (
            "lambada_openai_mt_stablelm_pt",
            "pt",
            {"text": "Em vez disso, ele fitava para frente, como se estivesse profundamente interessado no que acontecia na parte da frente do avião"},
            "Em vez disso, ele fitava para frente, como se estivesse profundamente interessado no que acontecia na parte da frente do",
            " avião",
        ),
    ],
)
def test_lambada_multilingual_stablelm_scores_single_continuation_loglikelihood(
    monkeypatch,
    factory_name: str,
    dataset_name: str,
    doc: dict[str, str],
    expected_prompt: str,
    expected_target: str,
) -> None:
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
        "dataset_path": "EleutherAI/lambada_multilingual_stablelm",
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
