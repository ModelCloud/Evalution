# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import GenerationOutput

# Patch the shared IFEval loader because the Portuguese suite inherits it directly.
ifeval_module = importlib.import_module("evalution.benchmarks.ifeval")
# Import the Portuguese factory module so the test exercises the public benchmark entrypoint.
ifeval_pt_module = importlib.import_module("evalution.benchmarks.ifeval_pt")


class FakeSession:
    # Minimal generation session for single-sample instruction-following checks.
    """Provide the fake session helper used by the surrounding tests."""
    def __init__(self, responses: list[str]) -> None:
        """Initialize this object."""
        self.responses = responses

    def generate(self, requests, *, batch_size=None):
        # Mirror the benchmark session contract while returning fixed generations.
        """Generate generate."""
        assert batch_size in {1, 2, 4}
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text=response,
            )
            for request, response in zip(requests, self.responses, strict=True)
        ]

    def close(self) -> None:
        # Match the engine interface used by the benchmark runner.
        """Release the resources owned by this object."""
        return None


def test_ifeval_pt_scores_instruction_following(monkeypatch) -> None:
    """Verify IFEval pt scores instruction following. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "key": 1000,
                "prompt": "Escreva duas seções destacadas sem vírgulas.",
                "instruction_id_list": [
                    "punctuation:no_comma",
                    "detectable_format:number_highlighted_sections",
                ],
                "kwargs": [
                    {"num_highlights": None},
                    {"num_highlights": 2.0},
                ],
            }
        ]
    )
    monkeypatch.setattr(ifeval_module, "load_dataset", lambda *args, **kwargs: dataset)

    suite = ifeval_pt_module.ifeval_pt(max_rows=1, batch_size=4)
    result = suite.evaluate(FakeSession(["*Primeira seção*\n\n*Segunda seção*"]))

    assert result.name == "ifeval_pt"
    assert result.metrics == {
        "prompt_level_strict_acc": 1.0,
        "prompt_level_loose_acc": 1.0,
        "inst_level_strict_acc": 1.0,
        "inst_level_loose_acc": 1.0,
    }
    assert result.metadata["dataset_path"] == "Polygl0t/IFEval-PT"
    assert result.metadata["dataset_name"] is None
    assert result.metadata["split"] == "train"
    assert result.metadata["scoring_mode"] == "instruction_following"
    assert result.samples[0].prompt == "Escreva duas seções destacadas sem vírgulas."
    assert result.samples[0].target == result.samples[0].prompt
    assert result.samples[0].extracted["prompt_level_strict"] == "1"
    assert result.samples[0].metadata["instruction_count"] == 2


def test_ifeval_pt_scores_portuguese_case_checks(monkeypatch) -> None:
    """Verify IFEval pt scores portuguese case checks. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list(
        [
            {
                "key": 1001,
                "prompt": "Responda em portugues apenas com letras minusculas.",
                "instruction_id_list": ["change_case:portuguese_lowercase"],
                "kwargs": [{}],
            },
            {
                "key": 1002,
                "prompt": "Responda em portugues apenas com letras maiusculas.",
                "instruction_id_list": ["change_case:portuguese_capital"],
                "kwargs": [{}],
            },
        ]
    )
    monkeypatch.setattr(ifeval_module, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(ifeval_module, "_detect_language", lambda value: "pt")

    suite = ifeval_pt_module.ifeval_pt(max_rows=2, batch_size=4)
    result = suite.evaluate(FakeSession(["texto em portugues", "TEXTO EM PORTUGUES"]))

    assert result.metrics == {
        "prompt_level_strict_acc": 1.0,
        "prompt_level_loose_acc": 1.0,
        "inst_level_strict_acc": 1.0,
        "inst_level_loose_acc": 1.0,
    }
