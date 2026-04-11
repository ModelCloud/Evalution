# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

from evalution.engines.base import GenerationOutput


def _row(**overrides):
    """Support the surrounding tests with row."""
    row = {
        "prompt_serial": "$hypothesis$ = h ; $context$ = c",
        "world_assump_label": "UNKNOWN",
        "proof_label": "UNKNOWN",
        "negative_world_assump_label": "UNKNOWN",
        "num_formula_distractors": 5,
        "num_translation_distractors": 0,
        "num_all_distractors": 5,
    }
    row.update(overrides)
    return row


class FakeSession:
    """Provide the fake session helper used by the surrounding tests."""
    def generate(self, requests, *, batch_size):
        """Generate generate."""
        assert batch_size == 1
        return [
            GenerationOutput(prompt=requests[0].prompt or "", text=" \n unknown \n ")
        ]


def test_fld_scores_normalized_generation(monkeypatch):
    """Verify fld scores normalized generation. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list([_row()])
    module = importlib.import_module("evalution.benchmarks.fld")

    def fake_load_dataset(*args, **kwargs):
        """Support the surrounding tests with fake load dataset."""
        return dataset

    monkeypatch.setattr(module, "load_dataset", fake_load_dataset)

    from evalution.benchmarks import fld

    result = fld(batch_size=1, max_rows=1).evaluate(FakeSession())
    sample = result.samples[0]

    assert result.name == "fld"
    assert result.metrics == {"em": 1.0}
    assert sample.prompt.startswith("Based on the provided facts")
    assert sample.target == "UNKNOWN"
    assert sample.extracted == {
        "prediction-stripped": "UNKNOWN",
        "target-stripped": "UNKNOWN",
    }
    assert sample.metadata["proof_label"] == "UNKNOWN"


def test_fld_requires_exact_label_after_whitespace_normalization(monkeypatch):
    """Verify fld requires exact label after whitespace normalization. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    dataset = Dataset.from_list([_row(world_assump_label="PROVED")])
    module = importlib.import_module("evalution.benchmarks.fld")

    class Session:
        """Define the session helper used by the surrounding tests."""
        def generate(self, requests, *, batch_size):
            """Generate generate."""
            return [
                GenerationOutput(prompt=requests[0].prompt or "", text="The verdict is proved")
            ]

    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: dataset)

    from evalution.benchmarks import fld

    result = fld(batch_size=1, max_rows=1).evaluate(Session())
    sample = result.samples[0]

    assert result.metrics == {"em": 0.0}
    assert sample.extracted == {
        "prediction-stripped": "THEVERDICTISPROVED",
        "target-stripped": "PROVED",
    }
