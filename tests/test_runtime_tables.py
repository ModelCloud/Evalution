# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

from datasets import Dataset

import evalution
from evalution.engines.base import BaseEngine, BaseInferenceSession, GenerationOutput
from evalution import runtime as runtime_module

# Keep shared test fixtures and expectations explicit at module scope.
gsm8k_platinum_module = importlib.import_module("evalution.benchmarks.gsm8k_platinum")


class FakeEngine(BaseEngine):
    """Provide the fake engine helper used by the surrounding tests."""
    def __init__(self) -> None:
        """Initialize this object."""
        self.session = FakeSession()
        self.model_config = None

    def build(self, model):
        """Build build."""
        self.model_config = model
        return self.session

    def to_dict(self):
        """Implement to dict for fake engine."""
        return {"name": "fake"}


class FakeSession(BaseInferenceSession):
    """Provide the fake session helper used by the surrounding tests."""
    def generate(self, requests, *, batch_size=None):
        """Generate generate."""
        del batch_size
        return [
            GenerationOutput(
                prompt=request.prompt if request.prompt is not None else str(request.messages),
                text="The answer is 42.",
            )
            for request in requests
        ]

    def describe_execution(self):
        """Implement describe execution for fake session."""
        return {"generation_backend": "fake"}

    def loglikelihood(self, requests, *, batch_size=None):
        """Implement loglikelihood for fake session."""
        del requests, batch_size
        raise NotImplementedError

    def loglikelihood_rolling(self, requests, *, batch_size=None):
        """Implement loglikelihood rolling for fake session."""
        del requests, batch_size
        raise NotImplementedError

    def generate_continuous(self, requests, *, batch_size=None):
        """Generate continuous."""
        request_items = list(requests)
        outputs = self.generate([request for _, request in request_items], batch_size=batch_size)
        for (item_id, _request), output in zip(request_items, outputs, strict=True):
            yield item_id, output

    def gc(self) -> None:
        """Release reusable intermediate state for this object."""
        return None

    def close(self) -> None:
        """Release the resources owned by this object."""
        return None


def _dataset() -> Dataset:
    """Support the surrounding tests with dataset."""
    return Dataset.from_list(
        [
            {
                "question": "What is 40 plus 2?",
                "answer": "40 + 2 = 42\n#### 42",
                "cleaning_status": "consensus",
            }
        ]
    )


def test_evaluation_run_renders_per_suite_table(monkeypatch) -> None:
    """Verify evaluation run renders per suite table."""
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: _dataset())
    rendered_results = []
    rendered_summaries = []
    monkeypatch.setattr(
        runtime_module,
        "render_test_result_table",
        lambda result, *, logger=None: rendered_results.append((result.name, logger is not None)),
    )
    monkeypatch.setattr(
        runtime_module,
        "render_test_summary_table",
        lambda results, *, logger=None: rendered_summaries.append((len(results), logger is not None)),
    )

    evalution.run(
        model={"path": "/tmp/model"},
        engine=FakeEngine(),
        tests=[evalution.benchmarks.gsm8k_platinum(max_rows=1)],
    )

    assert rendered_results == [("gsm8k_platinum_cot", True)]
    assert rendered_summaries == [(1, True)]


def test_evaluation_run_renders_consolidated_summary_for_multiple_suites(monkeypatch) -> None:
    """Verify evaluation run renders consolidated summary for multiple suites."""
    monkeypatch.setattr(gsm8k_platinum_module, "load_dataset", lambda *args, **kwargs: _dataset())
    rendered_results = []
    rendered_summaries = []
    monkeypatch.setattr(
        runtime_module,
        "render_test_result_table",
        lambda result, *, logger=None: rendered_results.append(result.name),
    )
    monkeypatch.setattr(
        runtime_module,
        "render_test_summary_table",
        lambda results, *, logger=None: rendered_summaries.append([result.name for result in results]),
    )

    evalution.run(
        model={"path": "/tmp/model"},
        engine=FakeEngine(),
        tests=[
            evalution.benchmarks.gsm8k_platinum(max_rows=1),
            evalution.benchmarks.gsm8k_platinum(max_rows=1),
        ],
    )

    assert rendered_results == [
        "gsm8k_platinum_cot",
        "gsm8k_platinum_cot",
    ]
    assert rendered_summaries == [[
        "gsm8k_platinum_cot",
        "gsm8k_platinum_cot",
    ]]
