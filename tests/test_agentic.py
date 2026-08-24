# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""CPU-only forward-pass validation for the new agentic benchmark suites.

These tests use ``sshleifer/tiny-gpt2`` on CPU with a single sample per suite
and a very small ``max_new_tokens`` value.  They verify that each suite can
load data, prepare a request, run a forward pass, and return a ``TestResult``
without crashing.  Scores are expected to be near zero because the model is
not solving agentic tasks.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from datasets import Dataset

import evalution.benchmarks.agentic as agentic_module
from evalution.benchmarks import (
    agentbench,
    deep_swe,
    gaia,
    gaia_level1,
    gaia_level2,
    gaia_level3,
    osworld,
    swe_atlas_qna,
    swe_bench,
    swe_bench_multilingual,
    swe_bench_pro,
    terminal_bench_21,
    toolathlon_verified,
    webarena,
    webarena_hard,
)
from evalution.config import Model
from evalution.engines.base import GenerationOutput
from evalution.engines.transformers_compat import TransformersCompat

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def session() -> Any:
    """Build a tiny CPU TransformersCompat session shared by all tests."""
    engine = TransformersCompat(
        device="cpu",
        attn_implementation="eager",
        batch_size=1,
        max_new_tokens=5,
    )
    return engine.build(Model(path=TINY_MODEL))


def _fake_gaia_loader(*args: Any, **kwargs: Any) -> Dataset:
    """Return a minimal in-memory GAIA split to avoid gated dataset auth."""
    del args, kwargs
    return Dataset.from_dict(
        {
            "task_id": ["gaia-1"],
            "Question": ["What is 2 + 2?"],
            "Final answer": ["4"],
            "file_name": [""],
            "file_path": [""],
            "Level": ["1"],
        }
    )


def _fake_swe_bench_loader(*args: Any, **kwargs: Any) -> Dataset:
    """Return a minimal in-memory SWE-bench split for CPU validation."""
    del args, kwargs
    return Dataset.from_dict(
        {
            "instance_id": ["swe-1"],
            "problem_statement": ["Fix the bug"],
            "patch": ["target patch"],
            "repo": ["test-repo"],
        }
    )


def _fake_swe_atlas_qna_loader(*args: Any, **kwargs: Any) -> Dataset:
    """Return a minimal in-memory SWE-Atlas-QnA split for CPU validation."""
    del args, kwargs
    return Dataset.from_dict(
        {
            "task_id": ["atlas-1"],
            "prompt": ["What does this code do?"],
            "reference_answer": ["reference answer"],
            "language": ["python"],
            "category": ["api"],
            "repository_url": ["https://github.com/test/repo"],
        }
    )


class FakeSession:
    """Lightweight inference session that returns a fixed string for every request."""

    batch_size = 1

    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, requests: list[Any], batch_size: int) -> list[GenerationOutput]:
        return [
            GenerationOutput(
                prompt=getattr(request, "prompt", "") or "",
                text=self._text,
            )
            for request in requests
        ]

    def close(self) -> None:
        pass

    def gc(self) -> None:
        pass


def _make_local_task_dir(root: Any, task_name: str, instruction: str, solution: str) -> None:
    """Create a minimal Harbor-style task directory under ``root/tasks``."""
    tasks_dir = root / "tasks"
    task_dir = tasks_dir / task_name
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(instruction)
    solution_dir = task_dir / "solution"
    solution_dir.mkdir()
    (solution_dir / "solution.patch").write_text(solution)


@pytest.mark.parametrize(
    "factory",
    [
        agentbench,
        osworld,
        webarena,
        webarena_hard,
    ],
)
def test_public_agentic_suite_forward_pass(factory: Any, session: Any) -> None:
    """Run one forward pass for each public agentic benchmark suite."""
    suite = factory(max_rows=1, batch_size=1, max_new_tokens=5)
    result = suite.evaluate(session)

    assert result.name == suite.task_name()
    assert len(result.samples) == 1
    assert "em" in result.samples[0].scores
    assert result.metrics == {"em": result.samples[0].scores["em"]}


def test_swe_bench_forward_pass(session: Any) -> None:
    """Run one forward pass for SWE-bench using the dev split for speed."""
    suite = swe_bench(split="dev", max_rows=1, batch_size=1, max_new_tokens=5)
    result = suite.evaluate(session)

    assert result.name == "swe_bench"
    assert len(result.samples) == 1
    assert "em" in result.samples[0].scores


@pytest.mark.parametrize(
    "factory,expected_name",
    [
        (gaia, "gaia"),
        (gaia_level1, "gaia_level1"),
        (gaia_level2, "gaia_level2"),
        (gaia_level3, "gaia_level3"),
    ],
)
def test_gaia_forward_pass(
    factory: Any,
    expected_name: str,
    session: Any,
    monkeypatch: Any,
) -> None:
    """Run one forward pass for each GAIA variant using a mocked dataset."""
    monkeypatch.setattr(agentic_module, "load_dataset", _fake_gaia_loader)

    suite = factory(max_rows=1, batch_size=1, max_new_tokens=5)
    result = suite.evaluate(session)

    assert result.name == expected_name
    assert len(result.samples) == 1
    assert "em" in result.samples[0].scores


@pytest.mark.parametrize(
    "factory,loader,output",
    [
        (swe_bench_multilingual, _fake_swe_bench_loader, "target patch"),
        (swe_bench_pro, _fake_swe_bench_loader, "target patch"),
        (swe_atlas_qna, _fake_swe_atlas_qna_loader, "reference answer"),
    ],
)
def test_public_laguna_agentic_suite_forward_pass(
    factory: Any,
    loader: Any,
    output: str,
    monkeypatch: Any,
) -> None:
    """Run one forward pass for the public PoolSide Laguna 2.1 suites."""
    monkeypatch.setattr(agentic_module, "load_dataset", loader)

    suite = factory(max_rows=1, batch_size=1, max_new_tokens=5)
    result = suite.evaluate(FakeSession(output))

    assert result.name == suite.task_name()
    assert len(result.samples) == 1
    assert "em" in result.samples[0].scores
    assert result.samples[0].scores["em"] == 1.0


def test_terminal_bench_21_local_task_forward_pass(tmp_path: Any) -> None:
    """Run one forward pass for Terminal-Bench 2.1 using a local task directory."""
    _make_local_task_dir(tmp_path, "task-1", "List files and exit.", "ls\n")

    suite = terminal_bench_21(dataset_path=str(tmp_path), max_rows=1, batch_size=1, max_new_tokens=5)
    result = suite.evaluate(FakeSession("ls\n"))

    assert result.name == "terminal_bench_21"
    assert len(result.samples) == 1
    assert result.samples[0].scores["em"] == 1.0


def test_deep_swe_local_task_forward_pass(tmp_path: Any) -> None:
    """Run one forward pass for DeepSWE using a local task directory."""
    _make_local_task_dir(tmp_path, "task-1", "Fix the bug.", "diff --git\n")

    suite = deep_swe(dataset_path=str(tmp_path), max_rows=1, batch_size=1, max_new_tokens=5)
    result = suite.evaluate(FakeSession("diff --git\n"))

    assert result.name == "deep_swe"
    assert len(result.samples) == 1
    assert result.samples[0].scores["em"] == 1.0


def test_toolathlon_verified_local_task_forward_pass(tmp_path: Any) -> None:
    """Run one forward pass for Toolathlon-Verified using a local task directory."""
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "task-1"
    task_dir.mkdir(parents=True)
    config = {
        "id": "toolathlon-task-1",
        "meta": {
            "description": "Use a tool.",
            "requirements": ["Open the file"],
        },
    }
    (task_dir / "task_config.json").write_text(__import__("json").dumps(config))
    solution_dir = task_dir / "solution"
    solution_dir.mkdir()
    (solution_dir / "solution.patch").write_text("expected tool output")

    suite = toolathlon_verified(dataset_path=str(tmp_path), max_rows=1, batch_size=1, max_new_tokens=5)
    result = suite.evaluate(FakeSession("expected tool output"))

    assert result.name == "toolathlon_verified"
    assert len(result.samples) == 1
    assert result.samples[0].scores["em"] == 1.0
