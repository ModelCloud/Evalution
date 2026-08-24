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
from evalution.agent_runtime import AgentRuntimeResult, BaseAgentRuntime
from evalution.benchmarks import (
    agentbench,
    babi,
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
    """Scripted inference session: pops one reply per generate call.

    A plain string repeats forever; a list is consumed in order and any extra
    generate call raises so runaway tool loops fail loudly in tests.
    """

    batch_size = 1

    def __init__(self, replies: Any) -> None:
        if isinstance(replies, str):
            self._replies: list[str] = [replies]
            self._infinite = True
        else:
            self._replies = list(replies)
            self._infinite = False
        self.prompts: list[str] = []

    def generate(self, requests: list[Any], batch_size: int) -> list[GenerationOutput]:
        del batch_size
        if not self._replies:
            raise AssertionError("FakeSession received an unexpected extra generate() call")
        text = self._replies[0] if self._infinite else self._replies.pop(0)
        prompt = getattr(requests[0], "prompt", "") or ""
        self.prompts.append(prompt)
        return [
            GenerationOutput(
                prompt=prompt,
                text=text,
            )
        ]

    def close(self) -> None:
        pass

    def gc(self) -> None:
        pass


class FakeAgentRuntime(BaseAgentRuntime):
    """Agent runtime test double that returns canned stdout without executing."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.commands: list[str] = []

    def run(self, command: str, **kwargs: Any) -> AgentRuntimeResult:
        del kwargs
        self.commands.append(command)
        return AgentRuntimeResult(
            stdout=self.stdout,
            stderr="",
            exit_code=0,
            command=[command],
            duration_s=0.0,
        )


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
    """Run the tool loop for Terminal-Bench 2.1 using a local task directory."""
    _make_local_task_dir(tmp_path, "task-1", "List files and exit.", "ls\n")

    runtime = FakeAgentRuntime("ls\n")
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        batch_size=1,
        max_new_tokens=5,
        agent_runtime=runtime,
    )
    session = FakeSession(["<bash>ls</bash>", "ls"])
    result = suite.evaluate(session)

    assert result.name == "terminal_bench_21"
    assert len(result.samples) == 1
    assert result.samples[0].scores["em"] == 1.0
    assert result.samples[0].metadata["commands_executed"] == 1


def test_deep_swe_local_task_forward_pass(tmp_path: Any) -> None:
    """Run the tool loop for DeepSWE using a local task directory."""
    _make_local_task_dir(tmp_path, "task-1", "Fix the bug.", "diff --git\n")

    suite = deep_swe(
        dataset_path=str(tmp_path),
        max_rows=1,
        batch_size=1,
        max_new_tokens=5,
        agent_runtime=FakeAgentRuntime("applied"),
    )
    result = suite.evaluate(FakeSession(["<bash>git apply fix.patch</bash>", "diff --git"]))

    assert result.name == "deep_swe"
    assert len(result.samples) == 1
    assert result.samples[0].scores["em"] == 1.0


def test_toolathlon_verified_local_task_forward_pass(tmp_path: Any) -> None:
    """Run the tool loop for Toolathlon-Verified using a local task directory."""
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

    suite = toolathlon_verified(
        dataset_path=str(tmp_path),
        max_rows=1,
        batch_size=1,
        max_new_tokens=5,
        agent_runtime=FakeAgentRuntime("expected tool output"),
    )
    result = suite.evaluate(FakeSession(["<bash>cat answer</bash>", "expected tool output"]))

    assert result.name == "toolathlon_verified"
    assert len(result.samples) == 1
    assert result.samples[0].scores["em"] == 1.0


def test_tool_loop_intercepts_and_resumes_inference(tmp_path: Any) -> None:
    """Evalution intercepts the tool call, executes it on the runtime, and resumes."""
    _make_local_task_dir(tmp_path, "task-1", "Print the marker.", "marker")
    runtime = FakeAgentRuntime("marker")
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        batch_size=1,
        max_new_tokens=5,
        agent_runtime=runtime,
    )
    session = FakeSession(["<bash>echo marker</bash>", "marker"])
    result = suite.evaluate(session)
    sample = result.samples[0]

    assert runtime.commands == ["echo marker"]
    assert len(session.prompts) == 2
    assert "Print the marker" in session.prompts[0]
    assert "<bash>echo marker</bash>" in session.prompts[1]
    assert "<bash_result>" in session.prompts[1]
    assert "marker" in session.prompts[1]
    assert sample.metadata["tool_turns"] == 2
    assert sample.metadata["commands_executed"] == 1
    assert sample.metadata["runtime_type"] == "FakeAgentRuntime"
    assert sample.scores["em"] == 1.0


def test_tool_loop_stops_at_max_tool_turns(tmp_path: Any) -> None:
    """A model that never stops emitting tool calls terminates at the turn cap."""
    _make_local_task_dir(tmp_path, "task-1", "Loop forever.", "anything")
    runtime = FakeAgentRuntime("ignored")
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        batch_size=1,
        max_new_tokens=5,
        max_tool_turns=3,
        agent_runtime=runtime,
    )
    result = suite.evaluate(FakeSession("<bash>echo loop</bash>"))

    sample = result.samples[0]
    assert sample.metadata["tool_turns"] == 3
    assert sample.metadata["commands_executed"] == 3
    assert runtime.commands == ["echo loop"] * 3


@pytest.mark.parametrize(
    "factory",
    [terminal_bench_21, deep_swe, toolathlon_verified],
)
def test_tool_calling_tasks_require_agent_runtime(factory: Any) -> None:
    """Refuse to evaluate tool-calling suites when no runtime is configured."""
    suite = factory(dataset_path="/nonexistent-tasks")

    with pytest.raises(ValueError, match="requires.*AgentRuntime"):
        suite.evaluate(FakeSession("any output"))


@pytest.mark.parametrize(
    "factory",
    [
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
    ],
)
def test_agentic_suites_declare_is_agentic(factory: Any) -> None:
    """Every agentic scaffold carries the declarative is_agentic flag."""
    assert factory().is_agentic is True


@pytest.mark.parametrize(
    "factory",
    [terminal_bench_21, deep_swe, toolathlon_verified],
)
def test_tool_calling_suites_declare_has_tool_calling(factory: Any) -> None:
    """Only command-executing suites carry has_tool_calling."""
    assert factory().has_tool_calling is True
    assert factory().is_agentic is True


def test_text_scaffold_is_not_flagged_as_tool_calling() -> None:
    """Dataset-backed agentic scaffolds do not execute generated commands."""
    suite = swe_bench()
    assert suite.is_agentic is True
    assert suite.has_tool_calling is False


def test_non_agentic_suite_defaults_to_unflagged() -> None:
    """Regular suites default to both flags off."""
    suite = babi()
    assert suite.is_agentic is False
    assert suite.has_tool_calling is False


def test_central_enforcement_applies_to_any_tool_calling_suite() -> None:
    """The shared pipeline blocks any suite that declares tool calling without a runtime."""
    suite = babi()
    suite.has_tool_calling = True

    with pytest.raises(ValueError, match="requires.*AgentRuntime"):
        suite.evaluate(FakeSession("any output"))
