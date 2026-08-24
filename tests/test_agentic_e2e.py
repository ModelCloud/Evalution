# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""End-to-end agentic runtime verification with Llama-3.2-1B-Instruct.

Runs a real tool-calling benchmark (Terminal-Bench 2.1 task directory) with a
real model and sandboxed runtimes to verify that:

a) model-generated commands execute on the configured runtime, not the host —
   the task command prints ``container`` only inside an Alpine runtime, and the
   host is asserted to lack ``/etc/alpine-release`` so a passing score proves
   non-local execution;
b) Evalution intercepts the model's tool call, executes it on the runtime, and
   resumes inference with the observation appended until the final answer.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from evalution.agent_runtime import DockerAgentRuntime, SmolVmAgentRuntime
from evalution.benchmarks import terminal_bench_21
from evalution.config import AgentRuntimeConfig, Model

MODEL_PATH = Path("/monster/data/model/Llama-3.2-1B-Instruct")

TASK_COMMAND = "test -f /etc/alpine-release && echo container || echo host"

INSTRUCTION = (
    "You are a terminal agent in a sandboxed shell.\n"
    "When you need to run a command, reply with a single bash code block "
    "containing exactly the command and nothing else.\n"
    "Example reply:\n"
    "```bash\necho hello\n```\n"
    "After you receive the command output, reply with only the output text "
    "and nothing else.\n\n"
    "Task: determine where this shell is running. Run exactly this command:\n"
    + TASK_COMMAND
)


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            capture_output=True,
            check=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


@functools.lru_cache(maxsize=1)
def _smolvm_bootable() -> bool:
    """Probe whether smolvm can actually boot a microVM on this host."""
    if shutil.which("smolvm") is None:
        return False
    try:
        probe = subprocess.run(
            ["smolvm", "machine", "run", "--image", "alpine", "--", "sh", "-c", "echo ok"],
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        return probe.returncode == 0 and "ok" in probe.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="module")
def session() -> Any:
    """Build one Llama-3.2-1B-Instruct inference session shared by the e2e tests."""
    import torch

    from evalution.engines.transformers_compat import TransformersCompat

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = TransformersCompat(
        device=device,
        attn_implementation="eager",
        batch_size=1,
        max_new_tokens=64,
    )
    built = engine.build(Model(path=str(MODEL_PATH)))
    yield built
    built.close()


def _make_runtime_task(root: Path) -> None:
    """Create a Terminal-Bench task whose answer proves runtime execution."""
    tasks_dir = root / "tasks"
    task_dir = tasks_dir / "runtime-probe"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(INSTRUCTION)
    solution_dir = task_dir / "solution"
    solution_dir.mkdir()
    (solution_dir / "solution.patch").write_text("container")


def _assert_tool_loop_result(result: Any, runtime_type: str) -> None:
    """Verify interception, runtime execution, resumed inference, and scoring."""
    assert len(result.samples) == 1
    sample = result.samples[0]

    # b) Evalution intercepted the tool call and resumed inference.
    assert sample.metadata["commands_executed"] >= 1
    assert sample.metadata["tool_turns"] >= 2
    assert TASK_COMMAND in sample.extracted["commands"]

    # a) The command executed on the sandbox runtime, not the host.
    assert sample.metadata["runtime_type"] == runtime_type
    assert sample.extracted["stdout"].strip() == "container"

    # The model resumed with the observed runtime output as its final answer.
    assert sample.scores["em"] == 1.0
    assert sample.prediction.strip() == "container"


@pytest.mark.skipif(not MODEL_PATH.is_dir(), reason="Llama-3.2-1B-Instruct weights not available")
@pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available")
def test_agentic_e2e_docker_runtime(session: Any, tmp_path: Path) -> None:
    """Llama-3.2-1B completes a Terminal-Bench task through the Docker runtime."""
    # Host sanity: without an Alpine release file, `container` can only come
    # from inside the sandboxed runtime.
    assert not Path("/etc/alpine-release").exists()

    _make_runtime_task(tmp_path)
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        batch_size=1,
        max_new_tokens=64,
        max_tool_turns=4,
        apply_chat_template=True,
        agent_runtime=AgentRuntimeConfig(
            agent_runtime=DockerAgentRuntime(image="alpine:latest", pull="missing"),
        ),
    )
    result = suite.evaluate(session)
    _assert_tool_loop_result(result, "DockerAgentRuntime")


@pytest.mark.skipif(not MODEL_PATH.is_dir(), reason="Llama-3.2-1B-Instruct weights not available")
@pytest.mark.skipif(not _smolvm_bootable(), reason="smolvm microVM boot is not available on this host")
def test_agentic_e2e_smolvm_runtime(session: Any, tmp_path: Path) -> None:
    """Llama-3.2-1B completes a Terminal-Bench task through a smolvm microVM."""
    assert not Path("/etc/alpine-release").exists()

    _make_runtime_task(tmp_path)
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        batch_size=1,
        max_new_tokens=64,
        max_tool_turns=4,
        apply_chat_template=True,
        agent_runtime=AgentRuntimeConfig(
            agent_runtime=SmolVmAgentRuntime(image="alpine"),
        ),
    )
    result = suite.evaluate(session)
    _assert_tool_loop_result(result, "SmolVmAgentRuntime")
