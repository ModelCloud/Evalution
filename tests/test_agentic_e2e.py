# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""End-to-end agentic runtime verification with real small models.

Runs a real tool-calling benchmark (Terminal-Bench 2.1 task directory) with
sandboxed runtimes to verify that:

a) model-generated commands execute on the configured runtime, not the host —
   the task command prints ``container`` only inside an Alpine runtime, and the
   host is asserted to lack ``/etc/alpine-release`` so a passing score proves
   non-local execution;
b) Evalution intercepts tool calls, executes them on the runtime, and resumes
   inference with the observation appended until the final answer.

Two model classes are covered, matching how models signal tool calls:

- native:    Llama-3.2-1B-Instruct is pre-trained with a tool-calling chat
             template; Evalution passes the OpenAI-style run_command schema
             through it and parses the model's native response encoding.
- prompted:  Falcon-H1-3B-Instruct has no native tool support; the generic,
             widely supported <tool_call></tool_call> marker syntax is injected as a
             system prompt and parsed from the generation.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from evalution.agent_runtime import DockerAgentRuntime, SmolVmAgentRuntime
from evalution.benchmarks import terminal_bench_21
from evalution.benchmarks.tool_calling import TOOL_CALL_FENCED_SHELL
from evalution.config import Model

MODEL_PATH = Path("/monster/data/model/Llama-3.2-1B-Instruct")
PROMPTED_MODEL_PATH = Path("/monster/data/model/Falcon-H1-3B-Instruct")

# Deterministic single-character probe: prints "0" only where Alpine's
# release file exists (inside the sandbox runtime), "1" everywhere else.
TASK_COMMAND = "ls /etc/alpine-release > /dev/null 2>&1; echo $?"

FENCED_INSTRUCTION = (
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

PROMPTED_INSTRUCTION = (
    "Determine where this shell is running. Run exactly this command:\n"
    + TASK_COMMAND
    + "\n\nBegin your reply with <tool_call>. Copy the command exactly "
    "without adding or changing anything."
)

NATIVE_INSTRUCTION = (
    "Determine where this shell is running. Run exactly this command:\n"
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


def _build_session(model_path: Path) -> Any:
    """Build one inference session for the given local model weights."""
    import torch

    from evalution.engines.transformers_compat import TransformersCompat

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = TransformersCompat(
        device=device,
        attn_implementation="eager",
        batch_size=1,
        max_new_tokens=96,
    )
    return engine.build(Model(path=str(model_path)))


@pytest.fixture(scope="module")
def smolvm_runtime(request: Any) -> Any:
    """Prepare an offline Alpine rootfs and verify smolvm can boot it.

    smolvm re-resolves registry images on every ephemeral run, so offline
    execution uses an unpacked rootfs directory exported from the local
    Docker daemon. The boot probe keeps the test honest about hosts where
    microVMs cannot actually start (for example nested LXC without working
    KVM passthrough). Note that smolvm runs its VMM under a per-VM
    unprivileged UID, so hosts with a root-restricted ``/dev/kvm`` must make
    it accessible (for example ``chmod 666 /dev/kvm``) for boot to succeed.
    The unpacked rootfs is likewise built in a world-traversable location.
    """
    if shutil.which("smolvm") is None:
        pytest.skip("smolvm CLI not available")
    if not _docker_available():
        pytest.skip("Docker daemon not available (needed to export the Alpine rootfs)")

    container_id = ""
    try:
        created = subprocess.run(
            ["docker", "create", "alpine:latest", "true"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        container_id = created.stdout.strip()
        exported = subprocess.run(
            ["docker", "export", container_id],
            capture_output=True,
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"cannot export Alpine rootfs for smolvm: {exc}")
    finally:
        if container_id:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                check=False,
                timeout=30,
            )

    # Build the rootfs in a world-traversable location: smolvm's per-VM
    # uid isolation runs the guest agent as an unprivileged UID that must be
    # able to read every directory on the path (pytest tmp dirs are 0700).
    rootfs = Path(tempfile.gettempdir()) / f"evalution-smolvm-alpine-{os.getpid()}"
    shutil.rmtree(rootfs, ignore_errors=True)
    rootfs.mkdir(parents=True)
    os.chmod(rootfs, 0o755)
    unpack = subprocess.run(
        ["tar", "-C", str(rootfs), "-xf", "-"],
        input=exported.stdout,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if unpack.returncode != 0:
        shutil.rmtree(rootfs, ignore_errors=True)
        pytest.skip(f"cannot unpack Alpine rootfs for smolvm: {unpack.stderr.decode()[:200]}")
    for current, dirs, files in os.walk(rootfs):
        try:
            for name in dirs:
                path = os.path.join(current, name)
                os.chmod(path, stat.S_IMODE(os.stat(path).st_mode) | 0o055)
            for name in files:
                path = os.path.join(current, name)
                os.chmod(path, stat.S_IMODE(os.stat(path).st_mode) | 0o044)
        except OSError:
            pass

    def _cleanup() -> None:
        shutil.rmtree(rootfs, ignore_errors=True)

    request.addfinalizer(_cleanup)

    probe = subprocess.run(
        [
            shutil.which("smolvm"),
            "machine",
            "run",
            "--image",
            str(rootfs),
            "--",
            "sh",
            "-c",
            "echo ok",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    if probe.returncode != 0 or "ok" not in probe.stdout:
        reason = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else probe.stdout
        pytest.skip(f"smolvm microVM boot is not functional on this host: {reason[:200]}")

    return SmolVmAgentRuntime(image=str(rootfs))


def _make_runtime_task(root: Path, instruction: str) -> None:
    """Create a Terminal-Bench task whose answer proves runtime execution."""
    tasks_dir = root / "tasks"
    task_dir = tasks_dir / "runtime-probe"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(instruction)
    solution_dir = task_dir / "solution"
    solution_dir.mkdir()
    (solution_dir / "solution.patch").write_text("0")


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
    assert sample.extracted["stdout"].strip() == "0"

    # The model resumed with the observed runtime output as its final answer.
    assert sample.scores["em"] == 1.0
    prediction_clean = re.sub(r"<\|[^>]*\|>", "", sample.prediction).strip().strip('"')
    assert prediction_clean == "0"


def test_agentic_e2e_native_tool_calling_model(tmp_path: Path) -> None:
    """Llama-3.2-1B-Instruct runs the task through its pre-trained native tools."""
    if not MODEL_PATH.is_dir():
        pytest.skip("Llama-3.2-1B-Instruct weights not available")
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    # Host sanity: without an Alpine release file, `container` can only come
    # from inside the sandboxed runtime.
    assert not Path("/etc/alpine-release").exists()

    session = _build_session(MODEL_PATH)
    try:
        _make_runtime_task(tmp_path, NATIVE_INSTRUCTION)
        suite = terminal_bench_21(
            dataset_path=str(tmp_path),
            max_rows=1,
            batch_size=1,
            max_new_tokens=96,
            max_tool_turns=4,
            apply_chat_template=True,
            tool_call_mode="native",
            agent_runtime=DockerAgentRuntime(image="alpine:latest", pull="missing"),
        )
        result = suite.evaluate(session)
    finally:
        session.close()

    # The model's pre-trained native template was used explicitly.
    assert result.metadata["tool_call_mode"] == "native"
    assert result.metadata["tool_call_format"] == "native_json"
    _assert_tool_loop_result(result, "DockerAgentRuntime")


def test_agentic_e2e_prompted_tool_calling_model(tmp_path: Path) -> None:
    """Falcon-H1-3B-Instruct (no native tools) runs via prompted <tool_call></tool_call>."""
    if not PROMPTED_MODEL_PATH.is_dir():
        pytest.skip("Falcon-H1-3B-Instruct weights not available")
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    assert not Path("/etc/alpine-release").exists()

    session = _build_session(PROMPTED_MODEL_PATH)
    try:
        _make_runtime_task(tmp_path, PROMPTED_INSTRUCTION)
        suite = terminal_bench_21(
            dataset_path=str(tmp_path),
            max_rows=1,
            batch_size=1,
            max_new_tokens=96,
            max_tool_turns=4,
            apply_chat_template=True,
            tool_call_mode="prompted",
            agent_runtime=DockerAgentRuntime(image="alpine:latest", pull="missing"),
        )
        result = suite.evaluate(session)
    finally:
        session.close()

    # The generic prompted <tool_call></tool_call> syntax was used explicitly.
    assert result.metadata["tool_call_mode"] == "prompted"
    assert result.metadata["tool_call_format"] == "tool_call_tags"
    _assert_tool_loop_result(result, "DockerAgentRuntime")


def test_agentic_e2e_fenced_shell_protocol(tmp_path: Path) -> None:
    """Llama-3.2-1B-Instruct also completes the task via fenced_shell protocol."""
    if not MODEL_PATH.is_dir():
        pytest.skip("Llama-3.2-1B-Instruct weights not available")
    if not _docker_available():
        pytest.skip("Docker daemon not available")
    assert not Path("/etc/alpine-release").exists()

    session = _build_session(MODEL_PATH)
    try:
        _make_runtime_task(tmp_path, FENCED_INSTRUCTION)
        suite = terminal_bench_21(
            dataset_path=str(tmp_path),
            max_rows=1,
            batch_size=1,
            max_new_tokens=96,
            max_tool_turns=4,
            apply_chat_template=True,
            tool_call_format=TOOL_CALL_FENCED_SHELL,
            agent_runtime=DockerAgentRuntime(image="alpine:latest", pull="missing"),
        )
        result = suite.evaluate(session)
    finally:
        session.close()

    _assert_tool_loop_result(result, "DockerAgentRuntime")


@pytest.mark.skipif(not MODEL_PATH.is_dir(), reason="Llama-3.2-1B-Instruct weights not available")
def test_agentic_e2e_smolvm_runtime(
    smolvm_runtime: Any,
    tmp_path: Path,
) -> None:
    """Llama-3.2-1B-Instruct completes a Terminal-Bench task through a smolvm microVM."""
    if not _docker_available():
        pytest.skip("Docker daemon not available (needed to export the Alpine rootfs)")
    assert not Path("/etc/alpine-release").exists()

    session = _build_session(MODEL_PATH)
    try:
        _make_runtime_task(tmp_path, NATIVE_INSTRUCTION)
        suite = terminal_bench_21(
            dataset_path=str(tmp_path),
            max_rows=1,
            batch_size=1,
            max_new_tokens=96,
            max_tool_turns=4,
            apply_chat_template=True,
            tool_call_mode="native",
            agent_runtime=smolvm_runtime,
        )
        result = suite.evaluate(session)
    finally:
        session.close()

    _assert_tool_loop_result(result, "SmolVmAgentRuntime")
