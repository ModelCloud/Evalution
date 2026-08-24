# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Unit tests for the agent runtimes used by agentic benchmarks."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from evalution.agent_runtime import (
    DockerAgentRuntime,
    SmolVmAgentRuntime,
    UnsafeLocalRuntime,
)
from evalution.benchmarks.agentic_docker import extract_command


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


@pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available")
def test_docker_agent_runtime_run_command() -> None:
    """Run a command in an Alpine container and capture stdout/stderr."""
    runtime = DockerAgentRuntime(pull="missing")
    result = runtime.run("echo hello && echo error >&2")

    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert "error" in result.stderr
    assert result.duration_s >= 0.0


@pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available")
def test_docker_agent_runtime_network_isolated() -> None:
    """Verify the default network mode prevents outbound traffic."""
    runtime = DockerAgentRuntime(pull="missing")
    result = runtime.run("wget -qO- https://example.com || echo 'network-blocked'")

    assert result.exit_code == 0 or "network-blocked" in result.stdout
    assert "Example Domain" not in result.stdout


def test_extract_command_strips_fences() -> None:
    """``extract_command`` removes markdown code fences and bash tags."""
    assert extract_command("```bash\necho hello\n```") == "echo hello"
    assert extract_command("```\necho hello\n```") == "echo hello"
    assert extract_command("<bash>echo hello</bash>") == "echo hello"
    assert extract_command("  echo hello  ") == "echo hello"


def test_docker_runtime_builds_configured_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass Docker runtime settings to the Docker CLI without shell parsing."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        assert kwargs["check"] is False
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = DockerAgentRuntime(docker_path="/opt/docker", network="host").run(
        "printf '%s' hello",
        image="test:latest",
        env={"TOKEN": "value"},
        volumes={"/tmp/host": "/workspace"},
        workdir="/workspace",
    )

    assert result.stdout == "ok"
    assert calls == [[
        "/opt/docker",
        "run",
        "--rm",
        "-i",
        "--pull",
        "never",
        "--network",
        "host",
        "-e",
        "TOKEN=value",
        "-v",
        "/tmp/host:/workspace",
        "-w",
        "/workspace",
        "test:latest",
        "sh",
        "-c",
        "printf '%s' hello",
    ]]


def test_smolvm_runtime_builds_isolated_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate runtime options to the smolvm machine-run CLI."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    SmolVmAgentRuntime(smolvm_path="/opt/smolvm", network=True).run(
        "echo hello",
        image="alpine",
        env={"MODE": "test"},
        volumes={"/tmp/host": "/workspace"},
        workdir="/workspace",
    )

    assert calls == [[
        "/opt/smolvm",
        "machine",
        "run",
        "--net",
        "--image",
        "alpine",
        "--env",
        "MODE=test",
        "--volume",
        "/tmp/host:/workspace",
        "--workdir",
        "/workspace",
        "--",
        "sh",
        "-c",
        "echo hello",
    ]]


def test_unsafe_local_runtime_warns_on_construction() -> None:
    """Constructing the host runtime emits a security warning."""
    with pytest.warns(RuntimeWarning, match="without sandboxing"):
        UnsafeLocalRuntime()


def test_unsafe_local_runtime_runs_on_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate commands to a host shell invocation, ignoring isolation options."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        assert kwargs["check"] is False
        return SimpleNamespace(stdout="host", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.warns(RuntimeWarning, match="without sandboxing"):
        runtime = UnsafeLocalRuntime()
    result = runtime.run(
        "echo hello",
        image="alpine:latest",
        env={"MODE": "test"},
        volumes={"/tmp/host": "/workspace"},
        workdir="/workspace",
    )

    assert result.stdout == "host"
    assert result.exit_code == 0
    assert calls == [["sh", "-c", "echo hello"]]
