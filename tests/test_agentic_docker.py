# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Unit tests for the Docker sandbox used by agentic benchmarks."""

from __future__ import annotations

import subprocess

import pytest

from evalution.benchmarks.agentic_docker import DockerSandbox, extract_command


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
def test_docker_sandbox_run_command() -> None:
    """Run a command in an Alpine container and capture stdout/stderr."""
    sandbox = DockerSandbox(pull="missing")
    result = sandbox.run("echo hello && echo error >&2")

    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert "error" in result.stderr
    assert result.duration_s >= 0.0


@pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available")
def test_docker_sandbox_network_isolated() -> None:
    """Verify the default network mode prevents outbound traffic."""
    sandbox = DockerSandbox(pull="missing")
    result = sandbox.run("wget -qO- https://example.com || echo 'network-blocked'")

    assert result.exit_code == 0 or "network-blocked" in result.stdout
    assert "Example Domain" not in result.stdout


def test_extract_command_strips_fences() -> None:
    """``extract_command`` removes markdown code fences and bash tags."""
    assert extract_command("```bash\necho hello\n```") == "echo hello"
    assert extract_command("```\necho hello\n```") == "echo hello"
    assert extract_command("<bash>echo hello</bash>") == "echo hello"
    assert extract_command("  echo hello  ") == "echo hello"
