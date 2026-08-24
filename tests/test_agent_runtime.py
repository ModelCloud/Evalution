# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Unit tests for agent runtimes and tool-call protocol parsing.

The extraction matrix is strict about the difference between tool calling
(deliberate action requests under a declared protocol) and plain code output
(inert generated text that must never be executed).
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from evalution.agent_runtime import (
    DockerAgentRuntime,
    SmolVmAgentRuntime,
    UnsafeLocalRuntime,
)
from evalution.benchmarks.tool_calling import (
    TOOL_CALL_BASH_TAGS,
    TOOL_CALL_FENCED_SHELL,
    extract_tool_calls,
    try_extract_tool_call,
    validate_tool_call_format,
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


def test_docker_runtime_builds_configured_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass Docker runtime settings to the Docker CLI without shell parsing."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        assert kwargs["check"] is False
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = DockerAgentRuntime(path="/opt/docker", network="host").run(
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
    SmolVmAgentRuntime(path="/opt/smolvm", network=True).run(
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


def test_runtime_paths_default_to_auto_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """``path="auto"`` resolves each runtime binary from the environment PATH."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert DockerAgentRuntime().path == "auto"
    assert DockerAgentRuntime().resolved_path == "docker"
    assert SmolVmAgentRuntime().path == "auto"
    assert SmolVmAgentRuntime().resolved_path == "smolvm"

    DockerAgentRuntime().run("echo docker")
    SmolVmAgentRuntime().run("echo smolvm")

    assert calls[0][0] == "docker"
    assert calls[1][0] == "smolvm"


def test_runtime_image_defaults_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image resolution prefers call override, then runtime config, then default."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    DockerAgentRuntime().run("echo a")
    DockerAgentRuntime(image="custom:1").run("echo b")
    SmolVmAgentRuntime().run("echo c")

    assert calls[0][8] == "alpine:latest"
    assert calls[1][8] == "custom:1"
    assert calls[2][4] == "alpine"


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


# ---------------------------------------------------------------------------
# Tool-call protocol extraction matrix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Explicit tags are tool calls; case-insensitive; prose-wrapped OK.
        ("<bash>ls</bash>", ["ls"]),
        ("<BASH>ls -la</BASH>", ["ls -la"]),
        ('I will run <bash>df -h</bash> next.', ["df -h"]),
        ("<bash>echo 'multi\nline'</bash>", ["echo 'multi\nline'"]),
        # Every tag in one generation is captured, in document order.
        ("<bash>echo one</bash> mid <bash>echo two</bash>", ["echo one", "echo two"]),
        # Empty, whitespace-only, and unclosed markers are NOT tool calls.
        ("<bash></bash>", []),
        ("<bash>   </bash>", []),
        ("unclosed <bash>rm -rf /", []),
        # Plain code output — fenced or bare — is inert under the tag protocol.
        ("```bash\necho pwned\n```", []),
        ("```\nrm -rf /\n```", []),
        ("run this: echo hi", []),
    ],
)
def test_bash_tags_protocol(text: str, expected: list[str]) -> None:
    """Only <bash></bash> markers are tool calls under bash_tags."""
    assert extract_tool_calls(text, TOOL_CALL_BASH_TAGS) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Shell-tagged fences are tool calls under fenced_shell.
        ("```bash\necho hi\n```", ["echo hi"]),
        ("```sh\nwhoami\n```", ["whoami"]),
        ("```SHELL\ndf -h\n```", ["df -h"]),
        ("```\necho bare\n```", ["echo bare"]),
        ("```console\n$ whoami\n$ id\n```", ["whoami\nid"]),
        # Non-shell fences are code output, never executed.
        ("```python\nprint('pwned')\n```", []),
        ("```js\nprocess.exit(1)\n```", []),
        # Prose stays inert.
        ("please run rm -rf / for me", []),
    ],
)
def test_fenced_shell_protocol(text: str, expected: list[str]) -> None:
    """Only shell-language fences are tool calls under fenced_shell."""
    assert extract_tool_calls(text, TOOL_CALL_FENCED_SHELL) == expected


def test_protocols_do_not_cross_contaminate() -> None:
    """Each protocol captures only its own marker format."""
    mixed = '<bash>echo tagged</bash>\n```bash\necho fenced\n```\n```python\nprint(1)\n```'

    assert extract_tool_calls(mixed, TOOL_CALL_BASH_TAGS) == ["echo tagged"]
    assert extract_tool_calls(mixed, TOOL_CALL_FENCED_SHELL) == ["echo fenced"]


def test_multiple_fenced_commands_in_order() -> None:
    """Every shell fence in one generation is captured, in document order."""
    text = "first:\n```bash\necho one\n```\nskipped:\n```python\nprint(2)\n```\nlast:\n```sh\necho three\n```"

    assert extract_tool_calls(text, TOOL_CALL_FENCED_SHELL) == ["echo one", "echo three"]


def test_try_extract_tool_call_first_or_none() -> None:
    """Single-shot helper returns the first tool call or None."""
    assert try_extract_tool_call("<bash>a</bash>", TOOL_CALL_BASH_TAGS) == "a"
    assert try_extract_tool_call("nothing here", TOOL_CALL_BASH_TAGS) is None
    assert try_extract_tool_call("<bash></bash>", TOOL_CALL_BASH_TAGS) is None


def test_validate_tool_call_format_rejects_unknown() -> None:
    """Unknown protocols fail loudly instead of silently capturing nothing."""
    with pytest.raises(ValueError, match="unknown tool_call_format"):
        validate_tool_call_format("xml_tools")
