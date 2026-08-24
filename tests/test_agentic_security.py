# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Strict security tests for agentic benchmark execution.

Guarantees enforced here:

1. Benchmark modules can never execute commands themselves — every execution
   must route through the configured sandboxed ``BaseAgentRuntime``.
2. Tool calling is separated from code output: under the declared protocol,
   100% of tool calls are captured and routed to the runtime; anything else
   (plain code output, prose, undeclared formats) is never executed.
3. Missing runtimes fail closed, for both the loop and single-shot scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalution.agent_runtime import AgentRuntimeResult, BaseAgentRuntime
from evalution.benchmarks import terminal_bench_21
from evalution.benchmarks.tool_calling import RUN_COMMAND_TOOL, TOOL_CALL_FENCED_SHELL
from evalution.engines.base import GenerationOutput

_BENCHMARK_SOURCES = [
    Path("evalution/benchmarks/agentic.py"),
    Path("evalution/benchmarks/tool_calling.py"),
]

_FORBIDDEN_EXECUTION_TOKENS = (
    "subprocess",
    "os.system",
    "os.popen",
    "Popen(",
    "__import__",
    "exec(",
    "eval(",
)


class RecordingRuntime(BaseAgentRuntime):
    """Sandboxed runtime double that records every routed command."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.images: list[str | None] = []

    def run(self, command: str, *, image: str | None = None, **kwargs: Any) -> AgentRuntimeResult:
        del kwargs
        self.commands.append(command)
        self.images.append(image)
        return AgentRuntimeResult(
            stdout=f"out-of-{command}",
            stderr="",
            exit_code=0,
            command=[command],
            duration_s=0.0,
        )


class ScriptedSession:
    """Pops one reply per generate call; extra calls fail loudly."""

    batch_size = 1

    def __init__(self, replies: list[str], tokenizer: Any = None) -> None:
        self._replies = list(replies)
        self.prompts: list[Any] = []
        self.requests: list[Any] = []
        self.tokenizer = tokenizer

    def generate(self, requests: list[Any], batch_size: int) -> list[GenerationOutput]:
        del batch_size
        assert self._replies, "unexpected extra generate() call"
        prompt_request = requests[0]
        self.requests.append(prompt_request)
        self.prompts.append(prompt_request.prompt or prompt_request.messages)
        text = self._replies.pop(0)
        return [GenerationOutput(prompt=prompt_request.prompt or "", text=text)]

    def close(self) -> None:
        pass

    def gc(self) -> None:
        pass


class NativeCapableTokenizer:
    """Tokenizer double whose chat template accepts native tool schemas."""

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        assert kwargs.get("tools"), "native detection must pass the tool schema"
        return "<rendered>"


def test_auto_mode_resolves_native_for_capable_models(tmp_path: Path) -> None:
    """Models with pre-trained tool templates use their native format."""
    _make_task(tmp_path)
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        agent_runtime=RecordingRuntime(),
    )
    session = ScriptedSession(["done"], tokenizer=NativeCapableTokenizer())
    mode, fmt = suite._resolve_tool_calling(session)

    assert mode == "native"
    assert fmt == "native_json"


def test_auto_mode_falls_back_to_prompted_syntax(tmp_path: Path) -> None:
    """Models without native tools fall back to generic <bash></bash> markers."""
    _make_task(tmp_path)
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        agent_runtime=RecordingRuntime(),
    )
    session = ScriptedSession(["done"])  # no tokenizer => no native support
    mode, fmt = suite._resolve_tool_calling(session)

    assert mode == "prompted"
    assert fmt == "bash_tags"


def test_forced_native_without_support_fails_closed(tmp_path: Path) -> None:
    """Requesting native mode on an incapable model raises instead of degrading silently."""
    _make_task(tmp_path)
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        tool_call_mode="native",
        agent_runtime=RecordingRuntime(),
    )
    session = ScriptedSession(["done"])  # no tokenizer => not native-capable

    with pytest.raises(ValueError, match="native tool calling"):
        suite._resolve_tool_calling(session)


def test_prompted_mode_cannot_use_native_json(tmp_path: Path) -> None:
    """Prompted models cannot declare the native_json wire format."""
    _make_task(tmp_path)
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        tool_call_mode="prompted",
        tool_call_format="native_json",
        agent_runtime=RecordingRuntime(),
    )

    with pytest.raises(ValueError):
        suite._resolve_tool_calling(ScriptedSession(["done"]))


def test_unknown_tool_call_mode_rejected_at_construction(tmp_path: Path) -> None:
    """Invalid modes fail at construction time, before any model is loaded."""
    with pytest.raises(ValueError, match="unknown tool_call_mode"):
        terminal_bench_21(dataset_path=str(tmp_path), tool_call_mode="telepathy")


def test_native_tool_calls_route_through_runtime(tmp_path: Path) -> None:
    """Native JSON responses (python_tag encoding) execute on the sandbox runtime."""
    _make_task(tmp_path, solution="hi")
    runtime = RecordingRuntime()
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        tool_call_mode="native",
        agent_runtime=runtime,
    )
    session = ScriptedSession(
        [
            '<|python_tag|>{"name": "run_command", "parameters": {"command": "echo hi"}}',
            "hi",
        ],
        tokenizer=NativeCapableTokenizer(),
    )
    result = suite.evaluate(session)

    assert runtime.commands == ["echo hi"]
    turn_request = session.requests[0]
    assert turn_request.tools == [{"type": "function", "function": RUN_COMMAND_TOOL["function"]}]
    sample = result.samples[0]
    assert sample.metadata["tool_call_mode"] == "native"
    assert sample.scores["em"] == 1.0


def _make_task(root: Path, solution: str = "unused") -> Path:
    tasks_dir = root / "tasks"
    task_dir = tasks_dir / "security-probe"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Run the probe.")
    solution_dir = task_dir / "solution"
    solution_dir.mkdir()
    (solution_dir / "solution.patch").write_text(solution)
    return root


def test_benchmark_sources_never_execute_commands_directly() -> None:
    """Agentic benchmark sources contain no execution primitives at all.

    This is a tripwire: any future direct subprocess/os execution added to
    benchmark modules fails here instead of silently bypassing the runtime.
    """
    for source_path in _BENCHMARK_SOURCES:
        source = source_path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_EXECUTION_TOKENS:
            assert token not in source, f"forbidden {token!r} in {source_path}"


def test_code_output_is_inert_under_default_protocol(tmp_path: Path) -> None:
    """A fenced bash block that is mere code output must never execute."""
    _make_task(tmp_path)
    runtime = RecordingRuntime()
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        agent_runtime=runtime,
    )
    # The model answers with a plain bash snippet — classic code output.
    result = suite.evaluate(ScriptedSession(["```bash\necho pwned\n```"]))
    sample = result.samples[0]

    assert runtime.commands == [], "code output was executed as a tool call"
    assert sample.metadata["commands_executed"] == 0
    assert sample.metadata["tool_turns"] == 1
    # The fence stays inert model output and is treated as the final answer.
    assert "echo pwned" in sample.prediction


def test_declared_protocol_routes_every_tool_call_through_runtime(tmp_path: Path) -> None:
    """100% of tool calls in one generation execute on the sandbox runtime."""
    _make_task(tmp_path)
    runtime = RecordingRuntime()
    suite = terminal_bench_21(
        dataset_path=str(tmp_path),
        max_rows=1,
        tool_call_format=TOOL_CALL_FENCED_SHELL,
        agent_runtime=runtime,
    )
    session = ScriptedSession([
        "I will probe.\n```bash\necho one\n```\nthen:\n```sh\necho two\n```",
        "done",
    ])
    suite.evaluate(session)

    assert runtime.commands == ["echo one", "echo two"]
    assert len(session.prompts) == 2
    second_turn_prompt = str(session.prompts[1])
    assert "out-of-echo one" in second_turn_prompt
    assert "out-of-echo two" in second_turn_prompt


def test_task_image_is_routed_to_runtime(tmp_path: Path) -> None:
    """A task.toml docker_image is forwarded to the runtime, not used locally."""
    root = _make_task(tmp_path)
    task_toml = root / "tasks" / "security-probe" / "task.toml"
    task_toml.write_text(
        '[environment]\ndocker_image = "harbor/security-probe:7"\n',
        encoding="utf-8",
    )
    runtime = RecordingRuntime()
    suite = terminal_bench_21(
        dataset_path=str(root),
        max_rows=1,
        agent_runtime=runtime,
    )
    suite.evaluate(ScriptedSession(["<bash>echo probe</bash>", "done"]))

    assert runtime.images == ["harbor/security-probe:7"]


def test_missing_runtime_fails_closed_for_loop_and_single_shot(tmp_path: Path) -> None:
    """Both evaluate() and score_sample() refuse to run without a runtime."""
    _make_task(tmp_path)
    suite = terminal_bench_21(dataset_path=str(tmp_path), max_rows=1)

    with pytest.raises(ValueError, match="requires.*AgentRuntime"):
        suite.evaluate(ScriptedSession(["<bash>anything</bash>"]))

    from evalution.benchmarks.agentic import _load_local_tasks_dataset
    from evalution.benchmarks.execution import PreparedSample

    docs = list(_load_local_tasks_dataset(str(tmp_path)))
    prepared = next(suite.iter_prepared_samples(docs))
    assert isinstance(prepared, PreparedSample)
    with pytest.raises(ValueError, match="requires.*AgentRuntime"):
        suite.score_sample(prepared, GenerationOutput(prompt="p", text="<bash>x</bash>"))
