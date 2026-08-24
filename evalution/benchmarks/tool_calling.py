# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Tool-call parsing for agentic benchmarks.

Tool calling is fundamentally different from code output: a tool call is the
model's deliberate request to execute an action in a sandboxed runtime, while
code output (for example a fenced ``bash`` snippet inside a plain answer) is
inert generated text that must never be executed. Because every model family
signals tool calls differently, the expected protocol is declared explicitly
per suite and :func:`extract_tool_calls` only ever captures that protocol;
everything else stays inert model output.
"""

from __future__ import annotations

from typing import Any

import pcre

TOOL_CALL_TAGS = "tool_call_tags"
TOOL_CALL_FENCED_SHELL = "fenced_shell"
TOOL_CALL_NATIVE_JSON = "native_json"
TOOL_CALL_FORMATS = (
    TOOL_CALL_TAGS,
    TOOL_CALL_FENCED_SHELL,
    TOOL_CALL_NATIVE_JSON,
)

# How tool calls are signalled: natively through the model's own pre-trained
# tool-calling template, or through a generic prompted syntax for models that
# were never trained to call tools.
TOOL_CALL_MODE_AUTO = "auto"
TOOL_CALL_MODE_NATIVE = "native"
TOOL_CALL_MODE_PROMPTED = "prompted"
TOOL_CALL_MODES = (TOOL_CALL_MODE_AUTO, TOOL_CALL_MODE_NATIVE, TOOL_CALL_MODE_PROMPTED)

# Generic prompted contract: explicit <tool_call></tool_call> action markers.
# Deliberately NOT <bash>/fences, which models also emit as plain code output;
# only the strict action marker is intercepted and executed.
PROMPTED_TOOL_SYSTEM_MESSAGE = (
    "You are a terminal agent connected to a sandboxed shell.\n"
    "To run a shell command you MUST reply with ONLY the command wrapped in "
    "<tool_call> and </tool_call> markers.\n"
    "Example reply:\n<tool_call>echo hello</tool_call>\n"
    "Never write a command without these markers. Never explain.\n"
    "After you receive the command output, reply with only the final answer."
)

# Policy message paired with the model's own native tool schema in native mode.
NATIVE_TOOL_SYSTEM_MESSAGE = (
    "You are a terminal agent connected to a sandboxed shell.\n"
    "Use the run_command tool to run shell commands when asked.\n"
    "After you receive the command output, reply with only the final answer."
)

RUN_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Run a shell command inside the sandboxed task environment "
            "and return its combined stdout/stderr output."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
        },
    },
}

_TOOL_CALL_TAG_OPEN_RE = pcre.compile(r"<tool_call>", pcre.IGNORECASE)
_TOOL_CALL_TAG_CLOSE_RE = pcre.compile(r"</tool_call>", pcre.IGNORECASE)
_FENCE_RE = pcre.compile(r"```([^\n`]*)\n(.*?)```", pcre.DOTALL)
_CONSOLE_PROMPT_RE = pcre.compile(r"(?m)^\$\s+")
_PYTHON_TAG_RE = pcre.compile(r"^\s*<\|python_tag\|>\s*")
_NATIVE_XML_RE = pcre.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", pcre.DOTALL)
_SPECIAL_TOKEN_RE = pcre.compile(r"<\|[^>]*\|>")

_SHELL_FENCE_LANGUAGES = frozenset(
    {"", "bash", "sh", "shell", "zsh", "ash", "console", "terminal"}
)


def validate_tool_call_format(tool_call_format: str) -> str:
    """Raise ``ValueError`` for unknown tool-call protocols."""
    if tool_call_format not in TOOL_CALL_FORMATS:
        raise ValueError(
            f"unknown tool_call_format {tool_call_format!r}; "
            f"expected one of {', '.join(TOOL_CALL_FORMATS)}"
        )
    return tool_call_format


def validate_tool_call_mode(tool_call_mode: str) -> str:
    """Raise ``ValueError`` for unknown tool-call modes."""
    if tool_call_mode not in TOOL_CALL_MODES:
        raise ValueError(
            f"unknown tool_call_mode {tool_call_mode!r}; "
            f"expected one of {', '.join(TOOL_CALL_MODES)}"
        )
    return tool_call_mode


def _tool_call_tag_commands(text: str) -> list[str]:
    """Capture every ``<tool_call>...</tool_call>`` action marker, in order.

    Only strict action markers are captured; ``<bash>``-style code output and
    fenced snippets are never matched by this protocol. A truncated final call
    (opening marker without a close, cut off at the generation stop) still
    counts: the opening marker is what makes it an explicit action request.
    """
    commands = []
    cursor = 0
    while True:
        # pcre patterns do not expose re-style search(pos=...), so scan slices
        # and shift offsets manually.
        open_match = _TOOL_CALL_TAG_OPEN_RE.search(text[cursor:])
        if not open_match:
            break
        body_start = cursor + open_match.end()
        close_match = _TOOL_CALL_TAG_CLOSE_RE.search(text[body_start:])
        if close_match:
            body_end = body_start + close_match.start()
            cursor = body_start + close_match.end()
        else:
            # Unterminated final call: run to end of generation.
            body_end = len(text)
            cursor = len(text)
        body = _SPECIAL_TOKEN_RE.sub("", text[body_start:body_end])
        command = _CONSOLE_PROMPT_RE.sub("", body).strip()
        if command:
            commands.append(command)
    return commands


def _fenced_shell_commands(text: str) -> list[str]:
    """Capture shell-language fenced blocks; other languages stay inert."""
    commands = []
    for match in _FENCE_RE.finditer(text):
        language = match.group(1).strip().lower()
        if language not in _SHELL_FENCE_LANGUAGES:
            continue
        # Strip `$ `-style console prompts so extracted strings are runnable.
        command = _CONSOLE_PROMPT_RE.sub("", match.group(2)).strip()
        if command:
            commands.append(command)
    return commands


def extract_tool_calls(text: str, tool_call_format: str) -> list[str]:
    """Return every tool call in ``text`` under the declared protocol.

    Plain prose, ordinary code output, and undeclared formats are never tool
    calls, which keeps generated code out of the execution path.
    """
    validate_tool_call_format(tool_call_format)
    if tool_call_format == TOOL_CALL_TAGS:
        return _tool_call_tag_commands(text)
    return _fenced_shell_commands(text)


def try_extract_tool_call(text: str, tool_call_format: str) -> str | None:
    """Return the first tool call under the protocol, or ``None`` if none."""
    commands = extract_tool_calls(text, tool_call_format)
    return commands[0] if commands else None


def _balanced_json_objects(text: str) -> list[str]:
    """Extract top-level ``{...}`` substrings with brace/string awareness."""
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def _decode_lenient_json(payload: str) -> Any:
    """Decode JSON, repairing single-backslash escapes models emit (``\\$``)."""
    import json
    import re

    try:
        return json.loads(payload)
    except ValueError:
        pass
    repaired = re.sub(r"\\(?![\"\\/bfnrtu])", "", payload)
    try:
        return json.loads(repaired)
    except ValueError:
        return None


_NATIVE_BLOCK_RE = pcre.compile(r"<tool_call>(.*?)</tool_call>", pcre.DOTALL)
_ARG_PAIR_RE = pcre.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    pcre.DOTALL | pcre.IGNORECASE,
)
_INVOKE_RE = pcre.compile(
    r'<invoke\s+name="([^"]+)"\s*>(.*?)</invoke>', pcre.DOTALL | pcre.IGNORECASE
)
_PARAM_RE = pcre.compile(r"<([A-Za-z_][\w.-]*)>(.*?)</\1>", pcre.DOTALL)


def _command_from_native_body(body: str) -> str | None:
    """Decode one native tool-call body (JSON, GLM/Laguna XML, MiniMax invoke).

    Returns the shell command when the body references a ``command`` argument;
    anything else is not a runnable action request.
    """
    # 1) JSON payloads: Hermes/Qwen style {"name": ..., "parameters"/"arguments": {...}}
    for payload in _balanced_json_objects(body):
        parsed = _decode_lenient_json(payload)
        if isinstance(parsed, dict):
            arguments = parsed.get("parameters", parsed.get("arguments"))
            command = arguments.get("command") if isinstance(arguments, dict) else None
            if isinstance(command, str) and command.strip():
                return command.strip()

    # 2) GLM / Laguna XML arguments:
    #    run_command<arg_key>command</arg_key><arg_value>ls</arg_value>
    pairs = {
        key.strip().lower(): value.strip()
        for key, value in _ARG_PAIR_RE.findall(body)
    }
    if pairs.get("command"):
        return pairs["command"]

    # 3) MiniMax-style <invoke name="..."><param>value</param></invoke>
    for _name, inner in _INVOKE_RE.findall(body):
        params = {
            tag.strip().lower(): value.strip()
            for tag, value in _PARAM_RE.findall(inner)
        }
        if params.get("command"):
            return params["command"]

    return None


def native_tool_commands(text: str) -> list[str]:
    """Parse model-native tool-call responses into shell commands.

    Covers the encodings observed across current open-model families:
    Llama ``<|python_tag|>{...}``, Hermes/Qwen JSON inside
    ``<tool_call></tool_call>``, GLM-5.2 and Laguna-S-2.1 XML
    ``arg_key``/``arg_value`` bodies, MiniMax-M3 ``<invoke>`` blocks, and bare
    JSON objects carrying ``name`` plus ``parameters``/``arguments``.
    Slightly malformed JSON (for example ``\\$ `` escapes around shell
    variables) is repaired before decoding.
    """
    if not text:
        return []

    candidates: list[str] = []
    stripped = _PYTHON_TAG_RE.sub("", text)

    segments = [stripped]
    for block_match in _NATIVE_BLOCK_RE.finditer(stripped):
        segments.append(block_match.group(1))

    for segment in segments:
        command = _command_from_native_body(segment)
        if command:
            candidates.append(command)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique = [command for command in candidates if not (command in seen or seen.add(command))]
    return unique


def session_supports_native_tool_calls(session: Any) -> bool:
    """Detect whether the session's tokenizer can render native tool schemas."""
    tokenizer = getattr(session, "tokenizer", None)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        return False
    probe_messages = [{"role": "user", "content": "probe"}]
    try:
        rendered = apply_template(
            probe_messages,
            tools=[RUN_COMMAND_TOOL],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:  # noqa: BLE001 — any template failure means "not native"
        return False
    return isinstance(rendered, str) and bool(rendered)
