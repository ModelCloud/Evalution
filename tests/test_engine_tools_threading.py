# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Per-engine verification that native tool schemas reach model rendering.

The agentic tool loop is engine-agnostic (it only drives
``session.generate``), but each engine must forward
``GenerationRequest.tools`` into its own chat-template or API path. These
tests exercise every engine's render/payload path with lightweight doubles so
no engine can silently drop the tool schema.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from evalution.benchmarks.tool_calling import RUN_COMMAND_TOOL, native_tool_commands
from evalution.engines.base import GenerationRequest


class RecordingChatTemplateTokenizer:
    """apply_chat_template double that records every kwargs invocation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str | list[int]:
        self.calls.append(kwargs)
        return [1, 2, 3] if kwargs.get("tokenize") else "<rendered>"


def _chat_request() -> GenerationRequest:
    return GenerationRequest(
        messages=[{"role": "user", "content": "probe"}],
        tools=[RUN_COMMAND_TOOL],
        add_generation_prompt=True,
        max_new_tokens=8,
    )


def _bare_session(session_cls: type, **attributes: Any) -> Any:
    """Instantiate a session without running its heavyweight build()."""
    session = object.__new__(session_cls)
    for name, value in attributes.items():
        setattr(session, name, value)
    return session


def _assert_tools_forwarded(session: Any) -> None:
    tokenizer = RecordingChatTemplateTokenizer()
    rendered = session._render_request_with_tokenizer(tokenizer, _chat_request())
    assert rendered == "<rendered>"
    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0]["tools"] == [RUN_COMMAND_TOOL]


def test_transformers_common_forwards_tools() -> None:
    from evalution.engines.transformers_common import BaseTransformerSession

    session = _bare_session(
        BaseTransformerSession, model=SimpleNamespace(generation_config=None)
    )
    _assert_tools_forwarded(session)


def test_vllm_forwards_tools() -> None:
    from evalution.engines.vllm_engine import VLLMSession

    _assert_tools_forwarded(_bare_session(VLLMSession))


def test_sglang_forwards_tools() -> None:
    from evalution.engines.sglang_engine import SGLangSession

    _assert_tools_forwarded(_bare_session(SGLangSession))


def test_tensorrt_llm_forwards_tools() -> None:
    from evalution.engines.tensorrt_llm_engine import TensorRTLLMSession

    _assert_tools_forwarded(_bare_session(TensorRTLLMSession))


def test_tinygrad_render_forwards_tools() -> None:
    from evalution.engines.tinygrad_engine import TinygradSession

    _assert_tools_forwarded(_bare_session(TinygradSession))


def test_tinygrad_tokenize_path_forwards_tools() -> None:
    """The tokenize=True chat encoding path also carries the schema."""
    from evalution.engines.tinygrad_engine import TinygradSession

    tokenizer = RecordingChatTemplateTokenizer()

    def apply_with_ids(messages: Any, **kwargs: Any) -> list[int]:
        tokenizer.calls.append(kwargs)
        return [1, 2, 3]

    tokenizer.apply_chat_template = apply_with_ids  # type: ignore[method-assign]
    session = _bare_session(TinygradSession)
    token_ids = session._tokenize_chat_messages_with_tokenizer(tokenizer, _chat_request())

    assert token_ids == [1, 2, 3]
    assert tokenizer.calls[0]["tools"] == [RUN_COMMAND_TOOL]


def test_llama_cpp_tokenizer_path_forwards_tools() -> None:
    from evalution.engines.llama_cpp_engine import LlamaCppSession

    tokenizer = RecordingChatTemplateTokenizer()
    session = _bare_session(LlamaCppSession, prepare_tokenizer=tokenizer)
    session._tokenize_text = lambda text, add_bos=True: [1, 2, 3]
    prompt_text, _prompt_tokens = session._prepare_generation_prompt(_chat_request())

    assert prompt_text == "<rendered>"
    assert tokenizer.calls[0]["tools"] == [RUN_COMMAND_TOOL]


def test_llama_cpp_native_chat_api_passes_and_serializes_tool_calls() -> None:
    """llama.cpp create_chat_completion receives tools; tool_calls become text."""
    from evalution.engines.llama_cpp_engine import LlamaCppSession

    captured: dict[str, Any] = {}

    class FakeLlama:
        def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "run_command",
                                        "arguments": json.dumps({"command": "echo hi"}),
                                    }
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {},
            }

    session = _bare_session(
        LlamaCppSession,
        llm=FakeLlama(),
        prepare_tokenizer=None,
        config=SimpleNamespace(seed=None),
    )
    output = session._generate_one(_chat_request())

    assert captured["tools"] == [RUN_COMMAND_TOOL]
    # Native tool-call responses are serialized back into parseable text.
    assert native_tool_commands(output.text) == ["echo hi"]


def test_openai_payload_carries_tools_and_tool_calls_become_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evalution.engines.openai_engine import OpenAICompatibleSession

    captured_payloads: list[dict[str, Any]] = []

    def fake_post_json(self: Any, endpoint_path: str, payload: dict[str, Any]) -> dict[str, Any]:
        captured_payloads.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "run_command",
                                    "arguments": json.dumps({"command": "echo hi"}),
                                }
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr(OpenAICompatibleSession, "_post_json", fake_post_json)
    session = _bare_session(
        OpenAICompatibleSession,
        model_name="test-model",
        config=SimpleNamespace(chat_completions_path="/v1/chat/completions"),
    )
    output = session._generate_one(_chat_request())

    assert captured_payloads[0]["tools"] == [RUN_COMMAND_TOOL]
    assert native_tool_commands(output.text) == ["echo hi"]


def test_shared_renderer_is_inherited_by_wrapper_sessions() -> None:
    """GPTQModel/OpenVINO/TransformersCompat reuse the patched renderer."""
    from evalution.engines.gptqmodel_engine import GPTQModelSession
    from evalution.engines.openvino_engine import OpenVINOSession
    from evalution.engines.transformers_common import BaseTransformerSession
    from evalution.engines.transformers_compat import TransformersCompatSession

    expected = BaseTransformerSession._render_request_with_tokenizer
    for session_cls in (GPTQModelSession, OpenVINOSession, TransformersCompatSession):
        assert session_cls._render_request_with_tokenizer is expected


def test_transformers_common_merges_chat_template_kwargs() -> None:
    """Request-level template kwargs merge over defaults alongside tools."""
    from evalution.engines.transformers_common import BaseTransformerSession

    session = _bare_session(
        BaseTransformerSession, model=SimpleNamespace(generation_config=None)
    )
    request = GenerationRequest(
        messages=[{"role": "user", "content": "probe"}],
        tools=[RUN_COMMAND_TOOL],
        chat_template_kwargs={"enable_thinking": False},
        add_generation_prompt=True,
    )
    tokenizer = RecordingChatTemplateTokenizer()
    rendered = session._render_request_with_tokenizer(tokenizer, request)

    assert rendered == "<rendered>"
    assert tokenizer.calls[0]["tools"] == [RUN_COMMAND_TOOL]
    assert tokenizer.calls[0]["enable_thinking"] is False
