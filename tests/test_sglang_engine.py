# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evalution.config import Model
from evalution.engines.base import GenerationRequest, LoglikelihoodRequest
from evalution.engines.sglang_engine import (
    SGLang,
    SGLangSession,
    _SGLangPythonClient,
    _build_sglang_client,
    _normalize_sglang_response,
)

_TINYLLAMA_GPTQ_MODEL = Path("/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit")


class FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"
    eos_token_id = 2
    bos_token_id = 1
    unk_token = "<unk>"
    padding_side = "left"
    model_max_length = 2048

    def __call__(self, text, add_special_tokens=True, padding=False):
        del padding
        if isinstance(text, list):
            return {"input_ids": [self._encode(item, add_special_tokens) for item in text]}
        return {"input_ids": self._encode(text, add_special_tokens)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return "".join(message["content"] for message in messages)

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(96 + token_id) for token_id in ids if token_id > 0)

    def _encode(self, text: str, add_special_tokens: bool) -> list[int]:
        tokens = [ord(char) - 96 for char in text]
        if add_special_tokens:
            return [self.bos_token_id] + tokens
        return tokens


def test_sglang_engine_defaults_batch_size_to_auto() -> None:
    engine = SGLang()

    assert engine.batch_size == "auto"
    assert engine.base_url is None
    assert engine.transport == "python"
    assert engine.default_auto_batch_size == 32
    assert engine.to_dict()["base_url"] is None


def test_normalize_sglang_response_accepts_batched_dict_shape() -> None:
    normalized = _normalize_sglang_response(
        {
            "text": ["abx", "aby"],
            "meta_info": [{"id": "r1"}, {"id": "r2"}],
        }
    )

    assert normalized == [
        {"text": "abx", "meta_info": {"id": "r1"}},
        {"text": "aby", "meta_info": {"id": "r2"}},
    ]


def test_sglang_session_generate_uses_in_process_engine_and_strips_prompt() -> None:
    payloads: list[dict[str, object]] = []

    class FakeClient:
        transport = "python"
        supports_raw_logits = False

        def generate(self, **payload):
            payloads.append(payload)
            return [
                {
                    "text": "abx",
                    "meta_info": {
                        "id": "req-1",
                        "completion_tokens": 1,
                    },
                }
            ]

        def gc(self) -> None:
            return None

        def close(self) -> None:
            return None

    session = SGLangSession(
        config=SGLang(batch_size=4),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048)),
        tokenizer=FakeTokenizer(),
        prepare_tokenizer=None,
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="sglang.generate",
        client=FakeClient(),
        transport="python",
    )

    outputs = session.generate([GenerationRequest(prompt="ab", metadata={"suite": "demo"})])

    assert payloads == [
        {
            "input_ids": [[1, 2]],
            "sampling_params": [{"max_new_tokens": 256, "temperature": 0.0}],
        }
    ]
    assert outputs[0].prompt == "ab"
    assert outputs[0].text == "x"
    assert outputs[0].metadata["suite"] == "demo"
    assert outputs[0].metadata["sglang_meta"]["id"] == "req-1"


def test_sglang_session_loglikelihood_uses_in_process_token_scores() -> None:
    payloads: list[dict[str, object]] = []

    class FakeClient:
        transport = "python"
        supports_raw_logits = False

        def generate(self, **payload):
            payloads.append(payload)
            return [
                {
                    "text": "",
                    "meta_info": {
                        "input_token_logprobs": [
                            (None, 1, None),
                            (-0.7, 2, None),
                            (-0.2, 3, None),
                            (-0.1, 4, None),
                        ],
                        "input_top_logprobs": [
                            [],
                            [(-0.7, 2, None)],
                            [(-0.2, 3, None)],
                            [(-0.1, 4, None)],
                        ],
                        "input_token_ids_logprobs": [
                            [],
                            [],
                            [(-0.2, 3, None), (-1.2, 4, None)],
                            [(-0.9, 3, None), (-0.1, 4, None)],
                        ],
                    },
                }
            ]

        def gc(self) -> None:
            return None

        def close(self) -> None:
            return None

    session = SGLangSession(
        config=SGLang(batch_size=8),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048)),
        tokenizer=FakeTokenizer(),
        prepare_tokenizer=None,
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="sglang.generate",
        client=FakeClient(),
        transport="python",
    )

    outputs = session.loglikelihood([LoglikelihoodRequest(context="ab", continuation="cd")])

    assert payloads == [
        {
            "input_ids": [[1, 2, 3, 4]],
            "sampling_params": [{"max_new_tokens": 1, "temperature": 0.0}],
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 2,
        }
    ]
    assert outputs[0].logprob == -0.30000000000000004
    assert outputs[0].is_greedy is True
    assert outputs[0].token_count == 2
    assert outputs[0].metadata["sglang_transport"] == "python"


def test_sglang_session_loglikelihood_preserves_monkey_patched_logits() -> None:
    class FakeClient:
        transport = "python"
        supports_raw_logits = True

        def generate(self, **payload):
            del payload
            return [
                {
                    "text": "",
                    "meta_info": {
                        "input_token_logprobs": [(-0.4, 3, None)],
                        "input_top_logprobs": [[(-0.1, 5, None)]],
                        "input_token_ids_logprobs": [[(-0.4, 3, None), (-0.1, 5, None)]],
                        "input_token_ids_logits": [[(2.4, 3, None), (3.1, 5, None)]],
                        "input_token_logits": [(2.4, 3, None)],
                    },
                }
            ]

        def gc(self) -> None:
            return None

        def close(self) -> None:
            return None

    session = SGLangSession(
        config=SGLang(batch_size=1),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048)),
        tokenizer=FakeTokenizer(),
        prepare_tokenizer=None,
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="sglang.generate",
        client=FakeClient(),
        transport="python",
        raw_logits_enabled=True,
    )

    output = session.loglikelihood([LoglikelihoodRequest(context="ab", continuation="c")])[0]

    assert output.logprob == -0.4
    assert output.is_greedy is False
    assert output.metadata["raw_logits_enabled"] is True
    assert output.token_count == 1


def test_build_sglang_client_uses_python_engine_when_base_url_is_missing(monkeypatch) -> None:
    fake_engine = object()

    monkeypatch.setattr(
        "evalution.engines.sglang_engine._import_sglang",
        lambda path: object(),
    )
    monkeypatch.setattr(
        "evalution.engines.sglang_engine.importlib.import_module",
        lambda name: SimpleNamespace(Engine=lambda **kwargs: fake_engine)
        if name == "sglang.srt.entrypoints.engine"
        else __import__(name),
    )

    client = _build_sglang_client(
        SGLang(base_url=None, transport="python"),
        Model(path="/tmp/model"),
    )

    assert isinstance(client, _SGLangPythonClient)
    assert client.engine is fake_engine


def test_build_sglang_client_rejects_server_mode() -> None:
    try:
        _build_sglang_client(
            SGLang(base_url="http://127.0.0.1:30000"),
            Model(path="/tmp/model"),
        )
    except ValueError as exc:
        assert str(exc) == "sglang engine no longer supports server/http mode; use in-process Engine"
    else:
        raise AssertionError("expected server mode to be rejected")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _TINYLLAMA_GPTQ_MODEL.exists(),
    reason="local TinyLlama GPTQ weights are not available",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the SGLang engine integration test",
)
def test_sglang_engine_can_generate_and_score_on_cuda() -> None:
    session = SGLang(
        batch_size=1,
    ).build(Model(path=str(_TINYLLAMA_GPTQ_MODEL)))

    try:
        outputs = session.generate(
            [
                GenerationRequest(
                    prompt="The capital of France is",
                    max_new_tokens=8,
                )
            ],
            batch_size=1,
        )
        scores = session.loglikelihood(
            [
                LoglikelihoodRequest(
                    context="The capital of France is",
                    continuation=" Paris",
                )
            ],
            batch_size=1,
        )
    finally:
        session.close()

    assert len(outputs) == 1
    assert outputs[0].prompt == "The capital of France is"
    assert isinstance(outputs[0].text, str)
    assert len(scores) == 1
    assert scores[0].token_count > 0
    assert math.isfinite(scores[0].logprob)
    assert scores[0].metadata["sglang_transport"] == "python"
    execution = session.describe_execution()
    assert execution["generation_backend"] == "sglang.generate"
    assert execution["transport"] == "python"
    assert execution["logprob_backend"] == "sglang.generate"
