# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import asyncio
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evalution.config import Model
from evalution.engines.base import (
    GenerationRequest,
    LoglikelihoodRequest,
    RollingLoglikelihoodRequest,
)
from evalution.engines.sglang_engine import (
    SGLang,
    SGLangSession,
    _SGLangPythonClient,
    _build_sglang_client,
    _normalize_sglang_response,
)

# Keep shared test fixtures and expectations explicit at module scope.
_TINYLLAMA_GPTQ_MODEL = Path("/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit")
# Skip the CUDA integration smoke test when the optional runtime is not installed locally.
_HAS_SGLANG_RUNTIME = importlib.util.find_spec("sglang") is not None


class FakeTokenizer:
    """Minimal tokenizer test double that makes SGLang payloads easy to assert against."""

    # Keep the class-level test state explicit for the surrounding assertions.
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"
    eos_token_id = 2
    bos_token_id = 1
    unk_token = "<unk>"
    padding_side = "left"
    model_max_length = 2048

    def __call__(self, text, add_special_tokens=True, padding=False):
        """Implement call for fake tokenizer."""
        del padding
        if isinstance(text, list):
            return {"input_ids": [self._encode(item, add_special_tokens) for item in text]}
        return {"input_ids": self._encode(text, add_special_tokens)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Implement apply chat template for fake tokenizer."""
        del tokenize, add_generation_prompt
        return "".join(message["content"] for message in messages)

    def decode(self, ids, skip_special_tokens=False):
        """Implement decode for fake tokenizer."""
        del skip_special_tokens
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(96 + token_id) for token_id in ids if token_id > 0)

    def _encode(self, text: str, add_special_tokens: bool) -> list[int]:
        """Implement encode for fake tokenizer."""
        tokens = [ord(char) - 96 for char in text]
        if add_special_tokens:
            return [self.bos_token_id] + tokens
        return tokens


def test_sglang_engine_defaults_batch_size_to_auto() -> None:
    """Verify the SGLang engine exposes stable defaults for config serialization."""

    engine = SGLang()

    assert engine.batch_size == "auto"
    assert engine.max_new_tokens == 256
    assert engine.base_url is None
    assert engine.tp_size == 1
    assert engine.dp_size == 1
    assert engine.pp_size == 1
    assert engine.to_dict()["max_new_tokens"] == 256
    assert engine.to_dict()["base_url"] is None


def test_normalize_sglang_response_accepts_batched_dict_shape() -> None:
    """SGLang sometimes batches `text` and `meta_info` into one dict payload."""

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
    """In-process generation should return only the completion text to Evalution."""

    payloads: list[dict[str, object]] = []

    class FakeClient:
        """Provide the fake client helper used by the surrounding tests."""
        def generate(self, **payload):
            """Generate generate."""
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
            """Release reusable intermediate state for this object."""
            return None

        def close(self) -> None:
            """Release the resources owned by this object."""
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


def test_sglang_session_generate_continuous_yields_completion_order() -> None:
    """Continuous mode should yield by completion order, not submission order."""

    payloads: list[dict[str, object]] = []

    class FakeClient:
        """Provide the fake client helper used by the surrounding tests."""
        def generate(self, **payload):
            """Generate generate."""
            del payload
            raise AssertionError("generate() should not be used for continuous generation")

        async def async_generate(self, **payload):
            """Implement async generate for fake client."""
            payloads.append(payload)
            input_ids = list(payload["input_ids"])
            await asyncio.sleep(0.02 if input_ids[-1] == 2 else 0.0)
            return {
                "text": "".join(chr(96 + token_id) for token_id in input_ids) + "z",
                "meta_info": {
                    "id": f"req-{input_ids[-1]}",
                },
            }

        def gc(self) -> None:
            """Release reusable intermediate state for this object."""
            return None

        def close(self) -> None:
            """Release the resources owned by this object."""
            return None

    session = SGLangSession(
        config=SGLang(batch_size="auto"),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048)),
        tokenizer=FakeTokenizer(),
        prepare_tokenizer=None,
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="sglang.generate",
        client=FakeClient(),
    )

    outputs = list(
        session.generate_continuous(
            [
                ("slow", GenerationRequest(prompt="ab", metadata={"slot": 1})),
                ("fast", GenerationRequest(prompt="ac", metadata={"slot": 2})),
            ]
        )
    )

    assert payloads == [
        {
            "input_ids": [1, 2],
            "sampling_params": {"max_new_tokens": 256, "temperature": 0.0},
        },
        {
            "input_ids": [1, 3],
            "sampling_params": {"max_new_tokens": 256, "temperature": 0.0},
        },
    ]
    assert [item_id for item_id, _ in outputs] == ["fast", "slow"]
    assert outputs[0][1].prompt == "ac"
    assert outputs[0][1].text == "z"
    assert outputs[0][1].metadata["slot"] == 2
    assert outputs[0][1].metadata["sglang_meta"]["id"] == "req-3"
    assert outputs[1][1].prompt == "ab"
    assert outputs[1][1].text == "z"
    assert outputs[1][1].metadata["slot"] == 1
    assert outputs[1][1].metadata["sglang_meta"]["id"] == "req-2"


def test_sglang_session_generate_continuous_refills_open_slots_immediately() -> None:
    """Continuous mode should submit queued work as soon as one in-flight request finishes."""

    event_log: list[str] = []

    class FakeClient:
        """Provide the fake client helper used by the surrounding tests."""
        def generate(self, **payload):
            """Generate generate."""
            del payload
            raise AssertionError("generate() should not be used for continuous generation")

        async def async_generate(self, **payload):
            """Implement async generate for fake client."""
            input_ids = list(payload["input_ids"])
            prompt_text = "".join(chr(96 + token_id) for token_id in input_ids)
            event_log.append(f"start:{prompt_text}")
            await asyncio.sleep(0.05 if prompt_text == "ab" else 0.0)
            event_log.append(f"end:{prompt_text}")
            return {
                "text": prompt_text + "z",
                "meta_info": {
                    "id": f"req-{prompt_text}",
                },
            }

        def gc(self) -> None:
            """Release reusable intermediate state for this object."""
            return None

        def close(self) -> None:
            """Release the resources owned by this object."""
            return None

    session = SGLangSession(
        config=SGLang(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048)),
        tokenizer=FakeTokenizer(),
        prepare_tokenizer=None,
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="sglang.generate",
        client=FakeClient(),
    )

    outputs = list(
        session.generate_continuous(
            [
                ("slow", GenerationRequest(prompt="ab")),
                ("fast-1", GenerationRequest(prompt="ac")),
                ("fast-2", GenerationRequest(prompt="ad")),
                ("fast-3", GenerationRequest(prompt="ae")),
            ],
            batch_size=2,
        )
    )

    assert [request_id for request_id, _output in outputs] == ["fast-1", "fast-2", "fast-3", "slow"]
    assert event_log.index("start:ad") < event_log.index("end:ab")
    assert event_log.index("start:ae") < event_log.index("end:ab")


def test_sglang_session_loglikelihood_uses_in_process_token_scores() -> None:
    """Score continuations from SGLang prompt-logprob metadata."""

    payloads: list[dict[str, object]] = []

    class FakeClient:
        """Provide the fake client helper used by the surrounding tests."""
        def generate(self, **payload):
            """Generate generate."""
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
            """Release reusable intermediate state for this object."""
            return None

        def close(self) -> None:
            """Release the resources owned by this object."""
            return None

    session = SGLangSession(
        config=SGLang(batch_size="auto"),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048)),
        tokenizer=FakeTokenizer(),
        prepare_tokenizer=None,
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="sglang.generate",
        client=FakeClient(),
    )

    outputs = session.loglikelihood([LoglikelihoodRequest(context="ab", continuation="cd")])

    assert payloads == [
        {
            "input_ids": [[1, 2, 3, 4]],
            "sampling_params": [{"max_new_tokens": 1, "temperature": 0.0}],
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 2,
            "token_ids_logprob": [[3, 4]],
        }
    ]
    assert outputs[0].logprob == -0.30000000000000004
    assert outputs[0].is_greedy is True
    assert outputs[0].token_count == 2


def test_sglang_session_loglikelihood_continuous_uses_auto_batch_size() -> None:
    """Continuous scoring should batch lazily submitted requests without requiring an explicit size."""

    payloads: list[dict[str, object]] = []

    class FakeClient:
        """Provide the fake client helper used by the surrounding tests."""
        def generate(self, **payload):
            """Generate generate."""
            payloads.append(payload)
            responses = []
            for row in payload["input_ids"]:
                responses.append(
                    {
                        "text": "",
                        "meta_info": {
                            "input_token_logprobs": [
                                (None, row[0], None),
                                (-0.5, row[1], None),
                                (-0.25, row[2], None),
                            ],
                            "input_top_logprobs": [
                                [],
                                [(-0.5, row[1], None)],
                                [(-0.25, row[2], None)],
                            ],
                            "input_token_ids_logprobs": [
                                [],
                                [(-0.5, row[1], None)],
                                [(-0.25, row[2], None)],
                            ],
                        },
                    }
                )
            return responses

        def gc(self) -> None:
            """Release reusable intermediate state for this object."""
            return None

        def close(self) -> None:
            """Release the resources owned by this object."""
            return None

    session = SGLangSession(
        config=SGLang(batch_size="auto"),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048)),
        tokenizer=FakeTokenizer(),
        prepare_tokenizer=None,
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="sglang.generate",
        client=FakeClient(),
    )

    outputs = list(
        session.loglikelihood_continuous(
            [
                ("first", LoglikelihoodRequest(context="ab", continuation="c", metadata={"slot": 1})),
                ("second", LoglikelihoodRequest(context="ab", continuation="d", metadata={"slot": 2})),
            ]
        )
    )

    assert payloads == [
        {
            "input_ids": [[1, 2, 3], [1, 2, 4]],
            "sampling_params": [
                {"max_new_tokens": 1, "temperature": 0.0},
                {"max_new_tokens": 1, "temperature": 0.0},
            ],
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 2,
            "token_ids_logprob": [[3], [4]],
        }
    ]
    assert [request_id for request_id, _output in outputs] == ["first", "second"]
    assert outputs[0][1].logprob == -0.25
    assert outputs[0][1].token_count == 1
    assert outputs[0][1].metadata["slot"] == 1
    assert outputs[1][1].logprob == -0.25
    assert outputs[1][1].token_count == 1
    assert outputs[1][1].metadata["slot"] == 2


def test_sglang_session_loglikelihood_preserves_monkey_patched_logits() -> None:
    """Greedy checks should still work when only patched logit views are present."""

    class FakeClient:
        """Provide the fake client helper used by the surrounding tests."""
        def generate(self, **payload):
            """Generate generate."""
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
            """Release reusable intermediate state for this object."""
            return None

        def close(self) -> None:
            """Release the resources owned by this object."""
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
    )

    output = session.loglikelihood([LoglikelihoodRequest(context="ab", continuation="c")])[0]

    assert output.logprob == -0.4
    assert output.is_greedy is False
    assert output.token_count == 1


def test_sglang_session_loglikelihood_rolling_aggregates_window_scores() -> None:
    """Rolling scoring should reuse ordinary scoring and sum scores across windows."""

    payloads: list[dict[str, object]] = []

    class FakeClient:
        """Provide the fake client helper used by the surrounding tests."""
        def generate(self, **payload):
            """Generate generate. Keep the nested traversal explicit so ordering and metadata stay aligned."""
            payloads.append(payload)
            rows = list(payload["input_ids"])
            responses = []
            for row in rows:
                meta_entries = [(None, row[0], None)]
                top_entries = [[]]
                requested_entries = [[]]
                for token_id in row[1:]:
                    meta_entries.append((-0.25, token_id, None))
                    top_entries.append([(-0.25, token_id, None)])
                    requested_entries.append([(-0.25, token_id, None)])
                responses.append(
                    {
                        "text": "",
                        "meta_info": {
                            "input_token_logprobs": meta_entries,
                            "input_top_logprobs": top_entries,
                            "input_token_ids_logprobs": requested_entries,
                        },
                    }
                )
            return responses

        def gc(self) -> None:
            """Release reusable intermediate state for this object."""
            return None

        def close(self) -> None:
            """Release the resources owned by this object."""
            return None

    session = SGLangSession(
        config=SGLang(batch_size=2),
        model_config=Model(path="/tmp/model"),
        model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4)),
        tokenizer=FakeTokenizer(),
        prepare_tokenizer=None,
        input_device=SimpleNamespace(type="cpu"),
        generation_backend="sglang.generate",
        client=FakeClient(),
    )

    outputs = session.loglikelihood_rolling(
        [
            RollingLoglikelihoodRequest(
                text="unused",
                input_ids=[2, 3, 4, 5],
                metadata={"suite": "rolling"},
            )
        ],
        batch_size=2,
    )

    assert len(outputs) == 1
    assert outputs[0].token_count == 4
    assert outputs[0].logprob == pytest.approx(-1.0)
    assert outputs[0].metadata["suite"] == "rolling"
    assert payloads == [
        {
            "input_ids": [[1, 2, 3, 4, 5]],
            "sampling_params": [{"max_new_tokens": 1, "temperature": 0.0}],
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 2,
            "token_ids_logprob": [[2, 3, 4, 5]],
        }
    ]


def test_build_sglang_client_uses_python_engine_when_base_url_is_missing(monkeypatch) -> None:
    """SGLang should always build the local Engine client in Evalution."""

    fake_engine = object()

    monkeypatch.setattr(
        "evalution.engines.sglang_engine.importlib.import_module",
        lambda name: SimpleNamespace(Engine=lambda **kwargs: fake_engine)
        if name == "sglang.srt.entrypoints.engine"
        else __import__(name),
    )

    client = _build_sglang_client(
        SGLang(base_url=None),
        Model(path="/tmp/model"),
    )

    assert isinstance(client, _SGLangPythonClient)
    assert client.engine is fake_engine


def test_build_sglang_client_rejects_server_mode() -> None:
    """The legacy HTTP path is intentionally disabled for this backend."""

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
@pytest.mark.skipif(
    not _HAS_SGLANG_RUNTIME,
    reason="sglang runtime is not installed",
)
def test_sglang_engine_can_generate_and_score_on_cuda() -> None:
    """Smoke-test real generation and scoring against a local CUDA runtime."""

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
    execution = session.describe_execution()
    assert execution["generation_backend"] == "sglang.generate"
    assert execution["logprob_backend"] == "sglang.generate"
