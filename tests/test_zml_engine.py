# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import evalution
from evalution.config import Model
from evalution.engines.base import GenerationRequest
from evalution.engines.zml_engine import ZML, ZMLSession, _launch_server


def test_zml_defaults_expose_llmd_optimized_serving_contract() -> None:
    """Keep the ZML engine defaults aligned with LLMD's optimized server path."""

    engine = ZML()

    assert engine.batch_size == 16
    assert engine.max_parallel_requests == 64
    assert engine.continuous_batching is True
    assert engine.paged_attention is True
    assert engine.prefix_caching is True
    assert engine.attention_backend == "auto"
    assert engine.tensor_parallel_size == "auto"
    assert engine.launch_server is False


def test_zml_is_available_from_yaml_and_public_package() -> None:
    """Make sure Python and YAML users resolve the same public engine."""

    from evalution.yaml import _build_engine

    engine = _build_engine(
        {
            "type": "ZML",
            "batch_size": 8,
            "attention_backend": "auto",
            "paged_attention": True,
        }
    )

    assert isinstance(engine, evalution.ZML)
    assert engine.batch_size == 8


def test_zml_session_reuses_openai_chat_translation(monkeypatch) -> None:
    """Translate Evalution chat requests to the LLMD OpenAI-compatible route."""

    session = ZML().build(Model(path="/monster/data/model/Llama-3.2-1B-Instruct"))
    assert isinstance(session, ZMLSession)
    payloads: list[tuple[str, dict[str, object]]] = []

    def post_json(
        _session: ZMLSession,
        path: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        payloads.append((path, payload))
        return {
            "id": "zml-test",
            "model": "Llama-3.2-1B-Instruct",
            "choices": [{"message": {"role": "assistant", "content": "42"}}],
        }

    monkeypatch.setattr(ZMLSession, "_post_json", post_json)
    try:
        outputs = session.generate(
            [
                GenerationRequest(
                    messages=[{"role": "user", "content": "What is 6 times 7?"}],
                    max_new_tokens=12,
                )
            ]
        )
    finally:
        session.close()

    assert outputs[0].text == "42"
    assert outputs[0].metadata["openai_response"]["id"] == "zml-test"
    assert payloads[0][0] == "/v1/chat/completions"
    assert payloads[0][1]["model"] == "Llama-3.2-1B-Instruct"
    assert payloads[0][1]["max_tokens"] == 12
    assert session.describe_execution()["generation_backend"] == "zml_llmd_openai_http"


def test_launch_server_builds_local_llmd_command(monkeypatch) -> None:
    """Add the model and documented DFlash option without hiding raw server args."""

    calls: list[tuple[list[str], dict[str, str]]] = []

    class FakeProcess:
        """Provide the small process surface used by the launcher test."""

        def poll(self) -> None:
            return None

    def popen(command, *, env):
        calls.append((command, env))
        return FakeProcess()

    monkeypatch.setattr("evalution.engines.zml_engine.subprocess.Popen", popen)
    process = _launch_server(
        ZML(
            executable="llmd-test",
            dflash_model="/models/draft",
            model_name="llama-test",
            batch_size=8,
            token_batch_size=512,
            max_context_len=4096,
            prefill_chunk_size=128,
            page_chunk_size=16,
            attention_backend="cuda_fa3",
            gpu_memory_fraction=0.8,
            cache_memory_fraction=0.7,
            topk=20,
            dflash_draft_count=4,
            base_url="http://127.0.0.1:18000",
            server_args=["--max-batch-size=16"],
            environment={"ZML_TEST_FLAG": "1"},
        ),
        Model(path="/models/Llama-3.2-1B-Instruct"),
    )

    assert isinstance(process, FakeProcess)
    assert calls[0][0] == [
        "llmd-test",
        "--model=/models/Llama-3.2-1B-Instruct",
        "--batch-size=8",
        "--token-batch-size=512",
        "--max-context-len=4096",
        "--prefill-chunk-size=128",
        "--page-chunk-size=16",
        "--backend=cuda_fa3",
        "--gpu-memory-fraction=0.8",
        "--cache-memory-fraction=0.7",
        "--topk=20",
        "--listen=127.0.0.1:18000",
        "--model-name=llama-test",
        "--dflash-model=/models/draft",
        "--dflash-draft-count=4",
        "--max-batch-size=16",
    ]
    assert calls[0][1]["ZML_TEST_FLAG"] == "1"
