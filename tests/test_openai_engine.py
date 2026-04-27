# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

from evalution.config import Model
from evalution.engines.base import (
    BaseInferenceSession,
    GenerationOutput,
    GenerationRequest,
    LoglikelihoodOutput,
    LoglikelihoodRequest,
    RollingLoglikelihoodOutput,
    RollingLoglikelihoodRequest,
)
from evalution.engines.openai_engine import OpenAICompatible
from evalution.engines.openai_server import OpenAICompatibleServer


class FakeSession(BaseInferenceSession):
    """Small in-memory session used to exercise the HTTP engine and server."""

    # Keep the class-level counters explicit so the surrounding tests can assert batching.
    def __init__(self) -> None:
        """Initialize the fake session state used by the tests."""

        self.generate_batches: list[list[str]] = []
        self.loglikelihood_batches: list[int] = []
        self.rolling_batches: list[int] = []
        self.gc_calls = 0
        self.close_calls = 0

    def generate(self, requests, *, batch_size=None):
        """Return deterministic completions while recording server-side microbatches."""

        del batch_size
        batch_prompts: list[str] = []
        outputs: list[GenerationOutput] = []
        for request in requests:
            prompt = self._render_prompt(request)
            batch_prompts.append(prompt)
            if prompt == "slow":
                time.sleep(0.05)
            outputs.append(
                GenerationOutput(
                    prompt=prompt,
                    text=f"{prompt}->done",
                    metadata={"completion_tokens": len(prompt)},
                )
            )
        self.generate_batches.append(batch_prompts)
        return outputs

    def loglikelihood(self, requests, *, batch_size=None):
        """Return deterministic scores while recording batched request counts."""

        del batch_size
        self.loglikelihood_batches.append(len(requests))
        outputs: list[LoglikelihoodOutput] = []
        for request in requests:
            continuation = request.continuation_input_ids
            token_count = len(continuation) if continuation is not None else len(request.continuation)
            outputs.append(
                LoglikelihoodOutput(
                    logprob=float(-token_count),
                    is_greedy=token_count <= 4,
                    token_count=token_count,
                    metadata={"source": "fake"},
                )
            )
        return outputs

    def loglikelihood_rolling(self, requests, *, batch_size=None):
        """Return deterministic rolling scores while recording batched request counts."""

        del batch_size
        self.rolling_batches.append(len(requests))
        outputs: list[RollingLoglikelihoodOutput] = []
        for request in requests:
            token_count = len(request.input_ids) if request.input_ids is not None else len(request.text)
            outputs.append(
                RollingLoglikelihoodOutput(
                    logprob=float(-token_count) / 2.0,
                    token_count=token_count,
                    metadata={"source": "fake"},
                )
            )
        return outputs

    def generate_continuous(self, requests, *, batch_size=None):
        """Expose the fixed-batch generate implementation through the continuous contract."""

        del batch_size
        request_items = list(requests)
        self.generate_batches.append([self._render_prompt(request) for _, request in request_items])
        ordered_items = sorted(
            request_items,
            key=lambda item: 1 if self._render_prompt(item[1]) == "slow" else 0,
        )
        for request_key, request_item in ordered_items:
            prompt = self._render_prompt(request_item)
            if prompt == "slow":
                time.sleep(0.05)
            yield request_key, GenerationOutput(
                prompt=prompt,
                text=f"{prompt}->done",
                metadata={"completion_tokens": len(prompt)},
            )

    def gc(self) -> None:
        """Record that the session gc hook was called."""

        self.gc_calls += 1

    def close(self) -> None:
        """Record that the session close hook was called."""

        self.close_calls += 1

    def _render_prompt(self, request: GenerationRequest) -> str:
        """Normalize prompt-only and chat-style requests into one deterministic prompt string."""

        if request.prompt is not None:
            return request.prompt
        if request.messages is not None:
            return " | ".join(
                f"{message['role']}:{message['content']}" for message in request.messages
            )
        return ""


def test_openai_compatible_engine_generate_and_score_round_trip() -> None:
    """Verify the HTTP engine can generate and score through the local OpenAI-style server."""

    backend = FakeSession()
    with OpenAICompatibleServer(
        session=backend,
        model_name="fake-model",
        max_batch_size=8,
        batch_window_s=0.05,
    ) as server:
        engine = OpenAICompatible(
            base_url=server.base_url,
            batch_size=4,
        )
        session = engine.build(Model(path="fake-model"))
        try:
            generation_outputs = session.generate(
                [
                    GenerationRequest(prompt="alpha"),
                    GenerationRequest(messages=[{"role": "user", "content": "beta"}]),
                ],
                batch_size=2,
            )
            loglikelihood_outputs = session.loglikelihood(
                [
                    LoglikelihoodRequest(context="Q:", continuation=" yes"),
                    LoglikelihoodRequest(context_input_ids=[1, 2], continuation_input_ids=[3, 4, 5]),
                ],
                batch_size=2,
            )
            rolling_outputs = session.loglikelihood_rolling(
                [
                    RollingLoglikelihoodRequest(text="abcdef"),
                    RollingLoglikelihoodRequest(text="", input_ids=[1, 2, 3, 4]),
                ],
                batch_size=2,
            )
        finally:
            session.close()

    assert [output.text for output in generation_outputs] == [
        "alpha->done",
        "user:beta->done",
    ]
    assert generation_outputs[0].metadata["openai_response"]["model"] == "fake-model"
    assert loglikelihood_outputs[0].logprob == -4.0
    assert loglikelihood_outputs[1].token_count == 3
    assert loglikelihood_outputs[1].metadata["source"] == "fake"
    assert rolling_outputs[0].token_count == 6
    assert rolling_outputs[1].logprob == -2.0


def test_openai_compatible_server_microbatches_parallel_generation_requests() -> None:
    """Verify the HTTP engine keeps multiple requests in flight under one bounded queue."""

    backend = FakeSession()
    with OpenAICompatibleServer(
        session=backend,
        model_name="fake-model",
        max_batch_size=8,
        batch_window_s=0.05,
    ) as server:
        engine = OpenAICompatible(
            base_url=server.base_url,
            batch_size=2,
        )
        session = engine.build(Model(path="fake-model"))
        try:
            outputs = session.generate(
                [
                    GenerationRequest(prompt="one"),
                    GenerationRequest(prompt="two"),
                    GenerationRequest(prompt="three"),
                    GenerationRequest(prompt="four"),
                ],
                batch_size=2,
            )
        finally:
            session.close()

    assert [output.text for output in outputs] == [
        "one->done",
        "two->done",
        "three->done",
        "four->done",
    ]
    assert sum(len(batch) for batch in backend.generate_batches) == 4
    assert sorted(prompt for batch in backend.generate_batches for prompt in batch) == [
        "four",
        "one",
        "three",
        "two",
    ]
    assert max(len(batch) for batch in backend.generate_batches) >= 2


def test_openai_compatible_server_preserves_client_generation_order_within_microbatch() -> None:
    """Verify the server sorts one microbatch back into the client's original request order."""

    backend = FakeSession()
    with OpenAICompatibleServer(
        session=backend,
        model_name="fake-model",
        max_batch_size=3,
        batch_window_s=0.02,
    ) as server:
        batcher = server._generate_batcher
        assert batcher is not None

        def submit(prompt: str, order_index: int) -> str:
            output = batcher.submit(
                GenerationRequest(
                    prompt=prompt,
                    metadata={"_evalution_generation_order": order_index},
                ),
                batch_key=("completion", 256, (), 0.0, False),
            )
            return output.text

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(submit, "two", 2),
                executor.submit(submit, "one", 1),
                executor.submit(submit, "zero", 0),
            ]
            outputs = [future.result() for future in futures]

    assert sorted(outputs) == ["one->done", "two->done", "zero->done"]
    assert backend.generate_batches == [["zero", "one", "two"]]


def test_openai_compatible_server_waits_for_missing_earlier_generation_order() -> None:
    """Verify one late earlier request can still anchor the first server microbatch."""

    backend = FakeSession()
    with OpenAICompatibleServer(
        session=backend,
        model_name="fake-model",
        max_batch_size=2,
        # Keep a wider window so slow CI schedulers still observe the delayed earlier
        # request within the same first refill stream.
        batch_window_s=0.2,
    ) as server:
        batcher = server._generate_batcher
        assert batcher is not None

        def submit(prompt: str, order_index: int, *, delay_s: float = 0.0) -> str:
            if delay_s > 0.0:
                time.sleep(delay_s)
            output = batcher.submit(
                GenerationRequest(
                    prompt=prompt,
                    metadata={"_evalution_generation_order": order_index},
                ),
                batch_key=("completion", 256, (), 0.0, False),
            )
            return output.text

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(submit, "one", 1),
                executor.submit(submit, "two", 2),
                executor.submit(submit, "zero", 0, delay_s=0.02),
            ]
            outputs = [future.result() for future in futures]

    assert sorted(outputs) == ["one->done", "two->done", "zero->done"]
    assert backend.generate_batches == [["zero", "one", "two"]]


def test_openai_compatible_engine_generate_continuous_yields_completion_order() -> None:
    """Verify the HTTP engine yields whichever request finishes first under bounded refill."""

    backend = FakeSession()
    with OpenAICompatibleServer(
        session=backend,
        model_name="fake-model",
        max_batch_size=2,
        batch_window_s=0.02,
    ) as server:
        engine = OpenAICompatible(
            base_url=server.base_url,
            batch_size=2,
        )
        session = engine.build(Model(path="fake-model"))
        try:
            outputs = list(
                session.generate_continuous(
                    [
                        ("slow", GenerationRequest(prompt="slow")),
                        ("fast", GenerationRequest(prompt="fast")),
                    ],
                    batch_size=2,
                )
            )
        finally:
            session.close()

    assert [request_key for request_key, _output in outputs] == ["fast", "slow"]


def test_openai_compatible_engine_batch_size_zero_disables_batching() -> None:
    """Verify `batch_size=0` falls back to single-request submits."""

    backend = FakeSession()
    with OpenAICompatibleServer(
        session=backend,
        model_name="fake-model",
        max_batch_size=8,
        batch_window_s=0.05,
    ) as server:
        engine = OpenAICompatible(
            base_url=server.base_url,
            batch_size=0,
        )
        session = engine.build(Model(path="fake-model"))
        try:
            outputs = session.generate(
                [
                    GenerationRequest(prompt="one"),
                    GenerationRequest(prompt="two"),
                    GenerationRequest(prompt="three"),
                ]
            )
        finally:
            session.close()

    assert [output.text for output in outputs] == [
        "one->done",
        "two->done",
        "three->done",
    ]
    assert backend.generate_batches == [
        ["one"],
        ["two"],
        ["three"],
    ]


def test_openai_compatible_engine_defaults_to_batched_groups_of_four() -> None:
    """Verify the HTTP engine defaults batching on with a client batch size of four."""

    engine = OpenAICompatible()

    assert engine.batch_size == 4
    assert engine.max_parallel_requests == 32
