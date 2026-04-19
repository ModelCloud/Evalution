# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import parse

from evalution.config import Model
from evalution.engines.base import (
    BaseEngine,
    BaseInferenceSession,
    GenerationOutput,
    GenerationRequest,
    LoglikelihoodOutput,
    LoglikelihoodRequest,
    RollingLoglikelihoodOutput,
    RollingLoglikelihoodRequest,
)

# Keep the local server defaults explicit at module scope.
_DEFAULT_BATCH_WINDOW_S = 0.01
_DEFAULT_SERVER_HOST = "127.0.0.1"
_INTERNAL_GENERATION_ORDER_KEY = "_evalution_generation_order"


@dataclass(slots=True)
class _BatchItem:
    """Carry one HTTP request across the server-side microbatch queue."""

    # Preserve one request payload plus its batch compatibility key and completion state.
    payload: Any
    batch_key: tuple[Any, ...]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Exception | None = None


class _QueuedBatcher:
    """Batch compatible requests together on one worker thread."""

    # Keep the class-level queue and thread state explicit because the server owns one batcher per
    # API surface and each one stays live until the whole HTTP server stops.
    def __init__(
        self,
        *,
        name: str,
        batch_window_s: float,
        max_batch_size: int,
        process_batch: Any,
    ) -> None:
        """Initialize the queue and worker thread for one server-side API path."""

        self._name = name
        self._batch_window_s = max(float(batch_window_s), 0.0)
        self._max_batch_size = max(int(max_batch_size), 1)
        self._process_batch = process_batch
        self._pending: list[_BatchItem] = []
        self._closed = False
        self._condition = threading.Condition()
        self._worker = threading.Thread(
            target=self._run,
            name=f"{name}_worker",
            daemon=True,
        )
        self._worker.start()

    def submit(self, payload: Any, *, batch_key: tuple[Any, ...]) -> Any:
        """Submit one request payload to the worker and block for its result."""

        item = _BatchItem(payload=payload, batch_key=batch_key)
        with self._condition:
            if self._closed:
                raise RuntimeError(f"{self._name} batcher is already closed")
            self._pending.append(item)
            self._condition.notify_all()
        item.done.wait()
        if item.error is not None:
            raise item.error
        return item.result

    def close(self) -> None:
        """Stop the worker thread after it drains any already queued requests."""

        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker.join(timeout=5.0)

    def _run(self) -> None:
        """Keep draining queued requests, grouping compatible items into one microbatch."""

        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending and self._closed:
                    return
                batch = self._pop_compatible_batch_locked()
            self._resolve_batch(batch)

    def _pop_compatible_batch_locked(self) -> list[_BatchItem]:
        """Collect one compatible microbatch while briefly waiting for late arrivals."""

        first = self._pending.pop(0)
        batch = [first]
        deadline = time.monotonic() + self._batch_window_s
        while len(batch) < self._max_batch_size:
            compatible_index = self._find_compatible_index_locked(first.batch_key)
            if compatible_index is not None:
                batch.append(self._pending.pop(compatible_index))
                continue
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                break
            self._condition.wait(timeout=remaining_s)
            if not self._pending:
                continue
        return batch

    def _find_compatible_index_locked(self, batch_key: tuple[Any, ...]) -> int | None:
        """Find one queued request that can safely share the current microbatch."""

        for index, item in enumerate(self._pending):
            if item.batch_key == batch_key:
                return index
        return None

    def _resolve_batch(self, batch: list[_BatchItem]) -> None:
        """Run one backend batch and deliver either aligned results or a shared failure."""

        try:
            outputs = self._process_batch([item.payload for item in batch])
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"{self._name} batcher expected {len(batch)} results but received {len(outputs)}"
                )
            for item, output in zip(batch, outputs, strict=True):
                item.result = output
                item.done.set()
        except Exception as exc:
            for item in batch:
                item.error = exc
                item.done.set()


class _GenerationBatcher(_QueuedBatcher):
    """Batch generation requests and release each HTTP caller as soon as its output is ready."""

    def __init__(
        self,
        *,
        name: str,
        batch_window_s: float,
        max_batch_size: int,
        process_batch: Any,
    ) -> None:
        """Track the next client ordinal so concurrent HTTP arrivals rebuild native batch groups."""

        super().__init__(
            name=name,
            batch_window_s=batch_window_s,
            max_batch_size=max_batch_size,
            process_batch=process_batch,
        )
        # Keep the next expected client ordinal explicit so the server can wait briefly for the
        # oldest missing request before sealing a batch around later arrivals.
        self._next_generation_order_index = 0

    def _run(self) -> None:
        """Keep one compatible continuous-refill stream open until arrivals go idle."""

        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if not self._pending and self._closed:
                    return
                batch_key = self._select_generation_stream_batch_key_locked()
            self._resolve_generation_stream(batch_key)

    def _select_generation_stream_batch_key_locked(self) -> tuple[Any, ...]:
        """Choose the next compatible generation stream using the oldest visible client order."""

        smallest_index = self._find_smallest_generation_index_locked()
        if smallest_index is not None:
            return self._pending[smallest_index].batch_key
        return self._pending[0].batch_key

    def _find_first_generation_index_locked(
        self,
        *,
        batch_key: tuple[Any, ...],
    ) -> int | None:
        """Return the first queued request for one compatibility group."""

        for index, item in enumerate(self._pending):
            if item.batch_key == batch_key:
                return index
        return None

    def _find_smallest_generation_index_locked(
        self,
        *,
        batch_key: tuple[Any, ...] | None = None,
    ) -> int | None:
        """Return the pending item index with the smallest visible client ordinal."""

        smallest_index: int | None = None
        smallest_order: int | None = None
        for index, item in enumerate(self._pending):
            if batch_key is not None and item.batch_key != batch_key:
                continue
            order_index = _generation_request_order_value(item)
            if order_index is None:
                continue
            if smallest_order is None or order_index < smallest_order:
                smallest_index = index
                smallest_order = order_index
        return smallest_index

    def _find_generation_index_locked(
        self,
        order_index: int,
        *,
        batch_key: tuple[Any, ...] | None = None,
    ) -> int | None:
        """Find one pending request with the requested client ordinal and optional batch key."""

        for index, item in enumerate(self._pending):
            if batch_key is not None and item.batch_key != batch_key:
                continue
            if _generation_request_order_value(item) == order_index:
                return index
        return None

    def _find_next_generation_index_locked(
        self,
        *,
        batch_key: tuple[Any, ...],
    ) -> int | None:
        """Return the next compatible request in client order when it is already available."""

        first_compatible_index = self._find_first_generation_index_locked(batch_key=batch_key)
        if first_compatible_index is None:
            return None

        smallest_index = self._find_smallest_generation_index_locked(batch_key=batch_key)
        if smallest_index is None:
            return first_compatible_index

        smallest_order = _generation_request_order_value(self._pending[smallest_index])
        if smallest_order is None:
            return first_compatible_index
        if smallest_order < self._next_generation_order_index:
            return smallest_index

        expected_index = self._find_generation_index_locked(
            self._next_generation_order_index,
            batch_key=batch_key,
        )
        if expected_index is not None:
            return expected_index
        return None

    def _resolve_generation_stream(self, batch_key: tuple[Any, ...]) -> None:
        """Run one compatible refill stream so later HTTP arrivals can join before the GPU idles."""

        emitted_indexes: set[int] = set()
        submitted_items: dict[int, _BatchItem] = {}
        next_stream_index = 0
        idle_deadline = time.monotonic() + self._batch_window_s

        def iter_requests() -> Any:
            """Yield queued requests in client order until one brief idle window closes the stream."""

            nonlocal idle_deadline, next_stream_index
            while True:
                with self._condition:
                    while True:
                        compatible_index = self._find_next_generation_index_locked(batch_key=batch_key)
                        if compatible_index is not None:
                            item = self._pending.pop(compatible_index)
                            break

                        remaining_s = idle_deadline - time.monotonic()
                        if remaining_s <= 0:
                            fallback_index = self._find_smallest_generation_index_locked(
                                batch_key=batch_key,
                            )
                            if fallback_index is None:
                                fallback_index = self._find_first_generation_index_locked(
                                    batch_key=batch_key,
                                )
                            if fallback_index is None:
                                return
                            item = self._pending.pop(fallback_index)
                            break

                        if self._closed and self._find_first_generation_index_locked(batch_key=batch_key) is None:
                            return
                        self._condition.wait(timeout=remaining_s)

                order_index = _generation_request_order_value(item)
                if order_index is not None:
                    # Advance the forward-looking cursor without moving it backwards when a delayed
                    # earlier request shows up after newer work has already been submitted.
                    self._next_generation_order_index = max(
                        self._next_generation_order_index,
                        order_index + 1,
                    )

                request_key = next_stream_index
                next_stream_index += 1
                submitted_items[request_key] = item
                idle_deadline = time.monotonic() + self._batch_window_s
                yield request_key, item.payload

        try:
            for batch_index, output in self._process_batch(iter_requests()):
                resolved_index = int(batch_index)
                batch_item = submitted_items[resolved_index]
                batch_item.result = output
                batch_item.done.set()
                emitted_indexes.add(resolved_index)
            missing_indexes = set(submitted_items) - emitted_indexes
            if missing_indexes:
                raise RuntimeError(
                    f"{self._name} batcher did not emit results for indexes {sorted(missing_indexes)}"
                )
        except Exception as exc:
            for batch_item in submitted_items.values():
                if batch_item.done.is_set():
                    continue
                batch_item.error = exc
                batch_item.done.set()


def _generation_request_order_value(item: _BatchItem) -> int | None:
    """Return the client's generation ordinal when the request was submitted by Evalution."""

    metadata = getattr(item.payload, "metadata", None)
    if isinstance(metadata, dict):
        order_index = metadata.get(_INTERNAL_GENERATION_ORDER_KEY)
        if isinstance(order_index, int):
            return order_index
    return None


def _generation_request_order_key(item: _BatchItem) -> tuple[int, int]:
    """Return a stable sort key that preserves the client's original generation order."""

    order_index = _generation_request_order_value(item)
    if order_index is not None:
        return (0, order_index)
    return (1, 0)


class _EvalutionOpenAIServerHTTP(ThreadingHTTPServer):
    """Bind the stdlib HTTP server to the higher-level Evalution API adapter."""

    # Keep the server-owned adapter explicit so request handlers can stay stateless.
    def __init__(self, server_address: tuple[str, int], adapter: OpenAICompatibleServer) -> None:
        """Initialize the HTTP server with a reference to the Evalution adapter."""

        super().__init__(server_address, _EvalutionOpenAIRequestHandler)
        self.adapter = adapter


class _EvalutionOpenAIRequestHandler(BaseHTTPRequestHandler):
    """Handle OpenAI-style HTTP requests by delegating into the server adapter."""

    # Silence the stdlib per-request log so tests keep deterministic output.
    def log_message(self, format: str, *args: Any) -> None:
        """Suppress the default request logger used by the stdlib server."""

        del format, args
        return None

    def do_GET(self) -> None:
        """Serve readiness and model-discovery endpoints."""

        route = parse.urlsplit(self.path).path
        if route == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        if route == "/v1/models":
            self._write_json(HTTPStatus.OK, self.server.adapter.list_models())
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": f"unknown route: {route}"}})

    def do_POST(self) -> None:
        """Serve generation and scoring endpoints."""

        route = parse.urlsplit(self.path).path
        try:
            payload = self._read_json()
            if route == "/v1/completions":
                response = self.server.adapter.handle_completions(payload)
            elif route == "/v1/chat/completions":
                response = self.server.adapter.handle_chat_completions(payload)
            elif route == "/v1/eval/loglikelihood":
                response = self.server.adapter.handle_loglikelihood(payload)
            elif route == "/v1/eval/loglikelihood/rolling":
                response = self.server.adapter.handle_loglikelihood_rolling(payload)
            else:
                self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": f"unknown route: {route}"}})
                return
        except ValueError as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            return
        except Exception as exc:
            self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": str(exc)}})
            return
        self._write_json(HTTPStatus.OK, response)

    def _read_json(self) -> dict[str, Any]:
        """Decode the request body as one JSON object."""

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must decode to a JSON object")
        return payload

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        """Write one JSON response with the supplied status code."""

        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@dataclass(slots=True)
class OpenAICompatibleServer:
    """Serve one Evalution inference session through an OpenAI-style HTTP facade."""

    # Keep the served-model identity and queueing knobs explicit for reproducible tests.
    session: BaseInferenceSession
    model_name: str
    host: str = _DEFAULT_SERVER_HOST
    port: int = 0
    max_batch_size: int = 8
    batch_window_s: float = _DEFAULT_BATCH_WINDOW_S
    close_session_on_close: bool = True
    _http_server: _EvalutionOpenAIServerHTTP | None = field(default=None, init=False, repr=False)
    _serve_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _generate_batcher: _QueuedBatcher | None = field(default=None, init=False, repr=False)
    _loglikelihood_batcher: _QueuedBatcher | None = field(default=None, init=False, repr=False)
    _rolling_batcher: _QueuedBatcher | None = field(default=None, init=False, repr=False)

    def start(self) -> OpenAICompatibleServer:
        """Start the HTTP server and its internal queue workers."""

        if self._http_server is not None:
            return self
        self._generate_batcher = _GenerationBatcher(
            name="generate",
            batch_window_s=self.batch_window_s,
            max_batch_size=self.max_batch_size,
            process_batch=self._process_generation_batch_stream,
        )
        self._loglikelihood_batcher = _QueuedBatcher(
            name="loglikelihood",
            batch_window_s=self.batch_window_s,
            max_batch_size=self.max_batch_size,
            process_batch=self._process_loglikelihood_batch,
        )
        self._rolling_batcher = _QueuedBatcher(
            name="rolling_loglikelihood",
            batch_window_s=self.batch_window_s,
            max_batch_size=self.max_batch_size,
            process_batch=self._process_rolling_loglikelihood_batch,
        )
        self._http_server = _EvalutionOpenAIServerHTTP((self.host, self.port), self)
        self._serve_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="evalution_openai_http_server",
            daemon=True,
        )
        self._serve_thread.start()
        return self

    @property
    def base_url(self) -> str:
        """Expose the base URL callers should use for this server."""

        if self._http_server is None:
            raise RuntimeError("OpenAI-compatible server has not been started")
        host, port = self._http_server.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        """Stop the HTTP server, stop batch workers, and optionally close the session."""

        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
        if self._serve_thread is not None:
            self._serve_thread.join(timeout=5.0)
            self._serve_thread = None
        for batcher in (self._generate_batcher, self._loglikelihood_batcher, self._rolling_batcher):
            if batcher is not None:
                batcher.close()
        self._generate_batcher = None
        self._loglikelihood_batcher = None
        self._rolling_batcher = None
        if self.close_session_on_close:
            self.session.close()

    def __enter__(self) -> OpenAICompatibleServer:
        """Start the server when entering a managed context."""

        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Stop the server when leaving a managed context."""

        del exc_type, exc, tb
        self.close()

    def list_models(self) -> dict[str, Any]:
        """Return one minimal OpenAI-style model list response."""

        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "owned_by": "evalution",
                }
            ],
        }

    def handle_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle one `/v1/completions` request."""

        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("`prompt` must be a string")
        request_item = GenerationRequest(
            prompt=prompt,
            stop=_coerce_stop(payload.get("stop")),
            max_new_tokens=int(payload.get("max_tokens", 256)),
            temperature=float(payload.get("temperature", 0.0)),
            do_sample=float(payload.get("temperature", 0.0)) > 0.0,
            metadata=dict(payload.get("metadata") or {}),
        )
        batch_key = (
            "completion",
            request_item.max_new_tokens,
            tuple(request_item.stop),
            request_item.temperature,
            request_item.do_sample,
        )
        output = self._require_batcher(self._generate_batcher, "generation").submit(
            request_item,
            batch_key=batch_key,
        )
        return {
            "id": "cmpl-evalution",
            "object": "text_completion",
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "text": output.text,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "completion_tokens": output.metadata.get("completion_tokens"),
            },
        }

    def handle_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle one `/v1/chat/completions` request."""

        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("`messages` must be a list")
        request_item = GenerationRequest(
            messages=messages,
            stop=_coerce_stop(payload.get("stop")),
            max_new_tokens=int(payload.get("max_tokens", payload.get("max_completion_tokens", 256))),
            temperature=float(payload.get("temperature", 0.0)),
            do_sample=float(payload.get("temperature", 0.0)) > 0.0,
            metadata=dict(payload.get("metadata") or {}),
        )
        batch_key = (
            "chat_completion",
            request_item.max_new_tokens,
            tuple(request_item.stop),
            request_item.temperature,
            request_item.do_sample,
        )
        output = self._require_batcher(self._generate_batcher, "generation").submit(
            request_item,
            batch_key=batch_key,
        )
        return {
            "id": "chatcmpl-evalution",
            "object": "chat.completion",
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": output.text,
                    },
                }
            ],
            "usage": {
                "completion_tokens": output.metadata.get("completion_tokens"),
            },
        }

    def handle_loglikelihood(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle one Evalution-specific log-likelihood request."""

        request_item = LoglikelihoodRequest(
            context=str(payload.get("context", "")),
            continuation=str(payload.get("continuation", "")),
            context_input_ids=self._coerce_optional_int_list(payload.get("context_input_ids")),
            continuation_input_ids=self._coerce_optional_int_list(payload.get("continuation_input_ids")),
            metadata=dict(payload.get("metadata") or {}),
        )
        batch_key = ("loglikelihood",)
        output = self._require_batcher(self._loglikelihood_batcher, "loglikelihood").submit(
            request_item,
            batch_key=batch_key,
        )
        return {
            "logprob": output.logprob,
            "is_greedy": output.is_greedy,
            "token_count": output.token_count,
            "metadata": dict(output.metadata),
        }

    def handle_loglikelihood_rolling(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle one Evalution-specific rolling log-likelihood request."""

        request_item = RollingLoglikelihoodRequest(
            text=str(payload.get("text", "")),
            input_ids=self._coerce_optional_int_list(payload.get("input_ids")),
            metadata=dict(payload.get("metadata") or {}),
        )
        batch_key = ("rolling_loglikelihood",)
        output = self._require_batcher(self._rolling_batcher, "rolling_loglikelihood").submit(
            request_item,
            batch_key=batch_key,
        )
        return {
            "logprob": output.logprob,
            "token_count": output.token_count,
            "metadata": dict(output.metadata),
        }

    def _process_generation_batch_stream(
        self,
        requests: Any,
    ) -> Any:
        """Run one queued generation stream with the server batch size as the refill cap."""

        return self.session.generate_continuous(requests, batch_size=self.max_batch_size)

    def _process_loglikelihood_batch(
        self,
        requests: list[LoglikelihoodRequest],
    ) -> list[LoglikelihoodOutput]:
        """Run one queued log-likelihood microbatch against the backing session."""

        return self.session.loglikelihood(requests, batch_size=len(requests))

    def _process_rolling_loglikelihood_batch(
        self,
        requests: list[RollingLoglikelihoodRequest],
    ) -> list[RollingLoglikelihoodOutput]:
        """Run one queued rolling log-likelihood microbatch against the backing session."""

        return self.session.loglikelihood_rolling(requests, batch_size=len(requests))

    def _coerce_optional_int_list(self, value: Any) -> list[int] | None:
        """Normalize optional token-id arrays from JSON payloads."""

        if value is None:
            return None
        return [int(item) for item in value]

    def _require_batcher(self, batcher: _QueuedBatcher | None, name: str) -> _QueuedBatcher:
        """Return a started batcher or fail with one stable server-state error."""

        if batcher is None:
            raise RuntimeError(f"{name} batcher is unavailable because the server is not started")
        return batcher


def _coerce_stop(stop: Any) -> list[str]:
    """Normalize OpenAI-style stop values into the list form used by Evalution."""

    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [str(item) for item in stop]


def build_openai_compatible_server(
    *,
    engine: BaseEngine,
    model: Model,
    model_name: str | None = None,
    host: str = _DEFAULT_SERVER_HOST,
    port: int = 0,
    max_batch_size: int = 8,
    batch_window_s: float = _DEFAULT_BATCH_WINDOW_S,
) -> OpenAICompatibleServer:
    """Build and start an OpenAI-compatible HTTP server around one Evalution engine session."""

    session = engine.build(model)
    server = OpenAICompatibleServer(
        session=session,
        model_name=model_name or model.label or model.path,
        host=host,
        port=port,
        max_batch_size=max_batch_size,
        batch_window_s=batch_window_s,
    )
    return server.start()
