# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import json
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass, field, replace
from typing import Any
from urllib import error, request

from evalution.config import Model
from evalution.engines.base import (
    BaseInferenceSession,
    GenerationOutput,
    GenerationRequest,
    LoglikelihoodOutput,
    LoglikelihoodRequest,
    RollingLoglikelihoodOutput,
    RollingLoglikelihoodRequest,
    SharedEngineConfig,
)

# Keep the engine defaults and endpoint paths explicit at module scope.
_DEFAULT_GENERATE_PATH = "/v1/completions"
_DEFAULT_CHAT_PATH = "/v1/chat/completions"
_DEFAULT_LOGLIKELIHOOD_PATH = "/v1/eval/loglikelihood"
_DEFAULT_ROLLING_LOGLIKELIHOOD_PATH = "/v1/eval/loglikelihood/rolling"
_INTERNAL_GENERATION_ORDER_KEY = "_evalution_generation_order"


def _coerce_batch_size(value: int | str | None, *, default: int) -> int:
    """Resolve the configured batch size, preserving zero as the explicit disable switch."""

    if value is None:
        return max(default, 0)
    return int(value)


def _normalize_base_url(base_url: str) -> str:
    """Normalize the configured base URL so endpoint joins stay predictable."""

    return base_url.rstrip("/")


def _join_url(base_url: str, path: str) -> str:
    """Join the API base URL with a relative endpoint path."""

    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{_normalize_base_url(base_url)}{normalized_path}"


def _coerce_stop(stop: Any) -> list[str]:
    """Normalize OpenAI-style `stop` values into Evalution's list representation."""

    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [str(item) for item in stop]


def _with_generation_order(request_item: GenerationRequest, order_index: int) -> GenerationRequest:
    """Attach a stable client-side ordinal so server microbatches can preserve request order."""

    metadata = dict(request_item.metadata)
    metadata[_INTERNAL_GENERATION_ORDER_KEY] = int(order_index)
    return replace(request_item, metadata=metadata)


@dataclass(slots=True)
class OpenAICompatible(SharedEngineConfig):
    """Configure Evalution to talk to an OpenAI-compatible HTTP endpoint."""

    # Keep the transport and compatibility knobs explicit for result serialization.
    base_url: str = "http://127.0.0.1:8000"
    api_key: str | None = None
    model_name: str | None = None
    timeout_s: float = 300.0
    batch_size: int = 4
    max_parallel_requests: int = 32
    completions_path: str = _DEFAULT_GENERATE_PATH
    chat_completions_path: str = _DEFAULT_CHAT_PATH
    loglikelihood_path: str = _DEFAULT_LOGLIKELIHOOD_PATH
    rolling_loglikelihood_path: str = _DEFAULT_ROLLING_LOGLIKELIHOOD_PATH
    extra_headers: dict[str, str] = field(default_factory=dict)

    def build(self, model: Model) -> BaseInferenceSession:
        """Construct an HTTP-backed inference session for one served model."""

        self.resolved_engine = "OpenAICompatible"
        return OpenAICompatibleSession.from_config(self, model)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the engine configuration for result payloads."""

        return asdict(self)


@dataclass(slots=True)
class OpenAICompatibleSession(BaseInferenceSession):
    """Drive Evalution generation and scoring against an HTTP API."""

    # Keep the session-level HTTP configuration explicit for execution metadata and tests.
    config: OpenAICompatible
    model_config: Model
    model_name: str
    _closed: bool = field(default=False, init=False, repr=False)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config: OpenAICompatible,
        model_config: Model,
    ) -> OpenAICompatibleSession:
        """Construct the session from engine and model configuration."""

        model_name = config.model_name or model_config.path
        return cls(
            config=config,
            model_config=model_config,
            model_name=model_name,
        )

    @property
    def request_executor_requires_non_main_thread(self) -> bool:
        """Allow HTTP-backed execution to run on the caller thread when needed."""

        return False

    def describe_execution(self) -> dict[str, Any]:
        """Report the HTTP endpoint and concurrency settings driving this session."""

        return {
            "generation_backend": "openai_http",
            "base_url": _normalize_base_url(self.config.base_url),
            "model_name": self.model_name,
            "batch_size": self.config.batch_size,
            "max_parallel_requests": self.config.max_parallel_requests,
        }

    def generate(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int | None = None,
    ) -> list[GenerationOutput]:
        """Generate completions with a bounded in-flight HTTP queue while preserving input order."""

        if not requests:
            return []
        return self._run_ordered_refill(
            requests=[
                _with_generation_order(request_item, order_index)
                for order_index, request_item in enumerate(requests)
            ],
            worker=self._generate_one,
            batch_size=batch_size,
        )

    def generate_continuous(
        self,
        requests: Any,
        *,
        batch_size: int | None = None,
    ) -> Any:
        """Yield outputs in completion order while refilling a bounded in-flight HTTP queue."""

        def iterator() -> Any:
            """Drive request submission with one bounded in-flight queue."""

            yield from self._stream_unordered_refill(
                items=(
                    (
                        request_key,
                        _with_generation_order(request_item, order_index),
                    )
                    for order_index, (request_key, request_item) in enumerate(requests)
                ),
                worker=self._generate_one,
                batch_size=batch_size,
            )

        return iterator()

    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]:
        """Score continuations with a bounded in-flight HTTP queue while preserving order."""

        if not requests:
            return []
        return self._run_ordered_refill(
            requests=requests,
            worker=self._loglikelihood_one,
            batch_size=batch_size,
        )

    def loglikelihood_rolling(
        self,
        requests: list[RollingLoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[RollingLoglikelihoodOutput]:
        """Score rolling likelihood spans with a bounded in-flight HTTP queue while preserving order."""

        if not requests:
            return []
        return self._run_ordered_refill(
            requests=requests,
            worker=self._loglikelihood_rolling_one,
            batch_size=batch_size,
        )

    def gc(self) -> None:
        """Release lightweight client-side state between suites."""

        return None

    def close(self) -> None:
        """Mark the HTTP session as closed so later requests fail fast."""

        with self._state_lock:
            self._closed = True

    def _run_ordered_refill(
        self,
        requests: list[Any],
        *,
        worker: Any,
        batch_size: int | None,
    ) -> list[Any]:
        """Keep up to `batch_size` requests in flight while preserving positional order."""

        outputs: list[Any | None] = [None] * len(requests)
        parallelism = self._effective_batch_size(batch_size)
        with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="evalution_openai_http") as executor:
            futures = {
                executor.submit(worker, request_item): request_index
                for request_index, request_item in enumerate(requests[:parallelism])
            }
            next_request_index = parallelism
            while futures:
                completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in completed:
                    request_index = futures.pop(future)
                    outputs[request_index] = future.result()
                    if next_request_index >= len(requests):
                        continue
                    futures[executor.submit(worker, requests[next_request_index])] = next_request_index
                    next_request_index += 1
        if any(output is None for output in outputs):
            raise RuntimeError("OpenAI-compatible engine returned incomplete outputs")
        return [output for output in outputs if output is not None]

    def _stream_unordered_refill(
        self,
        *,
        items: Any,
        worker: Any,
        batch_size: int | None,
    ) -> Any:
        """Yield outputs in completion order while refilling the in-flight queue slot by slot."""

        parallelism = self._effective_batch_size(batch_size)
        with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="evalution_openai_http") as executor:
            pending: dict[Future[Any], Any] = {}

            def submit_next() -> bool:
                """Submit one additional request when the source iterator still has work."""

                try:
                    request_key, request_item = next(items)
                except StopIteration:
                    return False
                pending[executor.submit(worker, request_item)] = request_key
                return True

            for _ in range(parallelism):
                if not submit_next():
                    break

            while pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in completed:
                    request_key = pending.pop(future)
                    yield request_key, future.result()
                    submit_next()

    def _effective_batch_size(self, batch_size: int | None) -> int:
        """Resolve the in-flight queue depth, where zero disables batching and falls back to singles."""

        configured_batch_size = (
            self.config.batch_size
            if batch_size is None
            else int(batch_size)
        )
        resolved_batch_size = _coerce_batch_size(configured_batch_size, default=4)
        if resolved_batch_size == 0:
            return 1
        return max(resolved_batch_size, 1)

    def _generate_one(self, request_item: GenerationRequest) -> GenerationOutput:
        """Issue one generation request against the appropriate OpenAI-style endpoint."""

        if request_item.messages is not None:
            payload = {
                "model": self.model_name,
                "messages": request_item.messages,
                "max_tokens": int(request_item.max_new_tokens),
                "temperature": float(request_item.temperature),
                "stop": list(request_item.stop),
                "metadata": dict(request_item.metadata),
            }
            response = self._post_json(self.config.chat_completions_path, payload)
            choices = response.get("choices") or []
            if not choices:
                raise RuntimeError("chat completion response did not include any choices")
            message = choices[0].get("message") or {}
            text = str(message.get("content", ""))
            prompt = request_item.rendered_prompt or str(request_item.messages)
            metadata = dict(request_item.metadata)
            metadata["openai_response"] = response
            return GenerationOutput(prompt=prompt, text=text, metadata=metadata)

        prompt = request_item.rendered_prompt or request_item.prompt or ""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": int(request_item.max_new_tokens),
            "temperature": float(request_item.temperature),
            "stop": list(request_item.stop),
            "metadata": dict(request_item.metadata),
        }
        response = self._post_json(self.config.completions_path, payload)
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("completion response did not include any choices")
        text = str(choices[0].get("text", ""))
        metadata = dict(request_item.metadata)
        metadata["openai_response"] = response
        return GenerationOutput(prompt=prompt, text=text, metadata=metadata)

    def _loglikelihood_one(self, request_item: LoglikelihoodRequest) -> LoglikelihoodOutput:
        """Issue one log-likelihood request against Evalution's scoring endpoint."""

        payload = {
            "model": self.model_name,
            "context": request_item.context,
            "continuation": request_item.continuation,
            "context_input_ids": request_item.context_input_ids,
            "continuation_input_ids": request_item.continuation_input_ids,
            "metadata": dict(request_item.metadata),
        }
        response = self._post_json(self.config.loglikelihood_path, payload)
        metadata = dict(request_item.metadata)
        response_metadata = response.get("metadata")
        if isinstance(response_metadata, dict):
            metadata.update(response_metadata)
        return LoglikelihoodOutput(
            logprob=float(response["logprob"]),
            is_greedy=bool(response["is_greedy"]),
            token_count=int(response["token_count"]),
            metadata=metadata,
        )

    def _loglikelihood_rolling_one(
        self,
        request_item: RollingLoglikelihoodRequest,
    ) -> RollingLoglikelihoodOutput:
        """Issue one rolling log-likelihood request against Evalution's scoring endpoint."""

        payload = {
            "model": self.model_name,
            "text": request_item.text,
            "input_ids": request_item.input_ids,
            "metadata": dict(request_item.metadata),
        }
        response = self._post_json(self.config.rolling_loglikelihood_path, payload)
        metadata = dict(request_item.metadata)
        response_metadata = response.get("metadata")
        if isinstance(response_metadata, dict):
            metadata.update(response_metadata)
        return RollingLoglikelihoodOutput(
            logprob=float(response["logprob"]),
            token_count=int(response["token_count"]),
            metadata=metadata,
        )

    def _post_json(self, endpoint_path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one JSON request and return the decoded JSON response body."""

        self._assert_open()
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.config.extra_headers,
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        http_request = request.Request(
            _join_url(self.config.base_url, endpoint_path),
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=float(self.config.timeout_s)) as response:
                response_body = response.read()
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible endpoint {endpoint_path} returned HTTP {exc.code}: {response_body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"failed to reach OpenAI-compatible endpoint {endpoint_path}: {exc}") from exc
        try:
            return json.loads(response_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"OpenAI-compatible endpoint {endpoint_path} returned invalid JSON"
            ) from exc

    def _assert_open(self) -> None:
        """Fail fast when callers try to use a closed HTTP session."""

        with self._state_lock:
            if self._closed:
                raise RuntimeError("OpenAI-compatible session is already closed")
