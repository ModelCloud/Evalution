# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evalution.config import Model
from evalution.engines.base import (
    GenerationOutput,
    GenerationRequest,
    LoglikelihoodOutput,
    LoglikelihoodRequest,
)
from evalution.engines.transformers_common import (
    BaseTransformerSession,
    _AUTO_BATCH_SIZE,
    _TransformersCommonConfig,
    _clone_prepare_tokenizer,
    _load_tokenizer_from_model,
    _normalize_batch_size,
    _normalize_tokenizer_special_tokens,
    _resolve_tokenizer_source,
)

_DEFAULT_SGLANG_PATH = "/root/projects/GPTQModel/hub/sglang/python"


class _SGLangClient(ABC):
    supports_raw_logits: bool = False
    transport: str = "unknown"

    @abstractmethod
    def generate(self, **payload: Any) -> list[dict[str, Any]]: ...

    def gc(self) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(slots=True)
class _SGLangPythonClient(_SGLangClient):
    engine: Any
    supports_raw_logits: bool = False
    transport: str = "python"

    def generate(self, **payload: Any) -> list[dict[str, Any]]:
        prompt = payload.pop("text", None)
        if "return_text_in_logprobs" in payload:
            payload.pop("return_text_in_logprobs")
        response = self.engine.generate(prompt=prompt, **payload)
        return _normalize_sglang_response(response)

    def gc(self) -> None:
        flush_cache = getattr(self.engine, "flush_cache", None)
        if callable(flush_cache):
            flush_cache()

    def close(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


@dataclass(slots=True)
class _LoadedSGLangRuntime:
    client: _SGLangClient
    tokenizer: Any
    prepare_tokenizer: Any | None
    model: Any
    input_device: Any


@dataclass(slots=True)
class SGLang(_TransformersCommonConfig):
    # SGLang integration stays in-process through `sglang.Engine`; no HTTP server is used.
    base_url: str | None = None
    transport: str = "python"
    sglang_path: str | None = _DEFAULT_SGLANG_PATH
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    server_kwargs: dict[str, Any] = field(default_factory=dict)
    default_auto_batch_size: int = 32

    def build(self, model: Model) -> BaseTransformerSession:
        self.resolved_engine = "SGLang"
        return SGLangSession.from_config(self, model)


@dataclass(slots=True)
class SGLangSession(BaseTransformerSession):
    client: _SGLangClient | None = field(default=None, repr=False)
    transport: str = "unknown"
    raw_logits_enabled: bool = False

    @classmethod
    def from_config(cls, config: SGLang, model_config: Model) -> SGLangSession:
        runtime = load_sglang_runtime(config, model_config)
        return cls(
            config=config,
            model_config=model_config,
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            prepare_tokenizer=runtime.prepare_tokenizer,
            input_device=runtime.input_device,
            generation_backend="sglang.generate",
            client=runtime.client,
            transport=runtime.client.transport,
            raw_logits_enabled=runtime.client.supports_raw_logits,
        )

    def describe_execution(self) -> dict[str, Any]:
        execution = super(SGLangSession, self).describe_execution()
        execution.update(
            {
                "transport": self.transport,
                "logprob_backend": "sglang.generate",
                "raw_logits_enabled": self.raw_logits_enabled,
            }
        )
        return execution

    def resolve_batch_size(self, requests: list[GenerationRequest]) -> int:
        del requests
        configured = _normalize_batch_size(self.config.batch_size)
        if configured != _AUTO_BATCH_SIZE:
            return configured
        return max(1, int(self.config.default_auto_batch_size))

    def loglikelihood(
        self,
        requests: list[LoglikelihoodRequest],
        *,
        batch_size: int | None = None,
    ) -> list[LoglikelihoodOutput]:
        if not requests:
            return []

        with self._generation_lock:
            prepared_requests = [self._prepare_loglikelihood_request(request) for request in requests]
            effective_batch_size = batch_size or self._resolve_scoring_batch_size(prepared_requests)
            chunk_groups: list[list[Any]] = [[] for _ in requests]
            ordered_requests = list(enumerate(prepared_requests))
            ordered_requests.sort(key=self._loglikelihood_request_sort_key)
            for request_index, (prefix_ids, target_ids, metadata) in ordered_requests:
                chunk_groups[request_index].extend(
                    self._build_loglikelihood_chunks(
                        request_index=request_index,
                        prefix_ids=prefix_ids,
                        target_ids=target_ids,
                        metadata=metadata,
                    )
                )

            chunks = [chunk for group in chunk_groups for chunk in group]
            chunk_scores = self._score_chunks(chunks, batch_size=effective_batch_size)
            totals = [
                LoglikelihoodOutput(
                    logprob=0.0,
                    is_greedy=True,
                    token_count=0,
                    metadata=dict(request.metadata),
                )
                for request in requests
            ]
            for chunk, chunk_output in zip(chunks, chunk_scores, strict=True):
                current = totals[chunk.request_index]
                combined_metadata = dict(current.metadata)
                combined_metadata["sglang_transport"] = self.transport
                if chunk_output.metadata.get("raw_logits_enabled"):
                    combined_metadata["raw_logits_enabled"] = True
                totals[chunk.request_index] = LoglikelihoodOutput(
                    logprob=current.logprob + chunk_output.logprob,
                    is_greedy=current.is_greedy and chunk_output.is_greedy,
                    token_count=current.token_count + chunk_output.token_count,
                    metadata=combined_metadata,
                )
            return totals

    def _generate_standard(
        self,
        requests: list[GenerationRequest],
        *,
        batch_size: int,
    ) -> list[GenerationOutput]:
        prepared = self.prepare_requests(requests)
        outputs: list[GenerationOutput] = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            payload = {
                "input_ids": [list(request.input_ids) for request in batch],
                "sampling_params": [self._build_sampling_params(request) for request in batch],
            }
            responses = self.client.generate(**payload)
            if len(responses) != len(batch):
                raise RuntimeError("sglang returned an unexpected number of generation results")
            for request, response in zip(batch, responses, strict=True):
                meta_info = dict(response.get("meta_info") or {})
                text = str(response.get("text") or "")
                prompt = request.rendered_prompt or self._render_request(request)
                if text.startswith(prompt):
                    text = text[len(prompt) :]
                outputs.append(
                    GenerationOutput(
                        prompt=prompt,
                        text=text,
                        metadata={
                            **dict(request.metadata),
                            "sglang_meta": meta_info,
                        },
                    )
                )
        return outputs

    def _score_chunks(
        self,
        chunks: list[Any],
        *,
        batch_size: int,
    ) -> list[LoglikelihoodOutput]:
        scored_chunks: list[LoglikelihoodOutput] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            payload = {
                "input_ids": [list(chunk.input_ids) for chunk in batch],
                "sampling_params": [
                    {
                        # Mirror lm-eval's SGLang path: prompt logprobs are requested in a
                        # non-generation scoring mode that still asks the engine to decode one token.
                        "max_new_tokens": 1,
                        "temperature": 0.0,
                    }
                    for _ in batch
                ],
                "return_logprob": [True for _ in batch],
                "logprob_start_len": [0 for _ in batch],
                "top_logprobs_num": [1 for _ in batch],
                "return_text_in_logprobs": False,
            }
            responses = self.client.generate(**payload)
            if len(responses) != len(batch):
                raise RuntimeError("sglang returned an unexpected number of scoring results")

            for chunk, response in zip(batch, responses, strict=True):
                meta_info = dict(response.get("meta_info") or {})
                target_ids = chunk.input_ids[chunk.score_start : chunk.score_start + chunk.score_count]
                all_input_token_logprobs = _coerce_position_entries(meta_info.get("input_token_logprobs"))
                if len(all_input_token_logprobs) == chunk.score_count:
                    input_token_logprobs = all_input_token_logprobs
                else:
                    input_token_logprobs = all_input_token_logprobs[
                        chunk.score_start : chunk.score_start + chunk.score_count
                    ]
                if len(input_token_logprobs) != chunk.score_count:
                    raise RuntimeError("sglang returned too few input_token_logprobs for scoring")

                all_top_logprobs = _coerce_nested_position_entries(meta_info.get("input_top_logprobs"))
                if len(all_top_logprobs) == chunk.score_count:
                    top_logprobs = all_top_logprobs
                else:
                    top_logprobs = all_top_logprobs[
                        chunk.score_start : chunk.score_start + chunk.score_count
                    ]
                requested_logprobs = _coerce_nested_position_entries(
                    meta_info.get("input_token_ids_logprobs")
                )
                requested_logits = _coerce_nested_position_entries(
                    meta_info.get("input_token_ids_logits")
                )
                token_logits = _coerce_position_entries(meta_info.get("input_token_logits"))
                output_top_logprobs = _coerce_nested_position_entries(meta_info.get("output_top_logprobs"))
                output_requested_logprobs = _coerce_nested_position_entries(
                    meta_info.get("output_token_ids_logprobs")
                )

                total_logprob = 0.0
                is_greedy = True
                for position, target_id in enumerate(target_ids):
                    selected = input_token_logprobs[position]
                    selected_token_id = selected.get("token_id")
                    requested_scores = _entries_to_score_map(
                        requested_logprobs[position] if position < len(requested_logprobs) else []
                    )
                    if not requested_scores:
                        requested_scores = _entries_to_score_map(
                            output_requested_logprobs[position]
                            if position < len(output_requested_logprobs)
                            else []
                        )
                    selected_score = selected.get("score")
                    if selected_score is None and selected_token_id is not None and int(selected_token_id) == int(target_id):
                        selected_score = requested_scores.get(int(target_id))
                    if selected_score is None:
                        raise RuntimeError(
                            "sglang did not provide a usable logprob for the requested continuation token"
                        )
                    total_logprob += float(selected_score)
                    top_choice = (
                        top_logprobs[position][0]
                        if position < len(top_logprobs) and top_logprobs[position]
                        else (
                            output_top_logprobs[position][0]
                            if position < len(output_top_logprobs) and output_top_logprobs[position]
                            else None
                        )
                    )
                    greedy_token_id = top_choice["token_id"] if top_choice is not None else None
                    if greedy_token_id is not None and greedy_token_id != int(target_id):
                        is_greedy = False

                scored_chunks.append(
                    LoglikelihoodOutput(
                        logprob=total_logprob,
                        is_greedy=is_greedy,
                        token_count=chunk.score_count,
                        metadata={
                            **dict(chunk.metadata),
                            "raw_logits_enabled": bool(requested_logits or token_logits),
                        },
                    )
                )
        return scored_chunks

    def gc(self) -> None:
        super(SGLangSession, self).gc()
        if self.client is not None:
            self.client.gc()

    def close(self) -> None:
        try:
            super(SGLangSession, self).close()
        finally:
            if self.client is not None:
                self.client.close()

    def _build_sampling_params(self, request: GenerationRequest) -> dict[str, Any]:
        if request.num_beams != 1:
            raise ValueError("sglang engine does not support beam search; set num_beams=1")
        params = {
            "max_new_tokens": request.max_new_tokens
            if request.max_new_tokens is not None
            else self.config.max_new_tokens,
            "temperature": request.temperature if request.do_sample else 0.0,
        }
        if request.stop:
            params["stop"] = list(request.stop)
        return params


def load_sglang_runtime(config: SGLang, model_config: Model) -> _LoadedSGLangRuntime:
    trust_remote_code = (
        config.trust_remote_code
        if config.trust_remote_code is not None
        else model_config.trust_remote_code
    )
    tokenizer = _load_tokenizer_from_model(
        _resolve_tokenizer_source(model_config),
        revision=model_config.revision,
        trust_remote_code=trust_remote_code,
        **model_config.tokenizer_kwargs,
    )
    tokenizer.padding_side = config.padding_side
    _normalize_tokenizer_special_tokens(tokenizer=tokenizer)

    prepare_tokenizer = _clone_prepare_tokenizer(
        tokenizer=tokenizer,
        model_config=model_config,
        trust_remote_code=trust_remote_code,
        model=None,
    )

    model_length = getattr(tokenizer, "model_max_length", None)
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=model_length))
    input_device = SimpleNamespace(type="cpu")
    client = _build_sglang_client(config, model_config)

    return _LoadedSGLangRuntime(
        client=client,
        tokenizer=tokenizer,
        prepare_tokenizer=prepare_tokenizer,
        model=model,
        input_device=input_device,
    )


def _build_sglang_client(config: SGLang, model_config: Model) -> _SGLangClient:
    transport = config.transport.lower()
    if transport not in {"auto", "python"}:
        raise ValueError("sglang transport must be one of: auto, python")
    if config.base_url is not None:
        raise ValueError("sglang engine no longer supports server/http mode; use in-process Engine")

    _import_sglang(config.sglang_path)
    engine_module = importlib.import_module("sglang.srt.entrypoints.engine")
    engine_kwargs = {
        "model_path": model_config.path,
        **config.server_kwargs,
        **config.engine_kwargs,
    }
    if model_config.revision is not None:
        engine_kwargs.setdefault("revision", model_config.revision)
    return _SGLangPythonClient(engine=engine_module.Engine(**engine_kwargs))


def _import_sglang(sglang_path: str | None) -> Any:
    try:
        return importlib.import_module("sglang")
    except ModuleNotFoundError as exc:
        if not sglang_path:
            raise ModuleNotFoundError(
                "sglang is not importable; install it or configure `sglang_path`"
            ) from exc

        checkout_path = Path(sglang_path)
        if not checkout_path.exists():
            raise ModuleNotFoundError(
                f"sglang is not importable and the configured checkout path does not exist: {checkout_path}"
            ) from exc

        checkout_root = str(checkout_path)
        if checkout_root not in sys.path:
            sys.path.insert(0, checkout_root)

        return importlib.import_module("sglang")

def _normalize_sglang_response(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if not isinstance(payload, dict):
        raise TypeError("sglang response must be a dict or a list of dicts")

    text = payload.get("text")
    meta_info = payload.get("meta_info")
    if isinstance(text, list):
        meta_items = meta_info if isinstance(meta_info, list) else [meta_info] * len(text)
        return [
            {
                "text": item_text,
                "meta_info": item_meta or {},
            }
            for item_text, item_meta in zip(text, meta_items, strict=True)
        ]
    return [{"text": text, "meta_info": meta_info or {}}]


def _coerce_position_entries(raw_positions: Any) -> list[dict[str, Any]]:
    if raw_positions is None:
        return []
    output: list[dict[str, Any]] = []
    for entry in raw_positions:
        if entry is None:
            output.append({"score": None, "token_id": None, "text": None})
            continue
        output.append(_coerce_score_entry(entry))
    return output


def _coerce_nested_position_entries(raw_positions: Any) -> list[list[dict[str, Any]]]:
    if raw_positions is None:
        return []
    output: list[list[dict[str, Any]]] = []
    for position in raw_positions:
        if position is None:
            output.append([])
            continue
        output.append(
            [
                _coerce_score_entry(entry)
                for entry in position
                if entry is not None
            ]
        )
    return output


def _coerce_score_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        token_id = entry.get("token_id", entry.get("id"))
        score = entry.get("score", entry.get("logprob", entry.get("logit")))
        return {
            "score": float(score) if score is not None else None,
            "token_id": int(token_id) if token_id is not None else None,
            "text": entry.get("text"),
        }
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return {
            "score": float(entry[0]) if entry[0] is not None else None,
            "token_id": int(entry[1]) if entry[1] is not None else None,
            "text": entry[2] if len(entry) > 2 else None,
        }
    raise TypeError(f"unsupported sglang score entry: {entry!r}")


def _entries_to_score_map(entries: list[dict[str, Any]]) -> dict[int, float]:
    output: dict[int, float] = {}
    for entry in entries:
        token_id = entry.get("token_id")
        score = entry.get("score")
        if token_id is None or score is None:
            continue
        output[int(token_id)] = float(score)
    return output


def _deduplicate_preserve_order(token_ids: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in seen:
            continue
        seen.add(token_id)
        output.append(token_id)
    return output
