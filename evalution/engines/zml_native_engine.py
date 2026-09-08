# SPDX-License-Identifier: Apache-2.0
"""Direct C ABI access to the public ZML Llama/QVQ paged runtime."""

from __future__ import annotations

import ctypes as C
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .base import BaseInferenceSession, GenerationOutput, SharedEngineConfig
from .continuous import stream_request_results


class NativeRuntime:
    """Own one in-process ZML handle. Buffer sizes derive from create geometry."""

    def __init__(self, library, model, context, batch, capacity):
        self.batch, self.capacity = batch, capacity
        self.pages = (context + 15) // 16
        # Fail in Python with the dynamic-loader diagnostic instead of letting
        # PJRT crash while formatting a missing-cuBLAS-symbol error. A process
        # that already imported another CUDA stack must be restarted with a
        # consistent library search path; changing os.environ here is too late.
        self._cuda_libraries = []
        runfiles = os.environ.get("RUNFILES_DIR")
        if runfiles:
            for cuda_lib in Path(runfiles).glob(
                "*/platforms/cuda/source_sandbox/lib/libcublas.so.13"
            ):
                try:
                    self._cuda_libraries.append(C.CDLL(str(cuda_lib)))
                except OSError as exc:
                    raise RuntimeError(
                        f"Incompatible CUDA libraries: {exc}. Restart Python with "
                        "matching cuBLAS/cuBLASLt dependencies before importing Evalution. "
                        f"ZML's library directory is {cuda_lib.parent}; PyTorch may preload "
                        "its own absolute library paths regardless of LD_LIBRARY_PATH."
                    ) from exc
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.zml_llama_abi_version.restype = C.c_uint32
        if self.lib.zml_llama_abi_version() != 1:
            raise RuntimeError("Unsupported ZML Llama ABI version")
        self.lib.zml_llama_last_error.restype = C.c_char_p
        self.lib.zml_llama_create.argtypes = [
            C.c_char_p,
            C.c_uint32,
            C.c_uint32,
            C.c_uint32,
            C.POINTER(C.c_void_p),
        ]
        self.lib.zml_llama_create.restype = C.c_int
        u32, i32 = C.POINTER(C.c_uint32), C.POINTER(C.c_int32)
        self.lib.zml_llama_step.argtypes = [
            C.c_void_p,
            C.c_uint32,
            u32,
            u32,
            u32,
            u32,
            i32,
            i32,
            i32,
            u32,
        ]
        self.lib.zml_llama_step.restype = C.c_int
        self.lib.zml_llama_destroy.argtypes = [C.c_void_p]
        self.lib.zml_llama_destroy.restype = C.c_int
        self.handle = C.c_void_p()
        self._check(
            self.lib.zml_llama_create(
                str(model).encode(), context, batch, capacity, C.byref(self.handle)
            )
        )

    def _check(self, status):
        if status:
            raise RuntimeError(self.lib.zml_llama_last_error().decode())

    def step(
        self, *, prefill, tokens, positions, slots, indices, table, lengths, starts
    ):
        if not self.handle.value:
            raise RuntimeError("ZML runtime is closed")
        rows = self.capacity if prefill else self.batch
        fields = [
            (tokens, rows, C.c_uint32),
            (positions, rows, C.c_uint32),
            (slots, rows, C.c_uint32),
            (indices, self.batch, C.c_uint32),
            (table, (self.batch + 1) * self.pages, C.c_int32),
            (lengths, self.batch + 1, C.c_int32),
            (starts, self.batch + 2, C.c_int32),
        ]
        arrays = []
        for values, size, dtype in fields:
            if len(values) != size:
                raise ValueError(
                    f"Native ABI buffer needs {size} values, received {len(values)}"
                )
            if any(not isinstance(v, int) or v < 0 or v > 0x7FFFFFFF for v in values):
                raise ValueError("Native ABI values must be nonnegative int32 values")
            arrays.append((dtype * size)(*values))
        output = (C.c_uint32 * self.batch)()
        self._check(self.lib.zml_llama_step(self.handle, int(prefill), *arrays, output))
        return list(output)

    def close(self):
        if self.handle.value:
            self._check(self.lib.zml_llama_destroy(self.handle))
            self.handle = C.c_void_p()


@dataclass
class ZMLNative(SharedEngineConfig):
    library: str = ""
    batch_size: int = 8
    max_context_len: int = 2048
    token_batch_size: int = 128
    collect_timing: bool = False

    def build(self, model):
        if not self.library:
            raise ValueError("ZMLNative requires the path to libzml_llama.so")
        if (
            not 1 <= self.max_context_len <= 131072
            or not 1 <= self.batch_size <= 128
            or not self.batch_size
            <= self.token_batch_size
            <= min(self.max_context_len, 8192)
        ):
            raise ValueError("Invalid ZML batch/context geometry")
        self.resolved_engine = "ZMLNative"
        return ZMLNativeSession(self, model)


class ZMLNativeSession(BaseInferenceSession):
    def __init__(self, config, model):
        self.config, self.model_config = config, model
        if model.tokenizer is not None:
            self.tokenizer = model.tokenizer
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model.tokenizer_path or model.path,
                revision=model.revision,
                trust_remote_code=model.trust_remote_code,
                **model.tokenizer_kwargs,
            )
        generation_config = Path(model.path) / "generation_config.json"
        model_config = Path(model.path) / "config.json"
        settings = json.loads(
            (
                generation_config if generation_config.exists() else model_config
            ).read_text()
        )
        eos = settings.get("eos_token_id", self.tokenizer.eos_token_id)
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.runtime = NativeRuntime(
            config.library,
            model.path,
            config.max_context_len,
            config.batch_size,
            config.token_batch_size,
        )
        self._lock = threading.Lock()
        self.step_metrics = []

    def describe_execution(self):
        return {
            "generation_backend": "zml_native_abi",
            "abi_version": 1,
            "batch_size": self.config.batch_size,
            "token_batch_size": self.config.token_batch_size,
            "continuous_batching": True,
            "paged_attention": True,
            "attention_backend": "triton",
            "prefix_caching": False,
            "command_buffer_enabled": "ZML_LLAMA_EAGER" not in os.environ,
            "max_context_len": self.config.max_context_len,
        }

    def _prepare(self, item_id, request):
        if request.do_sample or request.temperature != 0 or request.num_beams != 1:
            raise ValueError("ZML native ABI currently supports greedy generation only")
        if request.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if request.input_ids is not None:
            ids = list(request.input_ids)
        elif request.rendered_prompt is not None:
            ids = self.tokenizer.encode(
                request.rendered_prompt, add_special_tokens=False
            )
        elif request.messages is not None:
            kwargs = dict(request.chat_template_kwargs or {})
            if request.tools is not None:
                kwargs["tools"] = request.tools
            ids = self.tokenizer.apply_chat_template(
                request.messages,
                tokenize=True,
                add_generation_prompt=request.add_generation_prompt,
                **kwargs,
            )
        else:
            ids = self.tokenizer.encode(request.prompt or "", add_special_tokens=False)
        if not ids or len(ids) + request.max_new_tokens > self.config.max_context_len:
            raise ValueError(
                "Prompt plus generation exceeds ZML context; truncation is disabled"
            )
        return {
            "id": item_id,
            "request": request,
            "ids": ids,
            "position": 0,
            "output": [],
            "pending": None,
        }

    def _output(self, state, reason):
        text = self.tokenizer.decode(state["output"], skip_special_tokens=True)
        stops = [text.find(s) for s in state["request"].stop if s and s in text]
        if stops:
            text = text[: min(stops)]
            reason = "stop"
        return GenerationOutput(
            prompt=state["request"].rendered_prompt or state["request"].prompt or "",
            text=text,
            metadata={
                "finish_reason": reason,
                "output_ids": list(state["output"]),
                "generation_backend": "zml_native_abi",
            },
        )

    def _run(self, requests, limit, put, *, request_queue=None, stop_event=None):
        lanes = [None] * self.config.batch_size
        iterator = iter(requests)
        exhausted = False
        batch, capacity, pages = (
            self.config.batch_size,
            self.config.token_batch_size,
            self.runtime.pages,
        )
        table = list(range((batch + 1) * pages))
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            for lane in range(limit):
                while lanes[lane] is None and not exhausted:
                    if request_queue is not None:
                        # Do not stall live decoding waiting for a slow producer.
                        live = any(s is not None for s in lanes)
                        item = request_queue.get(timeout_s=0 if live else 0.05)
                        if item is None:
                            exhausted = request_queue.closed
                            break
                        item_id, request = item
                    else:
                        try:
                            item_id, request = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                    state = self._prepare(item_id, request)
                    if request.max_new_tokens == 0:
                        put(item_id, self._output(state, "length"))
                    else:
                        lanes[lane] = state
            if not any(s is not None for s in lanes):
                if exhausted:
                    break
                continue
            prefill = any(
                s is not None and s["position"] < len(s["ids"]) for s in lanes
            )
            rows = capacity if prefill else batch
            tokens, positions, slots, indices, lengths, starts = (
                [],
                [],
                [],
                [0] * batch,
                [],
                [0],
            )
            advanced = [0] * batch
            for lane, state in enumerate(lanes):
                if state is None:
                    # Decode has one row per lane, including idle lanes; each
                    # idle lane uses its own page so it cannot corrupt live KV.
                    if not prefill:
                        tokens.append(0)
                        positions.append(0)
                        slots.append(lane * pages * 16)
                    lengths.append(0 if prefill else 1)
                    starts.append(len(tokens))
                    continue
                pos = state["position"]
                remaining_lanes = sum(s is not None for s in lanes[lane + 1 :])
                count = (
                    min(len(state["ids"]) - pos, rows - len(tokens) - remaining_lanes)
                    if pos < len(state["ids"])
                    else 1
                )
                new = (
                    state["ids"][pos : pos + count]
                    if pos < len(state["ids"])
                    else [state["pending"]]
                )
                tokens.extend(new)
                positions.extend(range(pos, pos + count))
                slots.extend(lane * pages * 16 + p for p in range(pos, pos + count))
                advanced[lane] = count
                lengths.append(pos + count)
                indices[lane] = len(tokens) - 1
                starts.append(len(tokens))
            padding = rows - len(tokens)
            tokens.extend([0] * padding)
            positions.extend(range(padding))
            slots.extend(batch * pages * 16 + p for p in range(padding))
            lengths.append(padding)
            starts.append(rows)
            measure = getattr(self.config, "collect_timing", False)
            if measure:
                prompt_tokens = sum(
                    advanced[lane]
                    for lane, state in enumerate(lanes)
                    if state is not None and state["position"] < len(state["ids"])
                )
                active_tokens = sum(advanced)
                started = time.perf_counter()
            sampled = self.runtime.step(
                prefill=prefill,
                tokens=tokens,
                positions=positions,
                slots=slots,
                indices=indices,
                table=table,
                lengths=lengths,
                starts=starts,
            )
            if measure:
                # step returns only after native output readiness. This measures
                # host ABI + transfers + GPU execution, not kernel-only time.
                self.step_metrics.append(
                    {
                        "phase": "prefill" if prefill else "decode",
                        "seconds": time.perf_counter() - started,
                        "prompt_tokens": prompt_tokens,
                        "decode_tokens": active_tokens - prompt_tokens,
                        "padding_rows": rows - active_tokens,
                    }
                )
            for lane, state in enumerate(lanes):
                if state is None:
                    continue
                state["position"] += advanced[lane]
                if state["position"] < len(state["ids"]):
                    continue
                token = sampled[lane]
                reason = "stop" if token in self.eos else None
                if reason is None:
                    state["output"].append(token)
                    state["pending"] = token
                    text = self.tokenizer.decode(
                        state["output"], skip_special_tokens=True
                    )
                    if any(s and s in text for s in state["request"].stop):
                        reason = "stop"
                    elif len(state["output"]) >= state["request"].max_new_tokens:
                        reason = "length"
                if reason:
                    put(state["id"], self._output(state, reason))
                    lanes[lane] = None

    def generate(self, requests, *, batch_size=None):
        results = dict(
            self.generate_continuous(enumerate(requests), batch_size=batch_size)
        )
        return [results[i] for i in range(len(requests))]

    def generate_continuous(self, requests, *, batch_size=None):
        limit = (
            self.config.batch_size
            if batch_size is None
            else min(batch_size, self.config.batch_size)
        )
        if limit < 1:
            raise ValueError("batch_size must be positive")

        def consume(stop_event, request_queue, put_result):
            with self._lock:
                self._run(
                    (),
                    limit,
                    put_result,
                    request_queue=request_queue,
                    stop_event=stop_event,
                )

        return stream_request_results(
            requests,
            producer_name="zml-native-input",
            consumer_name="zml-native-scheduler",
            process_requests=consume,
            require_non_main_thread=self.request_executor_requires_non_main_thread,
            request_queue_max_size=limit * 2,
        )

    def loglikelihood(self, requests, *, batch_size=None):
        raise NotImplementedError("ZML native ABI does not expose loglikelihood")

    def loglikelihood_rolling(self, requests, *, batch_size=None):
        raise NotImplementedError(
            "ZML native ABI does not expose rolling loglikelihood"
        )

    def gc(self):
        pass

    def close(self):
        with self._lock:
            self.runtime.close()
