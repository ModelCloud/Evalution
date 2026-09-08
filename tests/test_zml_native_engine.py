from types import SimpleNamespace

import pytest

from evalution.engines.base import GenerationRequest
from evalution.engines.zml_native_engine import ZMLNativeSession


def test_native_engine_yaml_registration():
    from evalution.engines import ZMLNative
    from evalution.yaml import _build_engine

    engine = _build_engine(
        {"name": "ZMLNative", "library": "/test/libzml_llama.so", "batch_size": 4}
    )
    assert isinstance(engine, ZMLNative)
    assert engine.batch_size == 4


class Tokenizer:
    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


class CacheOracle:
    """Simulate physical KV pages; fail if sequence history crosses requests."""

    pages = 4

    def __init__(self):
        self.cache = {}
        self.calls = []

    def step(self, **kw):
        self.calls.append(kw)
        for slot, token in zip(kw["slots"], kw["tokens"], strict=True):
            self.cache[slot] = token
        result = []
        for lane in range(3):
            start, end = kw["starts"][lane : lane + 2]
            if start == end:
                result.append(0)
                continue
            length = kw["lengths"][lane]
            history = [self.cache[lane * 64 + pos] for pos in range(length)]
            result.append(sum(history) % 100 + 1)
        return result


def session():
    s = object.__new__(ZMLNativeSession)
    s.config = SimpleNamespace(batch_size=3, token_batch_size=8, max_context_len=64)
    s.runtime = CacheOracle()
    s.tokenizer = Tokenizer()
    s.eos = {999}
    return s


def expected(prompt, maximum):
    history, output = list(prompt), []
    for _ in range(maximum):
        value = sum(history) % 100 + 1
        output.append(value)
        history.append(value)
    return output


def test_native_phase_timing_counts_real_tokens_not_padding(monkeypatch):
    s = session()
    s.config.collect_timing = True
    s.step_metrics = []
    ticks = iter(range(1000))
    monkeypatch.setattr(
        "evalution.engines.zml_native_engine.time.perf_counter", lambda: next(ticks)
    )
    results = {}
    s._run(
        enumerate(
            [
                GenerationRequest(input_ids=[1, 2], max_new_tokens=3),
                GenerationRequest(input_ids=[3, 4, 5], max_new_tokens=2),
            ]
        ),
        3,
        results.__setitem__,
    )
    assert sum(m["prompt_tokens"] for m in s.step_metrics) == 5
    assert sum(m["decode_tokens"] for m in s.step_metrics) == 3
    assert {m["phase"] for m in s.step_metrics} == {"prefill", "decode"}
    assert all(m["seconds"] == 1 for m in s.step_metrics)
    assert sum(m["padding_rows"] for m in s.step_metrics) > 0


def test_ragged_chunked_prefill_refill_and_request_isolation():
    s = session()
    prompts = [[1, 2], list(range(1, 20)), [7], [31, 17, 8], [4] * 13]
    limits = [2, 4, 6, 3, 1]
    results = {}
    s._run(
        enumerate(
            GenerationRequest(input_ids=p, max_new_tokens=n)
            for p, n in zip(prompts, limits)
        ),
        3,
        results.__setitem__,
    )
    for i, (prompt, maximum) in enumerate(zip(prompts, limits)):
        assert results[i].metadata["output_ids"] == expected(prompt, maximum)
    assert any(
        c["prefill"]
        and any(0 < b - a == 1 for a, b in zip(c["starts"], c["starts"][1:]))
        for c in s.runtime.calls
    )
    # Slot zero is refilled while another lane is still generating.
    assert any(c["positions"][0] == 0 and c["tokens"][0] == 31 for c in s.runtime.calls)
    assert any(not c["prefill"] for c in s.runtime.calls)


def test_zero_tokens_eos_and_stop():
    s = session()
    s.eos = {4}
    result = {}
    s._run(
        enumerate(
            [
                GenerationRequest(input_ids=[3], max_new_tokens=5),
                GenerationRequest(input_ids=[2], max_new_tokens=0),
                GenerationRequest(input_ids=[8], max_new_tokens=5, stop=["9"]),
            ]
        ),
        3,
        result.__setitem__,
    )
    assert result[0].text == ""
    assert result[1].metadata["output_ids"] == []
    assert result[2].text == ""


def test_slow_producer_does_not_block_live_decode():
    s = session()

    class SlowQueue:
        closed = False
        sent = False

        def get(self, *, timeout_s):
            if not self.sent:
                self.sent = True
                return 0, GenerationRequest(input_ids=[1, 2], max_new_tokens=4)
            if len(s.runtime.calls) < 4:
                assert timeout_s == 0, "live decoding must never wait for input"
                return None
            self.closed = True
            return None

    results = {}
    s._run((), 3, results.__setitem__, request_queue=SlowQueue())
    assert results[0].metadata["output_ids"] == expected([1, 2], 4)


@pytest.mark.parametrize(
    "generation_request",
    [
        GenerationRequest(input_ids=[], max_new_tokens=1),
        GenerationRequest(input_ids=[1] * 63, max_new_tokens=2),
        GenerationRequest(input_ids=[1], do_sample=True),
    ],
)
def test_reject_invalid_requests(generation_request):
    with pytest.raises(ValueError):
        session()._prepare(0, generation_request)
