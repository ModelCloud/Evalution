# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import threading

import pytest
import torch

from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.engines.transformers_common import (
    _AUTO_BATCH_OOM_MAX_RETRIES,
    BaseTransformerSession,
    _is_out_of_memory_error,
)


def _requests(count: int) -> list[GenerationRequest]:
    return [
        GenerationRequest(prompt=f"prompt-{index}", max_new_tokens=1)
        for index in range(count)
    ]


class _RecordingSession(BaseTransformerSession):
    """Session double that records the batch sizes passed to `model.generate()`."""

    def __init__(self, oom_sizes: set[int]) -> None:
        self.oom_sizes = set(oom_sizes)
        self.batch_sizes: list[int] = []
        self._state_lock = threading.RLock()
        self._oom_batch_size_ceiling = None

    def _generate_standard_batch(self, batch: list[GenerationRequest]) -> list[GenerationOutput]:
        self.batch_sizes.append(len(batch))
        if len(batch) in self.oom_sizes:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return [
            GenerationOutput(prompt=request.prompt or "", text=f"generated-{request.prompt}")
            for request in batch
        ]


class _FailingSession(_RecordingSession):
    """Session double that raises a caller-supplied error on every batch."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(set())
        self.error = error

    def _generate_standard_batch(self, batch: list[GenerationRequest]) -> list[GenerationOutput]:
        self.batch_sizes.append(len(batch))
        raise self.error


def test_generate_standard_halves_batch_and_retries_after_oom() -> None:
    session = _RecordingSession(oom_sizes={8})
    requests = _requests(8)

    outputs = BaseTransformerSession._generate_standard(session, requests, batch_size=8)

    assert session.batch_sizes == [8, 4, 4]
    assert [output.prompt for output in outputs] == [request.prompt for request in requests]
    assert session._oom_batch_size_ceiling == 4


def test_generate_standard_keeps_results_in_submission_order_after_retry() -> None:
    session = _RecordingSession(oom_sizes={4})
    requests = _requests(10)

    outputs = BaseTransformerSession._generate_standard(session, requests, batch_size=4)

    assert session.batch_sizes == [4, 2, 2, 2, 2, 2]
    assert [output.prompt for output in outputs] == [request.prompt for request in requests]


def test_generate_standard_propagates_oom_at_batch_size_one() -> None:
    session = _RecordingSession(oom_sizes={1})

    with pytest.raises(torch.OutOfMemoryError):
        BaseTransformerSession._generate_standard(session, _requests(1), batch_size=1)

    assert session.batch_sizes == [1]


def test_generate_standard_bounds_the_number_of_retries() -> None:
    session = _RecordingSession(oom_sizes={1, 2, 4, 8, 16, 32, 64})

    with pytest.raises(torch.OutOfMemoryError):
        BaseTransformerSession._generate_standard(session, _requests(64), batch_size=64)

    assert len(session.batch_sizes) == _AUTO_BATCH_OOM_MAX_RETRIES + 1
    assert session.batch_sizes == [64, 32, 16, 8, 4, 2, 1]


def test_generate_standard_does_not_retry_unrelated_errors() -> None:
    session = _FailingSession(RuntimeError("shape mismatch"))

    with pytest.raises(RuntimeError, match="shape mismatch"):
        BaseTransformerSession._generate_standard(session, _requests(4), batch_size=4)

    assert session.batch_sizes == [4]


def test_oom_ceiling_caps_later_resolutions() -> None:
    session = _RecordingSession(oom_sizes=set())
    session._oom_batch_size_ceiling = None

    assert BaseTransformerSession._apply_oom_batch_ceiling(session, 16) == 16

    BaseTransformerSession._record_oom_batch_ceiling(session, 4)
    assert BaseTransformerSession._apply_oom_batch_ceiling(session, 16) == 4

    # A later OOM at a larger size must not raise the ceiling again.
    BaseTransformerSession._record_oom_batch_ceiling(session, 8)
    assert BaseTransformerSession._apply_oom_batch_ceiling(session, 16) == 4

    BaseTransformerSession._record_oom_batch_ceiling(session, 2)
    assert BaseTransformerSession._apply_oom_batch_ceiling(session, 16) == 2


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (torch.OutOfMemoryError("boom"), True),
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
        (RuntimeError("HIP out of memory"), True),
        (RuntimeError("MPS out of memory"), True),
        (RuntimeError("shape mismatch"), False),
        (ValueError("CUDA out of memory"), False),
    ],
)
def test_is_out_of_memory_error(error: BaseException, expected: bool) -> None:
    assert _is_out_of_memory_error(error) is expected
