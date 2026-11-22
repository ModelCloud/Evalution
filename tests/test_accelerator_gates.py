# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch

from tests.accelerator_gates import requires_nvidia_cuda


def test_requires_nvidia_cuda_accepts_a_cuda_build(monkeypatch) -> None:
    """An NVIDIA build with a visible device passes the gate."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)

    assert requires_nvidia_cuda() is True


def test_requires_nvidia_cuda_rejects_a_hip_build(monkeypatch) -> None:
    """ROCm exposes the CUDA namespace, so the HIP version must veto the gate."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.16.26385")

    assert requires_nvidia_cuda() is False


def test_requires_nvidia_cuda_requires_a_visible_device(monkeypatch) -> None:
    """A CUDA build without a visible device still skips."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "hip", None)

    assert requires_nvidia_cuda() is False
