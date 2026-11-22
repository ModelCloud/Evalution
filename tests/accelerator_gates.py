# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import torch


def requires_nvidia_cuda() -> bool:
    """Return whether this process can run an NVIDIA-only CUDA kernel.

    ``torch.cuda.is_available()`` is true on ROCm because HIP reuses the CUDA
    device namespace, so tests that drive a CUDA-only kernel must also require
    a non-HIP build. Use this for a marker or a body-level skip; it is a
    runtime predicate, not a capability probe.
    """
    return torch.cuda.is_available() and torch.version.hip is None
