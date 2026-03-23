# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os


# Keep CUDA ordinal assignment stable across test runs so cuda:0/cuda:1 map by PCI bus order.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
