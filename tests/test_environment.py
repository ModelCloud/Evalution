# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os


def test_pytest_session_forces_cuda_device_order_to_pci_bus_id() -> None:
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
