# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version


try:
    __version__ = package_version("Evalution")
except PackageNotFoundError:
    __version__ = "0.0.1"
