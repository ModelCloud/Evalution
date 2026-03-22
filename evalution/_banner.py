# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Iterable


ASCII_LOGO = r"""
┌─────────────┐    ┌────────────┐    ┌─────────────┐    ┌─────────────┐
│  Evalution  │ -> │   Models   │ -> │   Suites    │ -> │   Metrics   │
└─────────────┘    └────────────┘    └─────────────┘    └─────────────┘
"""

TRITON_PACKAGE_CANDIDATES = (
    "triton",
    "triton-windows",
    "pytorch_triton_xpu",
    "pytorch-triton-xpu",
)

DEPENDENCY_PACKAGE_CANDIDATES = (
    ("Transformers", ("transformers",)),
    ("Datasets", ("datasets",)),
    ("Torch", ("torch",)),
    ("LogBar", ("logbar",)),
    ("PyPcre", ("PyPcre", "pypcre")),
    ("PyYAML", ("PyYAML", "pyyaml")),
    ("Triton", TRITON_PACKAGE_CANDIDATES),
)


def resolve_installed_package_version(package_names: Iterable[str]) -> str | None:
    for package_name in package_names:
        try:
            resolved_version = package_version(package_name)
        except PackageNotFoundError:
            continue

        if resolved_version:
            return resolved_version

    return None


def build_startup_banner(
    ascii_logo: str,
    *,
    evalution_version: str,
    dependency_versions: Iterable[tuple[str, str]],
) -> str:
    version_rows = [("Evalution", evalution_version), *list(dependency_versions)]
    label_width = max(len(label) for label, _ in version_rows)
    formatted_rows = [
        f"{label:<{label_width}} : {value}" for label, value in version_rows
    ]
    return "\n".join([ascii_logo.rstrip("\n"), *formatted_rows])


def get_startup_banner(
    ascii_logo: str,
    *,
    evalution_version: str,
) -> str:
    dependency_versions = []
    for label, package_candidates in DEPENDENCY_PACKAGE_CANDIDATES:
        resolved_version = resolve_installed_package_version(package_candidates)
        if resolved_version:
            dependency_versions.append((label, resolved_version))

    return build_startup_banner(
        ascii_logo,
        evalution_version=evalution_version,
        dependency_versions=dependency_versions,
    )
