# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import importlib.util
import json
from importlib.metadata import (
    PackageNotFoundError,
    distributions,
    version as package_version,
)
from pathlib import Path
import subprocess
from typing import Iterable
from urllib.parse import unquote, urlparse


# Keep module-level state explicit for this module.
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
    ("GPTQModel", ("gptqmodel",)),
    ("vLLM", ("vllm",)),
    ("SGLang", ("sglang",)),
    ("Transformers", ("transformers",)),
    ("HuggingFace Hub", ("huggingface-hub", "huggingface_hub")),
    ("Datasets", ("datasets",)),
    ("Torch", ("torch",)),
    ("LogBar", ("logbar",)),
    ("PyPcre", ("PyPcre", "pypcre")),
    ("PyYAML", ("PyYAML", "pyyaml")),
    ("Triton", TRITON_PACKAGE_CANDIDATES),
)

SCORE_REPORT_PACKAGE_CANDIDATES = (
    ("evalution", ("Evalution",)),
    ("gptqmodel", ("gptqmodel",)),
    ("vllm", ("vllm",)),
    ("sglang", ("sglang",)),
    ("transformers", ("transformers",)),
    ("huggingface_hub", ("huggingface-hub", "huggingface_hub")),
    ("datasets", ("datasets",)),
    ("torch", ("torch",)),
    ("triton", TRITON_PACKAGE_CANDIDATES),
)

INSTALLED_PACKAGE_BOUNDARY_NAMES = frozenset({"site-packages", "dist-packages"})


def resolve_installed_package_version(package_names: Iterable[str]) -> str | None:
    """Resolve an installed version and identify a linked local Git checkout."""
    for package_name in package_names:
        try:
            resolved_version = package_version(package_name)
        except PackageNotFoundError:
            continue

        if not resolved_version:
            continue

        local_git = _resolve_local_git(package_name)
        if local_git is not None:
            repository_root, local_revision = local_git
            if _is_enclosing_host_repository(repository_root):
                return resolved_version
            return f"{resolved_version} (local-git-{local_revision})"
        return resolved_version

    return None


def collect_score_report_versions() -> dict[str, str | None]:
    """Collect reproducibility-critical package versions for serialized score reports."""
    return {
        report_name: resolve_installed_package_version(package_candidates)
        for report_name, package_candidates in SCORE_REPORT_PACKAGE_CANDIDATES
    }


def _resolve_local_git(package_name: str) -> tuple[Path, str] | None:
    """Return the repository root and revision for linked local source code."""
    source_paths = [*_direct_local_distribution_paths(package_name)]
    source_paths.extend(_module_source_paths(package_name))

    visited: set[Path] = set()
    for source_path in source_paths:
        try:
            normalized_path = source_path.resolve()
        except OSError:
            normalized_path = source_path
        if normalized_path in visited:
            continue
        visited.add(normalized_path)

        repository_root = _repository_root(normalized_path)
        if repository_root is None:
            continue
        revision = _git_revision(repository_root)
        if revision is not None:
            return repository_root, revision
    return None


def _resolve_local_git_revision(package_name: str) -> str | None:
    """Return only the revision portion of linked local Git provenance."""
    local_git = _resolve_local_git(package_name)
    return local_git[1] if local_git is not None else None


def _direct_local_distribution_paths(package_name: str) -> list[Path]:
    """Read editable and local-project locations from PEP 610 metadata."""
    paths: list[Path] = []
    try:
        installed_distributions = distributions(name=package_name)
    except Exception:
        return paths

    for distribution in installed_distributions:
        try:
            direct_url_text = distribution.read_text("direct_url.json")
        except Exception:
            continue
        if not direct_url_text:
            continue

        try:
            direct_url = json.loads(direct_url_text)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(direct_url, dict):
            continue
        source_path = _path_from_file_url(direct_url.get("url"))
        if source_path is None:
            continue
        dir_info = direct_url.get("dir_info")
        is_editable = isinstance(dir_info, dict) and dir_info.get("editable") is True
        if is_editable or source_path.is_dir():
            paths.append(source_path)
    return paths


def _module_source_paths(package_name: str) -> list[Path]:
    """Locate source linked through PYTHONPATH, a .pth file, or a legacy editable install."""
    module_names = {package_name.replace("-", "_").lower()}
    try:
        installed_distributions = distributions(name=package_name)
    except Exception:
        installed_distributions = ()
    for distribution in installed_distributions:
        try:
            top_level_text = distribution.read_text("top_level.txt") or ""
        except Exception:
            continue
        module_names.update(
            line.strip()
            for line in top_level_text.splitlines()
            if line.strip() and line.strip().isidentifier()
        )

    paths: list[Path] = []
    for module_name in module_names:
        try:
            module_spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
        if module_spec is None:
            continue

        origin = module_spec.origin
        if origin and origin not in {"built-in", "frozen"}:
            paths.append(Path(origin).parent)
        for location in module_spec.submodule_search_locations or ():
            paths.append(Path(location))
    return paths


def _path_from_file_url(value: object) -> Path | None:
    """Convert one local file URL from direct_url.json into a filesystem path."""
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    return Path(unquote(parsed.path))


def _repository_root(source_path: Path) -> Path | None:
    """Find the nearest repository without treating ordinary site-packages as source trees."""
    candidate = source_path if source_path.is_dir() else source_path.parent
    for parent in (candidate, *candidate.parents):
        if parent.name in INSTALLED_PACKAGE_BOUNDARY_NAMES:
            return None
        if (parent / ".git").exists():
            return parent
    return None


def _is_enclosing_host_repository(repository_root: Path) -> bool:
    """Protect provenance belonging to a repository that contains this project checkout."""
    evalution_root = _repository_root(Path(__file__).resolve())
    if evalution_root is None or repository_root == evalution_root:
        return False
    return evalution_root.is_relative_to(repository_root)


def _git_revision(repository_root: Path) -> str | None:
    """Resolve a stable short revision for one known repository root."""
    if not (repository_root / ".git").exists():
        return None

    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "--short=12", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None

    revision = completed.stdout.strip().lower()
    if not revision or any(character not in "0123456789abcdef" for character in revision):
        return None
    return revision


def build_startup_banner(
    ascii_logo: str,
    *,
    evalution_version: str,
    dependency_versions: Iterable[tuple[str, str]],
) -> str:
    """Build startup banner."""
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
    """Get startup banner."""
    evalution_revision = _resolve_local_git_revision("Evalution")
    displayed_evalution_version = evalution_version
    if evalution_revision is not None:
        displayed_evalution_version = (
            f"{displayed_evalution_version} (local-git-{evalution_revision})"
        )
    dependency_versions = []
    for label, package_candidates in DEPENDENCY_PACKAGE_CANDIDATES:
        resolved_version = resolve_installed_package_version(package_candidates)
        if resolved_version:
            dependency_versions.append((label, resolved_version))

    return build_startup_banner(
        ascii_logo,
        evalution_version=displayed_evalution_version,
        dependency_versions=dependency_versions,
    )
