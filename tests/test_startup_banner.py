# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
import importlib.util
from pathlib import Path


# Keep shared test fixtures and expectations explicit at module scope.
MODULE_PATH = Path(__file__).resolve().parents[1] / "evalution" / "_banner.py"
MODULE_SPEC = importlib.util.spec_from_file_location("evalution_banner_test_module", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None

# Keep shared test fixtures and expectations explicit at module scope.
banner_module = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(banner_module)


def test_build_startup_banner_aligns_versions() -> None:
    """Verify build startup banner aligns versions."""
    banner = banner_module.build_startup_banner(
        "LOGO\n",
        evalution_version="0.0.1",
        dependency_versions=[
            ("Transformers", "5.3.0"),
            ("Datasets", "4.8.3"),
            ("Torch", "2.12.0+cu130"),
            ("LogBar", "0.3.0"),
            ("PyPcre", "0.2.14"),
            ("PyYAML", "6.0.3"),
            ("Triton", "3.6.0"),
        ],
    )

    lines = banner.splitlines()
    assert lines[0] == "LOGO"
    assert lines[1].startswith("Evalution")
    assert lines[1].strip().endswith("0.0.1")
    assert lines[2].startswith("Transformers")
    assert lines[2].strip().endswith("5.3.0")
    assert lines[3].startswith("Datasets")
    assert lines[3].strip().endswith("4.8.3")
    assert lines[4].startswith("Torch")
    assert lines[4].strip().endswith("2.12.0+cu130")
    assert lines[5].startswith("LogBar")
    assert lines[5].strip().endswith("0.3.0")
    assert lines[6].startswith("PyPcre")
    assert lines[6].strip().endswith("0.2.14")
    assert lines[7].startswith("PyYAML")
    assert lines[7].strip().endswith("6.0.3")
    assert lines[8].startswith("Triton")
    assert lines[8].strip().endswith("3.6.0")
    assert {line.index(":") for line in lines[1:]} == {13}


def test_build_startup_banner_handles_project_only() -> None:
    """Verify build startup banner handles project only."""
    banner = banner_module.build_startup_banner(
        "LOGO\n",
        evalution_version="0.0.1",
        dependency_versions=[],
    )

    assert banner.splitlines() == [
        "LOGO",
        "Evalution : 0.0.1",
    ]


def test_get_startup_banner_resolves_dependency_versions(monkeypatch) -> None:
    """Verify get startup banner resolves dependency versions."""
    resolved_versions = {
        ("transformers",): "5.3.0",
        ("datasets",): "4.8.3",
        ("torch",): "2.12.0+cu130",
        ("logbar",): "0.3.0",
        ("PyPcre", "pypcre"): "0.2.14",
        ("PyYAML", "pyyaml"): "6.0.3",
        banner_module.TRITON_PACKAGE_CANDIDATES: "3.6.0",
    }

    def fake_resolve(package_names):
        """Support the surrounding tests with fake resolve."""
        return resolved_versions.get(tuple(package_names))

    monkeypatch.setattr(
        banner_module,
        "resolve_installed_package_version",
        fake_resolve,
    )

    banner = banner_module.get_startup_banner(
        "LOGO\n",
        evalution_version="0.0.1",
    )

    assert any(line.startswith("Evalution") and line.endswith("0.0.1") for line in banner.splitlines())
    assert any(line.startswith("Transformers") and line.endswith("5.3.0") for line in banner.splitlines())
    assert any(line.startswith("Datasets") and line.endswith("4.8.3") for line in banner.splitlines())
    assert any(line.startswith("Torch") and line.endswith("2.12.0+cu130") for line in banner.splitlines())
    assert any(line.startswith("LogBar") and line.endswith("0.3.0") for line in banner.splitlines())
    assert any(line.startswith("PyPcre") and line.endswith("0.2.14") for line in banner.splitlines())
    assert any(line.startswith("PyYAML") and line.endswith("6.0.3") for line in banner.splitlines())
    assert any(line.startswith("Triton") and line.endswith("3.6.0") for line in banner.splitlines())
