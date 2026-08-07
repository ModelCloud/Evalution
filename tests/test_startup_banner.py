# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


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
    monkeypatch.setattr(
        banner_module,
        "_resolve_local_git_revision",
        lambda name: None,
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


def test_resolve_installed_package_version_appends_local_git_revision(monkeypatch) -> None:
    """Keep the exact package version while identifying the checked-out source revision."""
    repository_root = Path("/source/gptqmodel")
    monkeypatch.setattr(banner_module, "package_version", lambda name: "7.3.3")
    monkeypatch.setattr(
        banner_module,
        "_resolve_local_git",
        lambda name: (repository_root, "0123456789ab"),
    )
    monkeypatch.setattr(
        banner_module,
        "_is_enclosing_host_repository",
        lambda path: False,
    )

    resolved = banner_module.resolve_installed_package_version(("gptqmodel",))

    assert resolved == "7.3.3 (local-git-0123456789ab)"


def test_resolve_installed_package_version_preserves_text_for_enclosing_repository(
    monkeypatch,
) -> None:
    """Preserve package version text while omitting an enclosing host revision."""
    repository_root = Path("/source/host")
    monkeypatch.setattr(banner_module, "package_version", lambda name: "7.3.3+vendor.build")
    monkeypatch.setattr(
        banner_module,
        "_resolve_local_git",
        lambda name: (repository_root, "0123456789ab"),
    )
    monkeypatch.setattr(
        banner_module,
        "_is_enclosing_host_repository",
        lambda path: True,
    )

    resolved = banner_module.resolve_installed_package_version(("gptqmodel",))

    assert resolved == "7.3.3+vendor.build"
    assert "local-git" not in resolved


def test_get_startup_banner_preserves_explicit_evalution_version(monkeypatch) -> None:
    """Add source provenance without replacing the package's supplied version string."""
    monkeypatch.setattr(
        banner_module,
        "_resolve_local_git_revision",
        lambda name: "0123456789ab" if name == "Evalution" else None,
    )
    monkeypatch.setattr(
        banner_module,
        "resolve_installed_package_version",
        lambda names: None,
    )

    banner = banner_module.get_startup_banner(
        "LOGO\n",
        evalution_version="9.8.7.dev1",
    )

    assert banner.splitlines() == [
        "LOGO",
        "Evalution : 9.8.7.dev1 (local-git-0123456789ab)",
    ]


def test_resolve_local_git_revision_reads_pep610_editable_location(
    monkeypatch,
    tmp_path,
) -> None:
    """Detect editable projects even when their import hook does not expose a source spec."""
    source_root = tmp_path / "GPTQModel source"

    class FakeDistribution:
        """Expose the subset of distribution metadata used by the resolver."""

        def read_text(self, filename):
            """Return synthetic PEP 610 metadata."""
            if filename == "direct_url.json":
                return json.dumps(
                    {
                        "dir_info": {"editable": True},
                        "url": source_root.as_uri(),
                    }
                )
            return None

    monkeypatch.setattr(
        banner_module,
        "distributions",
        lambda **kwargs: [FakeDistribution()],
    )
    monkeypatch.setattr(banner_module.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(banner_module, "_repository_root", lambda path: path)
    observed_paths = []

    def fake_git_revision(path):
        """Capture the decoded editable source path."""
        observed_paths.append(path)
        return "abcdef012345"

    monkeypatch.setattr(banner_module, "_git_revision", fake_git_revision)

    revision = banner_module._resolve_local_git_revision("gptqmodel")

    assert revision == "abcdef012345"
    assert observed_paths == [source_root.resolve()]


def test_resolve_local_git_revision_detects_module_linked_to_checkout(
    monkeypatch,
    tmp_path,
) -> None:
    """Detect local source made importable through PYTHONPATH or a .pth link."""
    package_root = tmp_path / "vllm-source" / "vllm"
    monkeypatch.setattr(banner_module, "distributions", lambda **kwargs: [])
    monkeypatch.setattr(
        banner_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=str(package_root / "__init__.py"),
            submodule_search_locations=[str(package_root)],
        ),
    )
    monkeypatch.setattr(
        banner_module,
        "_repository_root",
        lambda path: path,
    )
    monkeypatch.setattr(
        banner_module,
        "_git_revision",
        lambda path: "fedcba987654" if path == package_root.resolve() else None,
    )

    revision = banner_module._resolve_local_git_revision("vllm")

    assert revision == "fedcba987654"


def test_repository_root_does_not_escape_site_packages(tmp_path) -> None:
    """Do not attribute installed wheels to a Git-managed Python environment."""
    pyenv_root = tmp_path / "pyenv"
    (pyenv_root / ".git").mkdir(parents=True)
    package_root = (
        pyenv_root
        / "versions"
        / "3.14.2t"
        / "lib"
        / "python3.14t"
        / "site-packages"
        / "transformers"
    )
    package_root.mkdir(parents=True)

    assert banner_module._repository_root(package_root) is None


def test_repository_root_keeps_linked_source_checkout(tmp_path) -> None:
    """Continue reporting modules that resolve directly into a source checkout."""
    repository_root = tmp_path / "transformers-source"
    (repository_root / ".git").mkdir(parents=True)
    package_root = repository_root / "src" / "transformers"
    package_root.mkdir(parents=True)

    assert banner_module._repository_root(package_root) == repository_root


def test_collect_score_report_versions_keeps_missing_runtime_keys(monkeypatch) -> None:
    """Emit null entries for absent optional engines so reports have a stable schema."""
    resolved_versions = {
        ("Evalution",): "0.0.11",
        ("gptqmodel",): "7.3.3",
        ("torch",): "2.13.0",
    }
    monkeypatch.setattr(
        banner_module,
        "resolve_installed_package_version",
        lambda names: resolved_versions.get(tuple(names)),
    )

    versions = banner_module.collect_score_report_versions()

    assert versions["evalution"] == "0.0.11"
    assert versions["gptqmodel"] == "7.3.3"
    assert versions["vllm"] is None
    assert versions["sglang"] is None
    assert versions["transformers"] is None
    assert versions["huggingface_hub"] is None
    assert versions["torch"] == "2.13.0"
