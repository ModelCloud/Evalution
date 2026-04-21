from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ci_common import normalize_test_file, test_requires_gpu


# Keep the reusable workflow matrix payload constrained to the single field the runner needs.
@dataclass(frozen=True)
class TestMatrixEntry:
    """Represent one test case entry consumed by the reusable unit-test workflow."""

    test_file: str

    def as_dict(self) -> dict[str, str]:
        return {"test_file": self.test_file}


# Test discovery only schedules pytest files from the repo's test tree.
TEST_FILE_GLOB = "test_*.py"


def compile_regex(pattern: str) -> re.Pattern[str] | None:
    if not pattern:
        return None
    return re.compile(pattern)


def normalize_rel_test_path(path: Path) -> str:
    return normalize_test_file(path.as_posix())


def matches_test_regex(compiled: re.Pattern[str] | None, rel_path: str) -> bool:
    if compiled is None:
        return True
    candidates = {
        rel_path,
        rel_path.removeprefix("tests/"),
        Path(rel_path).name,
        Path(rel_path).stem,
    }
    return any(compiled.search(candidate) for candidate in candidates)


def is_model_test(rel_path: str) -> bool:
    return rel_path.startswith("tests/models/")


def sort_key(rel_path: str) -> tuple[int, str]:
    return (len(Path(rel_path).parts), rel_path)


def list_tests(
    *,
    tests_root: str | Path,
    test_regex: str,
) -> tuple[list[str], list[str], list[str]]:
    root = Path(tests_root)
    compiled_regex = compile_regex(test_regex)

    cpu_tests: list[str] = []
    torch_tests: list[str] = []
    model_tests: list[str] = []

    for path in sorted(root.rglob(TEST_FILE_GLOB)):
        rel_path = normalize_rel_test_path(path)
        if not matches_test_regex(compiled_regex, rel_path):
            continue
        if is_model_test(rel_path):
            model_tests.append(rel_path)
            continue
        if test_requires_gpu(rel_path):
            torch_tests.append(rel_path)
            continue
        cpu_tests.append(rel_path)

    return (
        sorted(cpu_tests, key=sort_key),
        sorted(torch_tests, key=sort_key),
        sorted(model_tests, key=sort_key),
    )


def build_group_matrix(tests: list[str]) -> list[dict[str, str]]:
    return [TestMatrixEntry(test_file=test_file).as_dict() for test_file in tests]


def build_test_plan(*, tests_root: str | Path, test_regex: str) -> dict[str, list[dict[str, str]] | list[str]]:
    cpu_tests, torch_tests, model_tests = list_tests(
        tests_root=tests_root,
        test_regex=test_regex,
    )
    return {
        "cpu_files": cpu_tests,
        "torch_files": torch_tests,
        "model_files": model_tests,
        "cpu_matrix": build_group_matrix(cpu_tests),
        "torch_matrix": build_group_matrix(torch_tests),
        "model_matrix": build_group_matrix(model_tests),
    }


def list_matching_versions(package: str, version_spec: str) -> list[str]:
    # Defer optional dependencies so local test discovery can run on the stock runner image.
    import requests
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    specifier = SpecifierSet(version_spec)
    response = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=30)
    response.raise_for_status()
    data = response.json()
    matched = sorted(
        (Version(version) for version in data["releases"].keys() if Version(version) in specifier),
        reverse=True,
    )
    return [str(version) for version in matched]


def cmd_list_tests(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            build_test_plan(
                tests_root=args.tests_root,
                test_regex=args.test_regex,
            )
        )
    )
    return 0


def cmd_loop_versions(args: argparse.Namespace) -> int:
    print(json.dumps(list_matching_versions(args.package, args.version)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-tests")
    list_parser.add_argument("--tests-root", default="tests")
    list_parser.add_argument("--test-regex", default="")

    loop_versions_parser = subparsers.add_parser("loop-versions")
    loop_versions_parser.add_argument("package")
    loop_versions_parser.add_argument("version")

    args = parser.parse_args()
    if args.command == "list-tests":
        return cmd_list_tests(args)
    if args.command == "loop-versions":
        return cmd_loop_versions(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
