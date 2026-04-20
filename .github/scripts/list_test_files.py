import argparse
import json
import re
from pathlib import Path

from common import normalize_test_file


def sort_key(path: Path, root: Path) -> tuple[int, str]:
    rel = path.relative_to(root)
    return (len(rel.parts), path.as_posix())


def list_test_files(tests_root: str = "tests", test_regex: str = "") -> list[str]:
    root = Path(tests_root)
    regex = re.compile(test_regex) if test_regex else None
    files: list[str] = []
    for path in sorted(root.rglob("test_*.py"), key=lambda item: sort_key(item, root)):
        rel = normalize_test_file(path.as_posix())
        if regex and not regex.search(rel):
            continue
        files.append(rel)
    return files


def split_evenly(files: list[str], group_count: int) -> list[list[str]]:
    if group_count <= 0:
        raise ValueError("group_count must be greater than 0")

    base_size, remainder = divmod(len(files), group_count)
    groups: list[list[str]] = []
    start = 0
    for index in range(group_count):
        size = base_size + (1 if index < remainder else 0)
        end = start + size
        groups.append(files[start:end])
        start = end
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-root", default="tests")
    parser.add_argument("--test-regex", default="")
    parser.add_argument("--group-count", type=int, default=1)
    args = parser.parse_args()

    files = list_test_files(args.tests_root, args.test_regex)
    if args.group_count == 1:
        print(json.dumps(files, ensure_ascii=False))
        return

    for group in split_evenly(files, args.group_count):
        print(json.dumps(group, ensure_ascii=False))


if __name__ == "__main__":
    main()
