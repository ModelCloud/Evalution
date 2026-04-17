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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-root", default="tests")
    parser.add_argument("--test-regex", default="")
    args = parser.parse_args()

    print(json.dumps(list_test_files(args.tests_root, args.test_regex), ensure_ascii=False))


if __name__ == "__main__":
    main()
