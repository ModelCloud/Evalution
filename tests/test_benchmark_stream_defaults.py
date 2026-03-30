from __future__ import annotations

from pathlib import Path


def test_only_explicit_non_stream_benchmarks_keep_stream_false_defaults() -> None:
    benchmark_dir = Path(__file__).resolve().parents[1] / "evalution" / "benchmarks"
    explicit_non_stream = {
        "aexams.py",
        "coqa.py",
        "race.py",
    }

    offenders: list[str] = []
    for path in sorted(benchmark_dir.glob("*.py")):
        if path.name in explicit_non_stream:
            continue
        text = path.read_text()
        if "stream: bool = False" in text:
            offenders.append(path.name)

    assert offenders == []
