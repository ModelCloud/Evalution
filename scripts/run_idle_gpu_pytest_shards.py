#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue


@dataclass(frozen=True)
class GpuSnapshot:
    """Describe one PCI-bus-ordered GPU as reported by nvidia-smi."""

    index: int
    bus_id: str
    name: str
    memory_used_mib: int
    gpu_util_percent: int
    memory_util_percent: int

    @property
    def is_idle(self) -> bool:
        """Apply the repo policy: idle means < 1 GiB VRAM used and no visible activity."""
        return (
            self.memory_used_mib < 1024
            and self.gpu_util_percent == 0
            and self.memory_util_percent == 0
        )


@dataclass(frozen=True)
class TestResult:
    """Capture the outcome for one standalone pytest file run."""

    gpu_index: int
    test_path: str
    returncode: int
    elapsed_s: float
    log_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run standalone pytest file shards concurrently across GPUs that are idle under "
            "Evalution's AGENTS policy."
        )
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Optional explicit test files. Defaults to every tests/**/test_*.py file.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for child pytest processes. Defaults to this interpreter.",
    )
    parser.add_argument(
        "--log-dir",
        default="artifacts/pytest_idle_gpu_shards",
        help="Directory where per-test logs are written.",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=None,
        help="Optional cap on how many idle GPUs to use.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop queueing new test files after the first failing shard.",
    )
    return parser.parse_args()


def _query_gpus() -> list[GpuSnapshot]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,name,memory.used,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    gpus: list[GpuSnapshot] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"could not parse nvidia-smi GPU line: {line!r}")
        gpus.append(
            GpuSnapshot(
                index=int(fields[0]),
                bus_id=fields[1],
                name=fields[2],
                memory_used_mib=int(fields[3]),
                gpu_util_percent=int(fields[4]),
                memory_util_percent=int(fields[5]),
            )
        )
    if not gpus:
        raise RuntimeError("nvidia-smi did not report any GPUs")
    return gpus


def _discover_tests(explicit_tests: list[str]) -> list[Path]:
    if explicit_tests:
        test_paths = [Path(test).resolve() for test in explicit_tests]
    else:
        test_paths = sorted(Path("tests").resolve().rglob("test_*.py"))
    missing = [path for path in test_paths if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"requested test files do not exist: {formatted}")
    return test_paths


def _safe_log_name(test_path: Path) -> str:
    """Turn a repository-relative test path into a log filename without shell escaping."""
    return "__".join(test_path.parts) + ".log"


def _run_worker(
    *,
    gpu: GpuSnapshot,
    python_executable: str,
    work_queue: Queue[Path | None],
    results: list[TestResult],
    results_lock: threading.Lock,
    log_dir: Path,
    root_dir: Path,
    stop_on_failure: bool,
    failure_event: threading.Event,
) -> None:
    """Keep one GPU leased to one worker thread so each pytest child sees exactly one device."""
    while True:
        if stop_on_failure and failure_event.is_set():
            break
        try:
            item = work_queue.get(timeout=0.2)
        except Empty:
            if failure_event.is_set():
                break
            continue
        if item is None:
            work_queue.task_done()
            break

        relative_test_path = item.relative_to(root_dir)
        log_path = log_dir / _safe_log_name(relative_test_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu.index),
            "PYTHON_GIL": "0",
        }
        command = [python_executable, "-m", "pytest", "-q", str(relative_test_path)]
        start_time = time.monotonic()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(
                f"# gpu_index={gpu.index} bus_id={gpu.bus_id} name={gpu.name}\n"
                f"# command={' '.join(command)}\n"
                f"# started_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n"
            )
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=root_dir,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed_s = time.monotonic() - start_time
        result = TestResult(
            gpu_index=gpu.index,
            test_path=str(relative_test_path),
            returncode=completed.returncode,
            elapsed_s=elapsed_s,
            log_path=log_path,
        )
        with results_lock:
            results.append(result)
        if completed.returncode != 0:
            failure_event.set()
        work_queue.task_done()


def _print_summary(*, gpus: list[GpuSnapshot], results: list[TestResult], total_tests: int) -> int:
    failures = [result for result in results if result.returncode != 0]
    print("# idle GPUs")
    for gpu in gpus:
        print(
            f"- gpu {gpu.index} {gpu.name} {gpu.bus_id}: "
            f"{gpu.memory_used_mib} MiB used, gpu {gpu.gpu_util_percent}%, mem {gpu.memory_util_percent}%"
        )
    print(f"# completed {len(results)}/{total_tests} test files")
    if failures:
        print("# failures")
        for failure in sorted(failures, key=lambda item: item.test_path):
            print(
                f"- {failure.test_path} on gpu {failure.gpu_index} "
                f"failed in {failure.elapsed_s:.1f}s (log: {failure.log_path})"
            )
        return 1
    print("# all test files passed")
    return 0


def main() -> int:
    args = _parse_args()
    root_dir = Path.cwd().resolve()
    tests = _discover_tests(args.tests)
    idle_gpus = [gpu for gpu in _query_gpus() if gpu.is_idle]
    if args.max_gpus is not None:
        idle_gpus = idle_gpus[: args.max_gpus]
    if not idle_gpus:
        raise RuntimeError("no GPUs are idle under the configured policy")

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    work_queue: Queue[Path | None] = Queue()
    for test_path in tests:
        work_queue.put(test_path)
    for _gpu in idle_gpus:
        work_queue.put(None)

    results: list[TestResult] = []
    results_lock = threading.Lock()
    failure_event = threading.Event()
    workers = [
        threading.Thread(
            target=_run_worker,
            kwargs={
                "gpu": gpu,
                "python_executable": args.python,
                "work_queue": work_queue,
                "results": results,
                "results_lock": results_lock,
                "log_dir": log_dir,
                "root_dir": root_dir,
                "stop_on_failure": args.stop_on_failure,
                "failure_event": failure_event,
            },
            daemon=True,
            name=f"gpu-worker-{gpu.index}",
        )
        for gpu in idle_gpus
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    return _print_summary(gpus=idle_gpus, results=results, total_tests=len(tests))


if __name__ == "__main__":
    raise SystemExit(main())
