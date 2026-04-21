from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
from dataclasses import asdict, dataclass
from pathlib import Path

from ci_common import (
    append_github_env,
    append_github_output,
    build_job_request,
    extract_gpu_ids,
    normalize_base_url,
    normalize_test_file,
    request_json,
    test_requires_gpu,
    to_safe_name,
)

# GPTQModel engine tests need a pinned CPU wheel because upstream CPU artifacts are version-sensitive.
TORCHAO_CPU_WHEEL = (
    "https://download.pytorch.org/whl/cpu/"
    "torchao-0.17.0%2Bcpu-py3-none-any.whl"
    "#sha256=6c0ce8b506c72be4efb1f0c6fd1679cb58145efebb20d51ac1adf7a7b3ebb872"
)
# CI uses a non-zero exit rewrite so GitHub summaries clearly distinguish test process failures.
ERROR_EXIT_CODE = 22


# Unit test metadata is computed once per matrix entry and exported to GitHub env/output files.
@dataclass(frozen=True)
class UnitTestConfig:
    """Describe the Python/runtime settings needed by one test file."""

    test_file: str
    safe_name: str
    requires_gpu: bool
    python_version: str
    uv_python: str


def resolve_unit_test_config(test_file: str) -> UnitTestConfig:
    normalized = normalize_test_file(test_file)
    python_version = "3.14t"
    uv_python = "3.14t"

    if normalized == "tests/test_tensorrt_llm_engine.py":
        python_version = "3.12"
        uv_python = "3.12"

    return UnitTestConfig(
        test_file=normalized,
        safe_name=to_safe_name(normalized),
        requires_gpu=test_requires_gpu(normalized),
        python_version=python_version,
        uv_python=uv_python,
    )


def export_unit_test_metadata(test_file: str) -> None:
    config = resolve_unit_test_config(test_file)

    append_github_env("SAFE_NAME", config.safe_name)
    append_github_env("TEST_REQUIRES_GPU", str(config.requires_gpu).lower())
    append_github_env("PYTHON_VERSION", config.python_version)
    append_github_env("UV_PYTHON", config.uv_python)

    append_github_output("safe-name", config.safe_name)
    append_github_output("requires-gpu", str(config.requires_gpu).lower())
    append_github_output("python-version", config.python_version)
    append_github_output("uv-python", config.uv_python)

    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def uv_install(*packages: str, upgrade: bool = False) -> None:
    if not packages:
        return
    cmd = ["uv", "pip", "install"]
    if upgrade:
        cmd.append("-U")
    cmd.extend(packages)
    run(cmd)


def install_flash_attn(uv_python: str, runner: str) -> None:
    if uv_python == "3.14t":
        uv_install(f"http://{runner}/files/flash_attn/flash_attn-2.8.4-cp314-cp314t-linux_x86_64.whl")
        return

    if uv_python == "3.12":
        append_github_env("EVALUTION_SKIP_GIL_CHECK", "1")
        uv_install("tensorrt_llm", upgrade=True)
        uv_install(f"http://{runner}/files/flash_attn/flash_attn-2.8.4-cp312-cp312-linux_x86_64.whl")
        return

    uv_install("flash-attn")
    run(["uv", "pip", "show", "flash-attn"])


def install_test_specific_deps(test_file: str) -> None:
    if test_file != "tests/test_gptqmodel_engine.py":
        return

    uv_install("accelerate", upgrade=True)
    uv_install(TORCHAO_CPU_WHEEL, upgrade=True)

    print("== installing gptqmodel ==")
    uv_install("gptqmodel", upgrade=True)
    run(["uv", "pip", "show", "gptqmodel"])


def install_test_deps(test_file: str, runner: str, uv_python: str, install_project: bool) -> None:
    normalized = normalize_test_file(test_file)

    if install_project:
        uv_install(".")
        uv_install("pytest", "datasets", "rouge_score", "sglang", "pybase64", upgrade=True)

    install_flash_attn(uv_python, runner)
    install_test_specific_deps(normalized)


def kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def start_keepalive_monitor(
    *,
    proc: subprocess.Popen[str],
    keepalive_endpoint: str,
    keepalive_payload: dict[str, object],
    expected_gpu_ids: str,
    interval_sec: int,
) -> tuple[threading.Thread, threading.Event, dict[str, int]]:
    stop_event = threading.Event()
    state = {"forced_exit_code": 0}

    def worker() -> None:
        print(f"start to keep alive... {keepalive_endpoint}")
        while not stop_event.wait(interval_sec):
            try:
                response = request_json(
                    keepalive_endpoint,
                    method="POST",
                    body=keepalive_payload,
                    timeout=10,
                )
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                print(f"Keepalive request failed: {exc}")
                continue

            resp = extract_gpu_ids(response)
            if resp == "-1":
                print(f"Server returned {resp}, terminating job...")
                state["forced_exit_code"] = 3
                kill_process_group(proc)
                stop_event.set()
                return
            if expected_gpu_ids and resp != expected_gpu_ids:
                print(f"Keepalive returned mismatched GPUs {resp}, expected {expected_gpu_ids}.")
                state["forced_exit_code"] = 3
                kill_process_group(proc)
                stop_event.set()
                return
            print("gpu is kept alive...")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread, stop_event, state


def stream_process_output(proc: subprocess.Popen[str], log_file: Path) -> int:
    assert proc.stdout is not None
    with log_file.open("w", encoding="utf-8") as fh:
        for line in proc.stdout:
            print(line, end="")
            fh.write(line)
    return proc.wait()


def log_python_and_pytest_resolution() -> None:
    print(f"sys.executable={sys.executable}")
    print(f"sys.version={sys.version}")
    pytest_path = shutil.which("pytest")
    print(f"which pytest={pytest_path}")
    if not pytest_path:
        return
    try:
        with open(pytest_path, encoding="utf-8") as fh:
            first_line = fh.readline().rstrip()
    except OSError as exc:
        print(f"failed to read pytest launcher: {exc}")
        return
    print(f"pytest shebang={first_line}")


def run_test(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    if args.clear_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
        print("CUDA_VISIBLE_DEVICES=")

    print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}")

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    test_file = normalize_test_file(args.test_file)
    if not Path("tests").is_dir():
        print("tests/ directory not found.")
        return 1
    if not Path(test_file).is_file():
        print(f"test file not found: {test_file}")
        return 1
    safe_name = to_safe_name(test_file)
    log_file = artifacts_dir / f"{safe_name}.log"
    junitxml = artifacts_dir / f"{safe_name}.xml"

    pytest_cmd = ["pytest", "--durations=0", test_file, f"--junitxml={junitxml}"]
    log_python_and_pytest_resolution()
    print(f"+ {' '.join(pytest_cmd)}")

    proc = subprocess.Popen(
        pytest_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        start_new_session=True,
    )

    keepalive_endpoint = f"{normalize_base_url(args.base_url)}/keepalive"
    keepalive_payload = build_job_request(
        runner_name=args.runner,
        run_id=args.run_id,
        test_name=test_file,
    )

    monitor_thread = None
    monitor_stop = None
    monitor_state = {"forced_exit_code": 0}
    if env.get("CUDA_VISIBLE_DEVICES", ""):
        monitor_thread, monitor_stop, monitor_state = start_keepalive_monitor(
            proc=proc,
            keepalive_endpoint=keepalive_endpoint,
            keepalive_payload=keepalive_payload,
            expected_gpu_ids=env.get("CUDA_VISIBLE_DEVICES", ""),
            interval_sec=args.monitor_interval_sec,
        )

    start_time = time.time()
    try:
        return_code = stream_process_output(proc, log_file)
    finally:
        if monitor_stop is not None:
            print("trap cleanup EXIT...")
            monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5)

    if monitor_state["forced_exit_code"]:
        append_github_env("ERROR", str(ERROR_EXIT_CODE))
        return ERROR_EXIT_CODE

    if return_code != 0:
        append_github_env("ERROR", str(ERROR_EXIT_CODE))
        print(f"pipe status wrong: {return_code}")
        return ERROR_EXIT_CODE

    execution_time = int(time.time() - start_time)
    print(f"{execution_time // 60}m {execution_time % 60}s")

    try:
        for entry in sorted(artifacts_dir.iterdir()):
            stat = entry.stat()
            print(f"{stat.st_size:>10} {entry.name}")
    except OSError as exc:
        print(f"Failed to list artifact dir: {exc}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_metadata_parser = subparsers.add_parser("set-metadata")
    set_metadata_parser.add_argument("--test-file", required=True)

    install_deps_parser = subparsers.add_parser("install-deps")
    install_deps_parser.add_argument("--test-file", required=True)
    install_deps_parser.add_argument("--runner", default=os.environ.get("RUNNER", "10.0.13.31"))
    install_deps_parser.add_argument("--uv-python", default=os.environ.get("UV_PYTHON", ""))
    install_deps_parser.add_argument("--install-project", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--base-url", required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--test-file", required=True)
    run_parser.add_argument("--runner", required=True)
    run_parser.add_argument("--gpu-id", default="")
    run_parser.add_argument("--monitor-interval-sec", type=int, default=60)
    run_parser.add_argument("--artifacts-dir", default="artifacts")
    run_parser.add_argument("--clear-cuda", action="store_true")

    args = parser.parse_args()
    if args.command == "set-metadata":
        export_unit_test_metadata(args.test_file)
        return 0
    if args.command == "install-deps":
        install_test_deps(args.test_file, args.runner, args.uv_python, args.install_project)
        return 0
    if args.command == "run":
        return run_test(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
