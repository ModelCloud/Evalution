import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path
from common import (
    append_github_env,
    build_job_request,
    extract_gpu_ids,
    normalize_base_url,
    normalize_test_file,
    request_json,
    to_safe_name,
)


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
                print(f"\n\nKeepalive request failed: {exc}\n\n")
                continue

            resp = extract_gpu_ids(response)
            if resp == "-1":
                print(f"\n\nServer returned {resp}, terminating job...\n\n")
                state["forced_exit_code"] = 3
                kill_process_group(proc)
                stop_event.set()
                return
            if expected_gpu_ids and resp != expected_gpu_ids:
                print(f"\n\nKeepalive returned mismatched GPUs {resp}, expected {expected_gpu_ids}.\n\n")
                state["forced_exit_code"] = 3
                kill_process_group(proc)
                stop_event.set()
                return
            print("\n\ngpu is kept alive...\n\n")

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--gpu-id", default="")
    parser.add_argument("--monitor-interval-sec", type=int, default=60)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--clear-cuda", action="store_true")
    args = parser.parse_args()

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
            print("\n\ntrap cleanup EXIT...\n\n")
            monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=5)

    if monitor_state["forced_exit_code"]:
        append_github_env("ERROR", "22")
        return 22

    if return_code != 0:
        append_github_env("ERROR", "22")
        print(f"\n\npipe status wrong: {return_code}\n\n")
        return 22

    execution_time = int(time.time() - start_time)
    print(f"\n\n{execution_time // 60}m {execution_time % 60}s\n\n")

    try:
        for entry in sorted(artifacts_dir.iterdir()):
            stat = entry.stat()
            print(f"\n\n{stat.st_size:>10} {entry.name}\n\n")
    except OSError as exc:
        print(f"\n\nFailed to list artifact dir: {exc}\n\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
