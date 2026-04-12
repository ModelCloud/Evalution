import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def append_github_env(name: str, value: str) -> None:
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        return
    with open(github_env, "a", encoding="utf-8") as fh:
        fh.write(f"{name}={value}\n")


def fetch_text(url: str, *, timeout: float, suppress_error: bool = False) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if suppress_error:
            print(f"Request failed for {url}: {exc}")
            return ""
        raise


def kill_process_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def start_keepalive_monitor(
    *,
    proc: subprocess.Popen[str],
    keep_alive_url: str,
    interval_sec: int,
) -> tuple[threading.Thread, threading.Event, dict[str, int]]:
    stop_event = threading.Event()
    state = {"forced_exit_code": 0}

    def worker() -> None:
        print(f"start to keep alive... {keep_alive_url}")
        while not stop_event.wait(interval_sec):
            resp = fetch_text(keep_alive_url, timeout=10, suppress_error=True)
            if int(resp.strip()) < 0:
                print(f"\n\n\nServer returned {resp.strip()}, terminating job...\n\n\n")
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


def to_safe_name(test_file: str) -> str:
    return test_file.replace("/", "__").replace(".", "_")


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
    safe_name = to_safe_name(args.test_file)
    log_file = artifacts_dir / f"{safe_name}.log"
    junitxml = artifacts_dir / f"{safe_name}.xml"

    pytest_cmd = ["pytest", "--durations=0", args.test_file, f"--junitxml={junitxml}"]
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

    encoded_test = urllib.parse.quote(args.test_file, safe="")
    encoded_runner = urllib.parse.quote(args.runner, safe="")
    keep_alive_url = (
        f"{args.base_url}/gpu/keepalive?runid={args.run_id}&test={encoded_test}"
        f"&runner={encoded_runner}&timestamp={int(time.time())}&gpu={env.get('CUDA_VISIBLE_DEVICES', '')}"
    )

    monitor_thread = None
    monitor_stop = None
    monitor_state = {"forced_exit_code": 0}
    if env.get("CUDA_VISIBLE_DEVICES", ""):
        monitor_thread, monitor_stop, monitor_state = start_keepalive_monitor(
            proc=proc,
            keep_alive_url=keep_alive_url,
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
        append_github_env("ERROR", "22")
        return 22

    if return_code != 0:
        append_github_env("ERROR", "22")
        print(f"pipe status wrong: {return_code}")
        return 22

    execution_time = int(time.time() - start_time)
    print(f"{execution_time // 60}m {execution_time % 60}s")

    try:
        for entry in sorted(artifacts_dir.iterdir()):
            stat = entry.stat()
            print(f"{stat.st_size:>10} {entry.name}")
    except OSError as exc:
        print(f"Failed to list artifact dir: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
