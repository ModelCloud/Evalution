import argparse
import subprocess
import sys
import time

from common import (
    append_github_env,
    build_get_request,
    build_server_info,
    extract_gpu_ids,
    format_info_url,
    normalize_base_url,
    request_json,
    request_json_with_retry,
)


def is_valid_gpu_response(value: str) -> bool:
    if not value:
        return False
    for part in value.split(","):
        if not part:
            return False
        if part.startswith("-"):
            if not part[1:].isdigit():
                return False
        elif not part.isdigit():
            return False
    return True


def print_status(base_url: str, runner_name: str) -> None:
    server = build_server_info(runner_name)
    try:
        status = request_json(
            format_info_url(base_url, server["platform"]),
            timeout=10,
        )
    except Exception as exc:
        print(f"Request failed for allocator info: {exc}")
        return
    if status is not None:
        print(status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--count", required=True)
    parser.add_argument("--sleep-sec", type=float, default=5)
    parser.add_argument("--timeout-sec", type=int, default=18000)
    parser.add_argument("--request-timeout", type=float, default=10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--require-single", action="store_true")
    args = parser.parse_args()

    start_s = time.time()
    endpoint = f"{normalize_base_url(args.base_url)}/get"

    print("Requesting GPU from allocator")
    print(f"run_id={args.run_id} test={args.test} runner={args.runner} count={args.count}")

    while True:
        request_body = build_get_request(
            runner_name=args.runner,
            run_id=args.run_id,
            test_name=args.test,
            count=args.count,
        )
        print(f"requesting GPU with: {endpoint}")

        response = request_json_with_retry(
            endpoint,
            method="POST",
            body=request_body,
            timeout=args.request_timeout,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
        resp = extract_gpu_ids(response)

        print(f"resp={{{resp}}}")

        if not is_valid_gpu_response(resp):
            print(f"Allocator returned invalid response: {resp!r} (temporary error)")
            print_status(args.base_url, args.runner)
            time.sleep(args.sleep_sec)
            continue

        if resp.startswith("-") and "," not in resp:
            elapsed = int(time.time() - start_s)
            if elapsed >= args.timeout_sec:
                print(
                    f"Timed out after {args.timeout_sec}s waiting for GPU "
                    f"(last response={resp})"
                )
                print_status(args.base_url, args.runner)
                return 1

            print(
                f"No GPU available (response={resp}). Waiting {args.sleep_sec}s..."
                f" elapsed={elapsed}s"
            )
            print_status(args.base_url, args.runner)
            time.sleep(args.sleep_sec)
            continue

        if args.require_single and "," in resp:
            print(f"Allocator returned multiple GPUs for job requiring one GPU: {resp}")
            return 1

        print(f"Allocated GPU ID: {resp}")
        append_github_env("CUDA_VISIBLE_DEVICES", resp)
        print(f"CUDA_VISIBLE_DEVICES set to {resp}")
        print(subprocess.getoutput(f"nvidia-smi -i {resp} --query-gpu=name --format=csv"))
        print_status(args.base_url, args.runner)
        return 0


if __name__ == "__main__":
    sys.exit(main())
