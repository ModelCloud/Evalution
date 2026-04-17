import argparse
import sys
import urllib.error

from common import build_job_request, extract_gpu_ids, normalize_base_url, request_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-id", default="")
    parser.add_argument("--timestamp")
    parser.add_argument("--test", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()

    request_body = build_job_request(
        runner_name=args.runner,
        run_id=args.run_id,
        test_name=args.test,
    )
    url = f"{normalize_base_url(args.base_url)}/release"
    print(url)

    try:
        response = request_json(url, method="POST", body=request_body, timeout=args.timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"Failed to release GPU: {exc}")
        return 0

    resp = extract_gpu_ids(response)
    print(f"response: {resp}")
    if args.gpu_id and resp not in {args.gpu_id, "-1"}:
        print(f"Error: response ({resp}) != expected ({args.gpu_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
