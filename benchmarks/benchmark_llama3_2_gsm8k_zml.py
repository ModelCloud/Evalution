"""Run GSM8K-Platinum against a running ZML/LLMD server."""

from __future__ import annotations

import argparse
import json
import os

import evalution


def main() -> None:
    """Run the reproducible Llama 3.2 1B ZML integration benchmark."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/monster/data/model/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ZML_BASE_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument("--model-name", default="Llama-3.2-1B-Instruct")
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-parallel-requests", type=int, default=64)
    parser.add_argument("--launch-server", action="store_true")
    parser.add_argument("--llmd-executable", default="llmd")
    parser.add_argument("--dflash-model")
    parser.add_argument("--server-arg", action="append", default=[])
    args = parser.parse_args()

    result = (
        evalution.ZML(
            base_url=args.base_url,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_parallel_requests=args.max_parallel_requests,
            launch_server=args.launch_server,
            executable=args.llmd_executable,
            dflash_model=args.dflash_model,
            server_args=args.server_arg,
        )
        .model(path=args.model_path)
        .run(
            evalution.benchmarks.gsm8k_platinum(
                variant="cot",
                apply_chat_template=True,
                stream=True,
                max_rows=args.max_rows,
                max_new_tokens=args.max_new_tokens,
            )
        )
        .result()
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
