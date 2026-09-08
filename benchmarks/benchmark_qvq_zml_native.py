"""Full-row generation benchmarks over the public ZML/QVQ C ABI (no HTTP).

Suite names are explicit: GSM8K-Platinum is not GSM8K-Pro, and canonical MMLU
humanities requires likelihood support that this native ABI does not yet expose.
"""

import argparse
import json
from pathlib import Path

import evalution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--suite", choices=["gsm8k-platinum", "mmlu-pro"], required=True
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-batch-size", type=int, default=128)
    parser.add_argument("--max-context-len", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    options = dict(
        apply_chat_template=True,
        stream=True,
        max_rows=None,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    suite = (
        evalution.benchmarks.gsm8k_platinum(variant="cot", **options)
        if args.suite == "gsm8k-platinum"
        else evalution.benchmarks.mmlu_pro(**options)
    )
    result = (
        evalution.ZMLNative(
            library=args.library,
            batch_size=args.batch_size,
            token_batch_size=args.token_batch_size,
            max_context_len=args.max_context_len,
        )
        .model(path=args.model_path)
        .run(suite)
        .result()
    )
    serialized = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
