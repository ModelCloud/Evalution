"""Full-row generation benchmarks over the public ZML/QVQ C ABI (no HTTP).

Suite names are explicit: GSM8K-Platinum is not GSM8K-Pro, and canonical MMLU
humanities requires likelihood support that this native ABI does not yet expose.
"""

import argparse
import json
import time
from pathlib import Path

import evalution
from evalution.engines.base import GenerationRequest


class TimedZMLNative(evalution.ZMLNative):
    def build(self, model):
        started = time.perf_counter()
        session = super().build(model)
        self.load_compile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        try:
            # Prepare both phase graphs before benchmark timing. Not dataset rows.
            session.generate(
                [
                    GenerationRequest(prompt="Count carefully. " * 64, max_new_tokens=8)
                    for _ in range(self.batch_size)
                ]
            )
        except Exception:
            session.close()
            raise
        self.warmup_seconds = time.perf_counter() - started
        session.step_metrics.clear()
        self.measured_session = session
        return session


def summarize_steps(steps):
    result = {}
    for phase in ("prefill", "decode"):
        selected = [s for s in steps if s["phase"] == phase]
        seconds = sum(s["seconds"] for s in selected)
        prompt = sum(s["prompt_tokens"] for s in selected)
        decode = sum(s["decode_tokens"] for s in selected)
        result[phase] = {
            "calls": len(selected),
            "seconds": seconds,
            "prompt_tokens": prompt,
            "decode_tokens": decode,
            "active_tokens_per_second": (prompt + decode) / seconds
            if seconds
            else None,
            "prompt_tokens_per_second": prompt / seconds if seconds else None,
            "padding_rows": sum(s["padding_rows"] for s in selected),
        }
    return result


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
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
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
    engine = TimedZMLNative(
        library=args.library,
        batch_size=args.batch_size,
        token_batch_size=args.token_batch_size,
        max_context_len=args.max_context_len,
        collect_timing=True,
    )
    started = time.perf_counter()
    with engine.model(path=args.model_path) as run:
        result = run.run(suite).result()
    total_seconds = time.perf_counter() - started
    report = result.to_dict()
    row_count = sum(len(test.samples) for test in result.tests)
    report["full_row_check"] = {
        "evaluated_rows": row_count,
        "expected_rows": args.expected_rows,
        "max_rows": None,
    }
    report["native_timing"] = {
        "load_compile_seconds": engine.load_compile_seconds,
        "warmup_seconds": engine.warmup_seconds,
        "total_seconds": total_seconds,
        "evaluation_and_close_seconds": total_seconds
        - engine.load_compile_seconds
        - engine.warmup_seconds,
        "phases": summarize_steps(engine.measured_session.step_metrics),
        "scope": "Synchronous native ABI wall time; excludes scheduler/tokenization. Prefill phase may include concurrent decode rows; padding excluded from token counts. Warmup excluded.",
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": row_count,
                "metrics": [test.metrics for test in result.tests],
                "native_timing": report["native_timing"],
            },
            indent=2,
        )
    )
    if args.expected_rows is not None and row_count != args.expected_rows:
        raise RuntimeError(
            f"Expected {args.expected_rows} full rows, received {row_count}"
        )


if __name__ == "__main__":
    main()
