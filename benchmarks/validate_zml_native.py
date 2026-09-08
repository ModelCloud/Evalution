"""Opt-in GPU validation of native batching, chunked prefill and lane reuse.

Compares batched results against serial requests on the same native executable.
This is an isolation/parity test, not an independent model-quality benchmark or
proof of command-buffer replay. Use the original ZML runner for that additional
model parity check.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

from evalution import Model, ZMLNative
from evalution.engines.base import GenerationRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--reference-runner", help="Optional original non-paged ZML JSONL runner"
    )
    args = parser.parse_args()
    prompts = [
        "The capital of France is",
        "What is two plus two?",
        "Count carefully. " * 20,
        "The opposite of hot is",
        "Hello!",
    ]
    requests = [
        GenerationRequest(prompt=prompt, max_new_tokens=n)
        for prompt, n in zip(prompts, [2, 9, 6, 7, 4])
    ]
    reference = None
    if args.reference_runner:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        runner = str(Path(args.reference_runner).absolute())
        env = dict(os.environ, RUNFILES_DIR=runner + ".runfiles")
        payload = "".join(
            json.dumps(
                {
                    "input_ids": tokenizer.encode(r.prompt, add_special_tokens=False),
                    "max_new_tokens": r.max_new_tokens,
                }
            )
            + "\n"
            for r in requests
        )
        # Finish and unload the reference before the native session reserves
        # GPU memory. This transport is diagnostic only, never benchmark access.
        completed = subprocess.run(
            [runner, f"--model={args.model_path}", "--seqlen=256"],
            input=payload,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
            env=env,
        )
        responses = [json.loads(line) for line in completed.stdout.splitlines()]
        reference = [item["output_ids"] for item in responses if "output_ids" in item]
        assert len(reference) == len(requests), responses
    session = ZMLNative(
        library=args.library, batch_size=3, max_context_len=256, token_batch_size=32
    ).build(Model(path=args.model_path))
    try:
        serial = session.generate(requests, batch_size=1)
        if reference is not None:
            for i, (expected, actual) in enumerate(zip(reference, serial, strict=True)):
                assert actual.metadata["output_ids"] == expected, (
                    i,
                    expected,
                    actual.metadata,
                )
        for iteration in range(3):
            batched = session.generate(requests, batch_size=3)
            for index, (expected, actual) in enumerate(
                zip(serial, batched, strict=True)
            ):
                assert (
                    actual.metadata["output_ids"] == expected.metadata["output_ids"]
                ), (iteration, index, expected.metadata, actual.metadata)
        print(
            json.dumps(
                {
                    "batch_isolation_parity": "PASS",
                    "nonpaged_reference_parity": "PASS"
                    if reference is not None
                    else "NOT_RUN",
                    "iterations": 3,
                    "outputs": [o.metadata["output_ids"] for o in serial],
                    "execution": session.describe_execution(),
                }
            )
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
