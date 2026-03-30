from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import evalution
from evalution.engines.memory import resolve_dtype
from evalution.engines.transformers import _truncate_at_stop
from evalution.suites.gsm8k_common import (
    FLEXIBLE_EXTRACT_PATTERN,
    exact_match,
    extract_match,
)
from evalution.suites.gsm8k_platinum import GSM8KPlatinum


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    name: str
    mode: str
    attn_implementation: str


BENCHMARK_CONFIGS = (
    BenchmarkConfig(name="static_eager", mode="static", attn_implementation="eager"),
    BenchmarkConfig(name="static_sdpa", mode="static", attn_implementation="sdpa"),
    BenchmarkConfig(name="static_flash_attention_2", mode="static", attn_implementation="flash_attention_2"),
    BenchmarkConfig(name="continuous_paged_sdpa", mode="continuous", attn_implementation="sdpa"),
    BenchmarkConfig(
        name="continuous_paged_flash_attention_2",
        mode="continuous",
        attn_implementation="flash_attention_2",
    ),
)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_device(requested_device: str) -> str:
    if not torch.cuda.is_available():
        return "cpu"

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_count = torch.cuda.device_count()
    if device_count != 1:
        raise RuntimeError(
            "benchmark must run with exactly one visible CUDA device; "
            f"got torch.cuda.device_count()={device_count}, CUDA_VISIBLE_DEVICES={visible_devices!r}"
        )
    if requested_device.startswith("cuda"):
        return "cuda:0"
    return requested_device


def describe_runtime(device: str) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resolved_device": device,
        "torch_cuda_is_available": torch.cuda.is_available(),
        "visible_cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available() and device.startswith("cuda"):
        gpu_index = torch.device(device).index or 0
        props = torch.cuda.get_device_properties(gpu_index)
        runtime["gpu_name"] = props.name
        runtime["gpu_total_vram_gib"] = round(props.total_memory / (1024**3), 2)
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/monster/data/model/Llama-3.2-1B-Instruct",
    )
    parser.add_argument("--variant", default="cot")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-path", default="madrylab/gsm8k-platinum")
    parser.add_argument("--dataset-name", default="main")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--static-batch-size", default="auto")
    parser.add_argument("--configs", nargs="*", default=[config.name for config in BENCHMARK_CONFIGS])
    args = parser.parse_args()
    device = resolve_device(args.device)

    suite = evalution.gsm8k_platinum(
        variant=args.variant,
        apply_chat_template=True,
        max_new_tokens=args.max_new_tokens,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    spec = GSM8KPlatinum.VARIANTS[args.variant]

    dataset_load_started = perf_counter()
    all_docs = list(load_dataset(args.dataset_path, args.dataset_name, split=args.split))
    dataset_load_s = perf_counter() - dataset_load_started
    if args.max_rows is not None:
        all_docs = all_docs[: args.max_rows]

    prompt_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if prompt_tokenizer.pad_token_id is None:
        if prompt_tokenizer.eos_token is not None:
            prompt_tokenizer.pad_token = prompt_tokenizer.eos_token
        elif prompt_tokenizer.unk_token is not None:
            prompt_tokenizer.pad_token = prompt_tokenizer.unk_token
        else:
            raise ValueError("tokenizer must define either a pad_token, eos_token, or unk_token")

    prep_started = perf_counter()
    requests = []
    targets: list[str] = []
    rendered_prompts: list[str] = []
    input_ids: list[list[int]] = []
    for index, doc in enumerate(all_docs):
        fewshots = suite._select_fewshots(spec=spec, docs=all_docs, doc=doc, index=index)
        request = suite._build_request(
            spec=spec,
            doc=doc,
            fewshots=fewshots,
            fewshot_as_multiturn=True,
        )
        if request.messages is None:
            raise ValueError("expected chat-template prompts for benchmark")
        rendered_prompt = prompt_tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=request.add_generation_prompt,
        )
        prompt_ids = prompt_tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
        requests.append(request)
        targets.append(spec.target_builder(doc))
        rendered_prompts.append(rendered_prompt)
        input_ids.append(prompt_ids)
    prompt_prepare_s = perf_counter() - prep_started
    prompt_lengths = [len(prompt) for prompt in input_ids]

    selected_configs = [config for config in BENCHMARK_CONFIGS if config.name in set(args.configs)]
    results = []
    for config in selected_configs:
        print(
            f"[benchmark] starting name={config.name} mode={config.mode} attn={config.attn_implementation}",
            file=sys.stderr,
            flush=True,
        )
        if config.mode == "static":
            result = benchmark_static(
                config=config,
                model_path=args.model_path,
                dtype=args.dtype,
                device=device,
                requests=requests,
                rendered_prompts=rendered_prompts,
                targets=targets,
                spec=spec,
                batch_size=args.static_batch_size,
            )
        elif config.mode == "continuous":
            result = benchmark_continuous(
                config=config,
                model_path=args.model_path,
                dtype=args.dtype,
                device=device,
                input_ids=input_ids,
                targets=targets,
                spec=spec,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            raise ValueError(f"unsupported mode: {config.mode}")
        results.append(result)
        print(
            "[benchmark] completed "
            f"name={config.name} total_eval_s={result['total_eval_s']} "
            f"samples_per_s={result['samples_per_s']} strict={result['exact_match_strict']} "
            f"flex={result['exact_match_flexible']} invalid={result['invalid_ratio']}",
            file=sys.stderr,
            flush=True,
        )

    print(
        json.dumps(
            {
                "model_path": args.model_path,
                "variant": args.variant,
                "dataset_path": args.dataset_path,
                "dataset_name": args.dataset_name,
                "split": args.split,
                "runtime": describe_runtime(device),
                "row_count": len(all_docs),
                "max_rows": args.max_rows,
                "dataset_load_s": round(dataset_load_s, 3),
                "prompt_prepare_s": round(prompt_prepare_s, 3),
                "prompt_tokens": {
                    "min": min(prompt_lengths),
                    "avg": round(sum(prompt_lengths) / len(prompt_lengths), 1),
                    "max": max(prompt_lengths),
                },
                "results": results,
            },
            indent=2,
        )
    )


def benchmark_static(
    *,
    config: BenchmarkConfig,
    model_path: str,
    dtype: str,
    device: str,
    requests: list[Any],
    rendered_prompts: list[str],
    targets: list[str],
    spec: Any,
    batch_size: str,
) -> dict[str, Any]:
    engine = evalution.Transformers(
        dtype=dtype,
        attn_implementation=config.attn_implementation,
        device=device,
        batch_size=batch_size,
        paged_attention=False,
    )

    model_load_started = perf_counter()
    with redirect_stdout(sys.stderr):
        session = engine.build(evalution.Model(path=model_path))
    sync()
    model_load_s = perf_counter() - model_load_started
    with redirect_stdout(sys.stderr):
        effective_batch_size = session.resolve_batch_size(requests)

    generation_s = 0.0
    scoring_s = 0.0
    predictions: list[str] = []
    generated_token_lengths: list[int] = []
    scores = {
        "exact_match,strict-match": 0.0,
        "exact_match,flexible-extract": 0.0,
    }
    invalid_predictions = 0
    try:
        for start in range(0, len(requests), effective_batch_size):
            batch_requests = requests[start : start + effective_batch_size]
            generation_started = perf_counter()
            with redirect_stdout(sys.stderr):
                outputs = session.generate(batch_requests, batch_size=len(batch_requests))
            sync()
            generation_s += perf_counter() - generation_started

            scoring_started = perf_counter()
            for batch_offset, output in enumerate(outputs):
                index = start + batch_offset
                prediction = output.text
                strict_prediction = extract_match(
                    prediction,
                    spec.strict_regex,
                    group_select=spec.strict_group_select,
                )
                flexible_prediction = extract_match(
                    prediction,
                    FLEXIBLE_EXTRACT_PATTERN,
                    group_select=-1,
                )
                scores["exact_match,strict-match"] += float(exact_match(strict_prediction, targets[index]))
                scores["exact_match,flexible-extract"] += float(exact_match(flexible_prediction, targets[index]))
                if flexible_prediction == "[invalid]":
                    invalid_predictions += 1
                predictions.append(prediction)
                generated_token_lengths.append(
                    len(session.tokenizer.encode(prediction, add_special_tokens=False))
                )
            scoring_s += perf_counter() - scoring_started
    finally:
        session.close()

    return summarize_result(
        config=config,
        row_count=len(requests),
        model_load_s=model_load_s,
        generation_s=generation_s,
        scoring_s=scoring_s,
        scores=scores,
        invalid_predictions=invalid_predictions,
        generated_token_lengths=generated_token_lengths,
        effective_batch_size=effective_batch_size,
    )


def benchmark_continuous(
    *,
    config: BenchmarkConfig,
    model_path: str,
    dtype: str,
    device: str,
    input_ids: list[list[int]],
    targets: list[str],
    spec: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError("tokenizer must define either a pad_token, eos_token, or unk_token")

    model_load_started = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=resolve_dtype(dtype),
        attn_implementation=config.attn_implementation,
    ).to(device).eval()
    sync()
    model_load_s = perf_counter() - model_load_started

    generation_config = GenerationConfig.from_model_config(model.config)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.do_sample = False
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.stop_strings = list(spec.stop_strings)
    if tokenizer.eos_token_id is not None:
        generation_config.eos_token_id = tokenizer.eos_token_id

    generation_started = perf_counter()
    outputs = model.generate_batch(
        input_ids,
        generation_config=generation_config,
        progress_bar=False,
    )
    sync()
    generation_s = perf_counter() - generation_started

    scoring_started = perf_counter()
    predictions: list[str] = []
    generated_token_lengths: list[int] = []
    scores = {
        "exact_match,strict-match": 0.0,
        "exact_match,flexible-extract": 0.0,
    }
    invalid_predictions = 0
    for index in range(len(input_ids)):
        output = outputs[f"req_{index}"]
        prediction = tokenizer.decode(output.generated_tokens, skip_special_tokens=False)
        prediction = _truncate_at_stop(prediction, list(spec.stop_strings)).strip()
        strict_prediction = extract_match(
            prediction,
            spec.strict_regex,
            group_select=spec.strict_group_select,
        )
        flexible_prediction = extract_match(
            prediction,
            FLEXIBLE_EXTRACT_PATTERN,
            group_select=-1,
        )
        scores["exact_match,strict-match"] += float(exact_match(strict_prediction, targets[index]))
        scores["exact_match,flexible-extract"] += float(exact_match(flexible_prediction, targets[index]))
        if flexible_prediction == "[invalid]":
            invalid_predictions += 1
        predictions.append(prediction)
        generated_token_lengths.append(len(output.generated_tokens))
    scoring_s = perf_counter() - scoring_started

    del model
    del tokenizer
    sync()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summarize_result(
        config=config,
        row_count=len(input_ids),
        model_load_s=model_load_s,
        generation_s=generation_s,
        scoring_s=scoring_s,
        scores=scores,
        invalid_predictions=invalid_predictions,
        generated_token_lengths=generated_token_lengths,
        effective_batch_size=None,
    )


def summarize_result(
    *,
    config: BenchmarkConfig,
    row_count: int,
    model_load_s: float,
    generation_s: float,
    scoring_s: float,
    scores: dict[str, float],
    invalid_predictions: int,
    generated_token_lengths: list[int],
    effective_batch_size: int | None,
) -> dict[str, Any]:
    return {
        "name": config.name,
        "mode": config.mode,
        "attn_implementation": config.attn_implementation,
        "effective_batch_size": effective_batch_size,
        "model_load_s": round(model_load_s, 3),
        "generation_s": round(generation_s, 3),
        "scoring_s": round(scoring_s, 3),
        "total_eval_s": round(generation_s + scoring_s, 3),
        "samples_per_s": round(row_count / generation_s, 3),
        "exact_match_strict": round(scores["exact_match,strict-match"] / row_count, 6),
        "exact_match_flexible": round(scores["exact_match,flexible-extract"] / row_count, 6),
        "invalid_ratio": round(invalid_predictions / row_count, 6),
        "generated_tokens": {
            "min": min(generated_token_lengths),
            "avg": round(sum(generated_token_lengths) / len(generated_token_lengths), 1),
            "max": max(generated_token_lengths),
        },
    }


if __name__ == "__main__":
    main()
