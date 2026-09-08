# Rank-8 GSM8K-Platinum native run — 2026-09-08

Status: preparing; final latest-runtime parity is pending. No performance or
accuracy result is claimed until the complete evaluation finishes.

Validated deployment snapshot selected for this run:

`/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-zml-cuda-v3__qvq-p32__yaqa125x__seed7__20260908`

Original spectral source snapshot (distinct artifact, not overwritten):

`/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-spectral__qvq-p32__yaqa125x__seed7__20260908`

The deployment copy retains rank-8 factors on 80 quantized modules, with dense
o/up projections required by this CUDA deployment. Its results must be labeled
as the CUDA-converted snapshot, not assumed identical to the original artifact.

## Workload and runtime

- Dataset: `madrylab/gsm8k-platinum`, config `main`, split `test`.
- Verified 1,209 rows; `max_rows=None`, expected-row check 1,209.
- Dataset cache revision: `e762492455a1cf7967de89f05b6bef72fc713b66`;
  datasets fingerprint: `b4f3f45576df4897`.
- Evalution CoT variant with chat template; greedy generation, 256 output-token
  cap; batch 8, prefill capacity 128, context 2048. No implicit truncation.
  All 1,209 chat-rendered prompts were preflighted: 796–959 tokens, 1,006,268
  prompt tokens total. The longest prompt plus generation fits context 2048.
- Public ZML C ABI, continuous slot refill, Triton paged attention, command
  buffers enabled. No HTTP/OpenAI transport, LLMD or GPT-QModel-Ultra.
- ZML merged revision: `db2a07b52225471c7c6920285aebd5b7d61d5d49`.
- QVQ dependency: `5b6fe5645b74c62089c4a64206e9c0c99e5a2ff5`.
- CUDA QVQ library SHA256:
  `abfc2aa259d948a5414f532270aef71198136d0a7afd9e6351eaae009d1a9bbe`.
- GPU: NVIDIA PG506-230, SM80, UUID
  `GPU-737e2423-874a-23a4-1126-dfbe3e77c294`, driver 610.43.02.

## Timing interpretation

The benchmark warms up prefill/decode before collecting per-step timings.
Times surround synchronous native ABI calls and include argument marshalling,
transfers, GPU execution and output readiness; they are not CUDA-kernel-only
timings. Tokenization and scheduler work are outside these phase timings and
remain in the evaluation wall time. Padding rows do not count as useful tokens.
Prefill phases may simultaneously advance existing decode lanes; the report
records prompt and decode counts separately and does not attribute all mixed
phase time solely to one type of work. Pure decode throughput is aggregate
useful decode tokens per second, not per-request speed.

Expected outputs:

- `/tmp/qvq_gsm8k_platinum_native_full_rank8.log`
- `/monster/data/model/qvq/gsm8k_platinum_native_full_rank8_20260908.json`

These paths are reserved for the forthcoming run, not evidence of completion.
