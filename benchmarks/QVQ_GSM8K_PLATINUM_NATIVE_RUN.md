# Rank-8 GSM8K-Platinum native run — 2026-09-08

IMPORTANT: subsequent tensor audit found that this artifact's spectral base
differs from the older F6/seed7 base in 268 tensors, including all 112 projection
trellises. This result does not isolate the effect of rank8 on the older base.
See `QVQ_RANK8_BASE_AUDIT.md`.

Status: complete. All 1,209 unique test rows were evaluated successfully;
533 correct, numeric accuracy **44.0860%**. Latest-runtime parity also passed
(non-paged reference and three batched repetitions, command buffers enabled).

## Results

| Measurement | Result |
| --- | ---: |
| Mixed prefill phase, useful prompt throughput | 4,630.80 prompt tokens/s |
| Mixed prefill phase, all useful token throughput | 4,882.71 tokens/s |
| Decode-only phase, aggregate throughput | 537.50 tokens/s |
| Prefill-phase wall time | 217.299 s |
| Decode-only phase wall time | 101.852 s |
| Evalution generation wall time | 326.553 s |
| Evaluation and close wall time | 332.321 s |
| Model load and compilation | 33.285 s |
| Warmup | 1.567 s |
| Total run wall time | 367.173 s |

Prefill: 8,516 calls, 1,006,268 prompt tokens and 54,741 concurrent decode
tokens; 29,039 padding rows excluded. Decode-only: 6,973 calls, 54,746 useful
decode tokens; 1,038 padding rows excluded. Mixed prefill timing cannot be
interpreted as an isolated prefill kernel benchmark. The reported decode rate
is aggregated over active requests, not single-request tokens/s.

Artifact checks passed: 1,209 samples, unique indices exactly 0–1208, all
1,006,268 independently preflighted prompt tokens accounted for, no row limit.
The result is 748,951 bytes; SHA256:
`58eded6fdf423b07810e0aadc2df44b6185f5479770745bd1dd79dc0fb0dfe46`.

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
- Evalution runner revision: `8c027726a8567eab3e1a3537e996ee9615a2c33c`
  (native adapter PR #144 atop latest `origin/main`).
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

The result JSON and complete log are present. Run exit code: 0.
