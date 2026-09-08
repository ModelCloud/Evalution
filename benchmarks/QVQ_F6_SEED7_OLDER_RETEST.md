# Older F6 seed-7 retest — 2026-09-08

Requested original checkpoint (no rank8 tensors in its shard index):

`/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32`

Attempted the same complete GSM8K-Platinum native-ABI workload as the rank8 run:
1,209 expected rows, batch 8, prefill 128, context 2048, output cap 256, CoT/chat
template, command buffers enabled. No requantization or checkpoint edits.

Initial attempt: loader failed with `IncompleteQvqProjection`; zero benchmark rows ran.
Log: `/tmp/qvq_gsm8k_platinum_native_full_f6_older.log`.

The original contains legacy W4 projections with trellis/SU/SV but no bank IDs
(for example layer 0 self-attention o_proj). ZML's current Projection loader
previously accepted that layout only with ROCm compilation plans, not CUDA.
Missing bank metadata must not be invented to force P32 loading: the legacy
representation is a different format.

## Native W4 fix and rerun

The user explicitly requested native non-P32 W4 support instead of a checkpoint
conversion. ZML PR https://github.com/ModelCloud/ZML-Ultra/pull/54 implements it.
Runtime revision: `7e31e0291d67b626470d8dc80bb5bc8c4883bedd`, based on `db2a07b`.
QVQ P32 dependency remains `5b6fe5645`.

48 eager/replay decoder checks against QVQ's CPU reference passed, including
actual W4 checkpoint tensors. Worst observed max error 0.00021744; CUDA memcheck
reported zero errors. Original-model serial and three batch3 runs matched;
eager and command-buffer output token IDs also matched.

Full rerun completed successfully: all 1,209 unique rows scored. Evalution source revision
`9b87bd0`, native ABI, continuous refill, batch8/prefill128/context2048/output256,
8-shot CoT with chat template. GPU0 UUID
`GPU-737e2423-874a-23a4-1126-dfbe3e77c294`; three-sample idle preflight passed.
Tokenizer, tokenizer config, chat template and generation config are byte-identical
to the preceding run. No rank8 tensors, no dense checkpoint conversion, no
requantization. `yaqa.spectral_refinement=false` in the original configuration.

Log: `/tmp/qvq_gsm8k_platinum_native_full_f6_older_w4fixed.log`

Result (751,190 bytes):
`/monster/data/model/qvq/gsm8k_platinum_native_full_f6_seed7_older_20260908.json`

SHA256: `aa5e9745f4c31d4a713022d1f0a0c13a07b31acd7bccfbb3395a367a456aeb36`.

## Full-run result

- Accuracy: **530/1,209 = 43.8379%**; indices exactly 0 through 1208, no row cap.
- Mixed prefill-phase prompt throughput: **4,018.64 prompt tokens/s**.
- Decode-only aggregate throughput: **513.83 tokens/s**, batch size 8.
- Load/compile: 38.063 s; excluded warmup: 1.794 s.
- Generation: 364.518 s; evaluation plus close: 370.535 s.
- Total wall time: **410.393 s (6m 50s)**.

Prefill: 8,527 calls, 250.400 s, 1,006,268 prompt tokens and 55,225
concurrent decode tokens (29,963 padding tokens excluded). Decode-only: 6,884
calls, 106.800 s, 54,877 tokens (195 padding tokens excluded). Phase times
measure synchronous native ABI calls, including transfers/readiness, not isolated
GPU kernels; Python scheduling and tokenization are outside those phase times.
Prefill includes concurrent decode work. Command buffers were enabled.

The native W4 implementation uses compiler-visible GPU tensor operations; this
result does not establish a fused or fully optimized W4 kernel. No checkpoint
conversion or requantization was performed.

The prior rank8 result used a DIFFERENT spectral-refined base, so differences
between these runs must not be attributed solely to rank8. See
`QVQ_RANK8_BASE_AUDIT.md`. All source checkpoint files remain unchanged.
