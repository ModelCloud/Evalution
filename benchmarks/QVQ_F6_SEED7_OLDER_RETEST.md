# Older F6 seed-7 retest — 2026-09-08

Requested original checkpoint (no rank8 tensors in its shard index):

`/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32`

Attempted the same complete GSM8K-Platinum native-ABI workload as the rank8 run:
1,209 expected rows, batch 8, prefill 128, context 2048, output cap 256, CoT/chat
template, command buffers enabled. No requantization or checkpoint edits.

Status: loader failed with `IncompleteQvqProjection`; zero benchmark rows ran.
Log: `/tmp/qvq_gsm8k_platinum_native_full_f6_older.log`.

The original contains legacy W4 projections with trellis/SU/SV but no bank IDs
(for example layer 0 self-attention o_proj). ZML's current Projection loader
accepts that layout only with ROCm compilation plans, not this CUDA runtime.
Missing bank metadata must not be invented to force P32 loading: the legacy
representation is a different format.

An existing deployment converter, `/root/materialize_zml_deploy.py`, could be
adapted to materialize legacy/mixed o/up projections as dense FP16 in a separate
copy, as was done for the rank8 CUDA deployment. This is not requantization,
but it changes the deployment representation and needs conversion validation.
No such baseline copy has been created, and no older-snapshot score or speed
is claimed from this failed attempt. The original snapshot remains unchanged.
