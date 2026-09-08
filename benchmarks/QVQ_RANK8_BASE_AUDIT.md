# Rank8 base audit — 2026-09-08

The requested controlled experiment was the older F6/seed7 base plus rank8
factors. The previously tested rank8 CUDA deployment is NOT that controlled
experiment. Its 44.0860% GSM8K-Platinum score remains a result for that specific
artifact, but must not be attributed solely to adding rank8 to the older base.

Older source:
`/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__20260904__commit5c5979194dc0__aff65a505e88/qvq-p32`

September 8 spectral base:
`/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-spectral__qvq-p32__yaqa125x__seed7__20260908`

Findings from configs and SHA256 hashes of each tensor's stored payload:

- Same 558 tensor keys, but 268 tensors differ: 112 trellises, 94 bank-ID
  tensors and 62 alternate-bank IDs. All 112 projection trellises differ.
- Dynamic bit-width/format mappings match, including legacy non-P32 W4 on
  all 16 o_proj and up_proj in layers 6 and 8.
- Older `yaqa.spectral_refinement=false`; spectral base sets it `true`.
- `yaqa.spectral_ranks` changes from `[8,16,32]` to `[8]`.
- Quantizer revision and timestamp metadata differ.
- The spectral base itself has zero serialized rank8 tensors in its shard
  index; it is a newly quantized/refined base, not a serialized low-rank adapter.
- Rank8 factors were added in a subsequent artifact. The tested CUDA-v3 copy
  has 240 rank8-related tensor keys and 32 dense o/up projections, with accepted
  P32 up_proj correction baked into dense weights according to its report.

Neither original checkpoint was modified during this audit. Native W4 loading
is being implemented to run the unchanged older source directly. Building a
true older-base-plus-rank8 artifact is separate work; do not silently replace
the base weights or reuse factors fitted against the different spectral base.
