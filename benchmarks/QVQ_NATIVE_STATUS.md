# QVQ rank-8 native evaluation status (2026-09-08)

PROVENANCE CORRECTION: the September 8 spectral base was not the unchanged
older F6/seed7 base. `spectral_refinement` was incorrectly enabled; 268 tensor
payloads differ, including all 112 projection trellises. The rank8 CUDA result
below is not a controlled rank8-only comparison. See `QVQ_RANK8_BASE_AUDIT.md`.
The user confirmed refinement should have remained disabled. A corrected
rank8 artifact must fit factors against the unchanged older base; changing
metadata on the newer artifact cannot undo its changed weights.

The unchanged older F6/seed7 checkpoint now runs natively with legacy non-P32 W4
support in ZML PR https://github.com/ModelCloud/ZML-Ultra/pull/54 (`7e31e029`).
With spectral refinement disabled and no rank8, the complete GSM8K-Platinum run
scored **530/1,209 (43.8379%)**. Mixed prefill prompt throughput was 4,018.64
tokens/s, decode-only aggregate throughput 513.83 tokens/s, total 410.393 s.
See `QVQ_F6_SEED7_OLDER_RETEST.md` for the original snapshot path, result path,
checksum, settings and timing scope. A corrected rank8-only artifact has not
yet been generated. Historical candidate details below are not a controlled
comparison against this baseline.

Requested snapshot:

`/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-spectral__qvq-p32__yaqa125x__seed7__20260908`

CUDA-converted candidate currently used for native runtime validation:

`/monster/data/model/qvq/modelcloud-qvq__llama-3.2-1b-instruct__f6-p32-r8-zml-cuda-v3__qvq-p32__yaqa125x__seed7__20260908`

These are distinct snapshots; validation of the CUDA candidate must not be
reported as a benchmark of the original snapshot without documenting conversion.

Implementation is in progress: public ZML/QVQ paged Llama runner, shared C ABI,
and Evalution `ZMLNative` continuous scheduler. No LLMD, GPT-QModel-Ultra, or
OpenAI HTTP transport is used by this new path. Scheduler unit tests cover
ragged/chunked prefill, refill, page isolation, EOS, stop strings, and slow input.
GPU parity and command-buffer replay validation passed on the CUDA candidate
on QVQ `470336257` after its merged-pin rebuild: final non-paged reference
parity and three batched repetitions passed (`/tmp/zml_native_merged198_parity.log`).
The subsequent QVQ `5b6fe5645` / ZML `db2a07b` integration also passed final
non-paged reference parity and three batched repetitions. The full 1,209-row
GSM8K-Platinum run completed through the native ABI, with phase timing:
533/1,209 correct (44.0860%), mixed-phase prompt throughput 4,630.80 tokens/s,
decode-only aggregate throughput 537.50 tokens/s, total wall time 367.173 s;
see `QVQ_GSM8K_PLATINUM_NATIVE_RUN.md`. Earlier 128-row smoke results are not full
results and must not be reused as scores from the corrected runtime.

Dependency issue found: importing Evalution loaded cuBLAS 13.1.1.3 from its Python
environment, while ZML bundles CUDA 13.3 with cuBLAS 13.5.1.27. Loading ZML cuBLAS then failed with
`undefined symbol: cublasLtZZZMatmulAlgoGetHeuristicForStream`. PJRT crashes while
formatting that failure. The CUDA libraries must be selected consistently before
Python imports. `LD_LIBRARY_PATH` alone did not work because PyTorch preloads
absolute package paths. Installed matching `nvidia-cublas==13.5.1.27` into
`/root/venv-py3.14t-gil0`; `pip check` passes.
Also installed `cuda-sanitizer-13-3` (13.3.75-1) for native decode diagnostics.

Matching cuBLAS resolved native startup with Evalution imported. Initial decode
reported `CUDA_ERROR_MISALIGNED_ADDRESS`.
Memcheck identified QVQ F6 K8192/N2048 dispatch passing a Boolean in the StaticK
template position, yielding stride one. The four M1..M4 launch sites are patched
and guarded in merged https://github.com/ModelCloud/QvQ/pull/198. Regression
tests passed (49 tests, including 48 CUDA cases with repeated graph replay).
Native CUDA memcheck subsequently reported zero errors. Serial, batched,
chunked-prefill and continuous-refill token outputs matched the corrected
non-paged reference; eager and command-buffer outputs matched. The trace
covered 18 prefill and 47 decode calls with stable input device addresses.
These checks cover the tested geometry/device, not every configuration.
Earlier serial scores using the faulty dispatch must also be rerun.

Latest fetched and integrated upstream bases: QVQ
`5b6fe5645b74c62089c4a64206e9c0c99e5a2ff5` (pinned by ZML), ZML
`db2a07b52225471c7c6920285aebd5b7d61d5d49` (`origin/master`),
Evalution `8cbed6e` (`origin/main`). Native feature PR preparation excludes the
earlier HTTP transport and unvalidated legacy W4 fallback.

ZML https://github.com/ModelCloud/ZML-Ultra/pull/52 is merged; Evalution
https://github.com/ModelCloud/Evalution/pull/144 remains open. The explicit
`Tensor.Pad` fix is included in merged ZML. Evalution additionally unwraps newer
Transformers chat-template `BatchEncoding` results before native generation;
17 focused tests pass, including that compatibility case and timing counts.

Native build targets in zml-ultra:

```sh
./bazel.sh build --config=release --jobs=16 --spawn_strategy=local \
  --@zml//platforms:cuda=true --@zml//platforms:cpu=false \
  //examples/llm:llama_native //examples/llm:llama_paged_token_runner
```

The direct library is `bazel-bin/examples/llm/libzml_llama.so`; the C contract is
`examples/llm/llama_native.h`. Runtime resources currently use the paged runner's
`.runfiles` tree via `RUNFILES_DIR`. This is a development setup, not yet a
standalone packaged distribution.

Scoring caveat: native ABI currently exposes greedy token generation only.
Canonical MMLU requires likelihood scoring, which is not implemented. The older
humanities script explicitly produces generated-choice accuracy, not canonical
MMLU accuracy. The user subsequently explicitly selected GSM8K-Platinum for the
current full-row run; it is not being labeled GSM8K-Pro.
