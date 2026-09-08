# QVQ rank-8 native evaluation status (2026-09-08)

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
The subsequent QVQ/ZML speedup integration is rebuilding and must be revalidated.
Full-row
benchmarks have not resumed; earlier 128-row smoke results are not full results.

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
`dde7319c65c30dc837ab9a7539d6da08a50379f5` (`origin/master`),
Evalution `8cbed6e` (`origin/main`). Native feature PR preparation excludes the
earlier HTTP transport and unvalidated legacy W4 fallback.

Open implementation PRs: https://github.com/ModelCloud/ZML-Ultra/pull/52 and
https://github.com/ModelCloud/Evalution/pull/144. The latest ZML integration also
needs an explicit `Tensor.Pad` type in grouped rank8 packing to compile with the
repository's Zig version; the fix is included in the native feature branch.

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
MMLU accuracy. GSM8K-Platinum is also not silently equivalent to GSM8K-Pro.
