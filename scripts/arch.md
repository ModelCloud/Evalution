# CI Architecture

## Naming

- Workflow entrypoints follow the same `ci_*.py` convention as GPTQModel.
- `ci_common.py` and `ci_gpu.py` are kept structurally aligned with GPTQModel so allocator and GitHub env handling stay consistent across repos.

## Evalution unit test flow

1. `list-test-files`
- `ci_workflow.py set-matrix-config` writes the matrix parallelism output.
- `ci_workflow.py list-tests` enumerates `tests/test_*.py`, applies the optional regex filter, emits the test matrix, and reports whether any `unit_test_common` cases are present.

2. `prepare-common-env`
- When any common tests are scheduled, the workflow activates the branch-scoped `unit_test_common` uv env, runs `ci_workflow.py setup-uv-env`, and runs `ci_workflow.py prepare-test-run` once before the test matrix starts.

3. `test` job setup
- `ci_workflow.py set-test-metadata` derives `SAFE_NAME`, `ENV_FAMILY`, and whether the test requires a GPU by reading the file marker.
- `ci_workflow.py activate-uv-env` resolves the test runtime signature, writes `PYTHON_VERSION`, `UV_PYTHON`, `ENV_NAME`, and `UV_CACHE_DIR`, then activates the shared uv env for that signature.
- `ci_workflow.py setup-uv-env` initializes compiler and torch state, then installs the Python-version-specific runtime dependencies once per shared env under a filesystem lock.
- `ci_workflow.py print-uv-env` prints the same diagnostic state the old shell step emitted.

4. execution
- `ci_gpu.py allocate` reserves a GPU only when the test requires one.
- Common-env matrix jobs skip `ci_workflow.py setup-uv-env` and `ci_workflow.py prepare-test-run` because `prepare-common-env` has already installed the shared runtime and project package.
- `unit_test_tensorrt_llm` keeps the original flow and prepares its env lazily when that test runs.
- `ci_tests.py run` executes pytest, writes artifacts, and optionally keeps the GPU lease alive while the test is running.
- `ci_workflow.py release-gpu-if-present` releases the GPU only when allocation metadata exists.

## Maintenance rule

- Keep Evalution and GPTQModel aligned at the script boundary: shared concerns should stay in similarly named `ci_*` entrypoints even if the repo-specific workflow steps differ.
