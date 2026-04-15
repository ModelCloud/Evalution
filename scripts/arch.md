# CI Architecture

## Naming

- Workflow entrypoints follow the same `ci_*.py` convention as GPTQModel.
- `ci_common.py` and `ci_gpu.py` are kept structurally aligned with GPTQModel so allocator and GitHub env handling stay consistent across repos.

## Evalution unit test flow

1. `list-test-files`
- `ci_workflow.py set-matrix-config` writes the matrix parallelism output.
- `ci_workflow.py list-tests` enumerates `tests/test_*.py`, applies the optional regex filter, and emits the test matrix.

2. `test` job setup
- `ci_workflow.py set-test-metadata` derives `SAFE_NAME` and whether the test requires a GPU by reading the file marker.
- `ci_workflow.py activate-uv-env` resolves the per-test Python version override, writes `PYTHON_VERSION` and `UV_PYTHON`, prints uv diagnostics, and activates the matching uv env.
- `ci_workflow.py setup-uv-env` initializes compiler and torch state, then installs the Python-version-specific runtime dependencies.
- `ci_workflow.py print-uv-env` prints the same diagnostic state the old shell step emitted.

3. execution
- `ci_gpu.py allocate` reserves a GPU only when the test requires one.
- `ci_workflow.py prepare-test-run` installs the project and runtime test dependencies.
- `ci_tests.py run` executes pytest, writes artifacts, and optionally keeps the GPU lease alive while the test is running.
- `ci_workflow.py release-gpu-if-present` releases the GPU only when allocation metadata exists.

## Maintenance rule

- Keep Evalution and GPTQModel aligned at the script boundary: shared concerns should stay in similarly named `ci_*` entrypoints even if the repo-specific workflow steps differ.
