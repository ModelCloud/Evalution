# CI Architecture

## Naming

- Workflow entrypoints follow the same `ci_*.py` convention as GPTQModel.
- `ci_common.py`, `ci_gpu.py`, `ci_tests.py`, and `ci_workflow.py` are the only Python CI entrypoints that workflows should call directly.

## Evalution unit test flow

1. `list-test-files`
- `ci_workflow.py list-tests` enumerates `tests/test_*.py`, applies the optional regex filter, and emits separate CPU, torch, and model matrices.

2. `test` job setup
- `ci_tests.py set-metadata` derives `SAFE_NAME` and whether the test requires a GPU by reading the file marker and test-specific Python runtime rules.
- `ci_restore_uv_cache.sh` restores the shared uv cache before each reusable job matrix starts.
- The reusable workflow activates the uv env and calls `ci_tests.py install-deps` to install test-specific Python dependencies.

3. execution
- `ci_gpu.py allocate` reserves a GPU only when the test requires one.
- `ci_tests.py run` executes pytest, writes artifacts, and optionally keeps the GPU lease alive while the test is running.
- `ci_gpu.py release` releases the GPU only when allocation metadata exists.

## Maintenance rule

- Keep Evalution and GPTQModel aligned at the script boundary: shared concerns should stay in similarly named `ci_*` entrypoints even if the repo-specific workflow steps differ.
