import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from ci_common import append_github_env, append_github_output, run_command

_COMMON_TEST_ENV_FAMILY = "unit_test_common"
_TENSORRT_LLM_TEST_ENV_FAMILY = "unit_test_tensorrt_llm"
_TENSORRT_LLM_TEST_FILE = "tests/test_tensorrt_llm_engine.py"


def sort_key(path: Path, root: Path) -> tuple[int, str]:
    rel = path.relative_to(root)
    return (len(rel.parts), path.as_posix())


def command_set_matrix_config(args: argparse.Namespace) -> int:
    append_github_output("max-parallel", args.max_parallel)
    print(f"max-parallel={args.max_parallel}")
    return 0


def command_list_tests(args: argparse.Namespace) -> int:
    root = Path(args.tests_root)
    regex = re.compile(args.test_regex) if args.test_regex else None

    files: list[str] = []
    for path in sorted(root.rglob("test_*.py"), key=lambda item: sort_key(item, root)):
        rel = path.as_posix()
        if regex and not regex.search(rel):
            continue
        files.append(rel)

    append_github_output("files", json.dumps(files))
    print("Matched test files:")
    for rel in files:
        print(rel)
    return 0


def command_set_test_metadata(args: argparse.Namespace) -> int:
    test_file = Path(args.test_file)
    safe_name = args.test_file.replace("/", "__").replace(".", "_")
    lines = test_file.read_text(encoding="utf-8").splitlines()
    requires_gpu = "false" if any(line == "# GPU=-1" for line in lines) else "true"

    append_github_env("SAFE_NAME", safe_name)
    append_github_env("TEST_REQUIRES_GPU", requires_gpu)
    append_github_output("safe-name", safe_name)
    append_github_output("requires-gpu", requires_gpu)
    print(f"safe-name={safe_name}")
    print(f"requires-gpu={requires_gpu}")
    return 0


def sanitize_env_component(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "shared"


def resolve_test_env_signature(
    test_file: str,
    default_python_version: str,
    default_uv_python: str,
) -> tuple[str, str, str]:
    if test_file == _TENSORRT_LLM_TEST_FILE:
        return "3.12", "3.12", _TENSORRT_LLM_TEST_ENV_FAMILY
    return default_python_version, default_uv_python, _COMMON_TEST_ENV_FAMILY


def build_uv_env_name(
    *,
    env_family: str,
    env_scope: str,
) -> str:
    safe_scope = sanitize_env_component(env_scope)
    return f"evalution_{env_family}_{safe_scope}"


def env_state_dir(env_name: str) -> Path:
    state_dir = Path("/opt/uv/.ci-state") / sanitize_env_component(env_name)
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


@contextlib.contextmanager
def hold_env_lock(env_name: str):
    lock_path = env_state_dir(env_name) / "env.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        print(f"waiting for env lock={lock_path}")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        print(f"acquired env lock={lock_path}")
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def stamp_path(env_name: str, stamp_name: str) -> Path:
    return env_state_dir(env_name) / f"{sanitize_env_component(stamp_name)}.stamp"


def write_stamp(path: Path, message: str) -> None:
    path.write_text(f"{message}\n", encoding="utf-8")


def command_activate_uv_env(args: argparse.Namespace) -> int:
    python_version, uv_python, env_family = resolve_test_env_signature(
        args.test_file,
        args.python_version,
        args.uv_python,
    )
    append_github_env("PYTHON_VERSION", python_version)
    append_github_env("UV_PYTHON", uv_python)
    append_github_env("ENV_FAMILY", env_family)

    env_name = build_uv_env_name(
        env_family=env_family,
        env_scope=args.env_scope,
    )
    append_github_env("ENV_NAME", env_name)

    base_uv_cache_dir = os.environ.get("UV_CACHE_DIR", "")
    uv_cache_dir = f"{base_uv_cache_dir.rstrip('/')}/{env_name}"
    Path(uv_cache_dir).mkdir(parents=True, exist_ok=True)
    append_github_env("UV_CACHE_DIR", uv_cache_dir)
    print(f"using shared env family={env_family}")
    print(f"using uv cache dir={uv_cache_dir}")

    script = f"""
set -e
echo "::group::printenv"
printenv | grep UV_
echo "::endgroup::"
echo "source /opt/uv/setup_uv_venv.sh {env_name} {python_version}"
source /opt/uv/setup_uv_venv.sh "{env_name}" "{python_version}"
python -VV
echo "::group::uv pip list"
uv pip list
echo "::endgroup::"
"""
    subprocess.check_call(["bash", "-lc", script])
    return 0


def command_setup_uv_env(args: argparse.Namespace) -> int:
    env_name = os.environ.get("ENV_NAME", "")
    if args.uv_python == "3.12":
        append_github_env("EVALUTION_SKIP_GIL_CHECK", "1")

    if not env_name:
        print("ENV_NAME is not set; falling back to eager per-step setup.")
    setup_stamp = stamp_path(env_name, "setup_uv_env") if env_name else None

    with hold_env_lock(env_name or "fallback_setup_uv_env"):
        if setup_stamp is not None and setup_stamp.exists():
            print(f"shared env already prepared: {env_name}")
            return 0

        run_command(
            [
                "bash",
                "-lc",
                f'echo "::group::init env..."\n'
                f'/opt/env/init_compiler_torch_only.sh {args.cuda_version} {args.torch_version} {args.uv_python}\n'
                f'echo "::endgroup::"',
            ]
        )

        if args.uv_python == "3.14t":
            run_command(
                [
                    "bash",
                    "-lc",
                    'echo "::group::installing flash_attn with 3.14t..."\n'
                    "uv pip install http://10.0.13.31/files/flash_attn/flash_attn-2.8.4-cp314-cp314t-linux_x86_64.whl\n"
                    'echo "::endgroup::"',
                ]
            )
        elif args.uv_python == "3.12":
            run_command(
                [
                    "bash",
                    "-lc",
                    'echo "::group::installing tensorrt_llm..."\n'
                    "uv pip install tensorrt_llm -U\n"
                    'echo "::endgroup::"',
                ]
            )
            run_command(
                [
                    "bash",
                    "-lc",
                    'echo "::group::installing flash_attn with 3.12..."\n'
                    "uv pip install http://10.0.13.31/files/flash_attn/flash_attn-2.8.4-cp312-cp312-linux_x86_64.whl\n"
                    'echo "::endgroup::"',
                ]
            )
        else:
            run_command(
                [
                    "bash",
                    "-lc",
                    'echo "::group::installing flash_attn..."\n'
                    "uv pip install flash-attn\n"
                    "uv pip show flash-attn\n"
                    'echo "::endgroup::"',
                ]
            )

        run_command(
            [
                "bash",
                "-lc",
                'echo "::group::installing accelerate..."\n'
                "uv pip install accelerate -U\n"
                'echo "::endgroup::"',
            ]
        )
        run_command(
            [
                "bash",
                "-lc",
                'echo "::group::installing gptqmodel..."\n'
                "uv pip install gptqmodel -U\n"
                "uv pip show gptqmodel\n"
                'echo "::endgroup::"',
            ]
        )

        if setup_stamp is not None:
            write_stamp(
                setup_stamp,
                f"uv_python={args.uv_python} cuda={args.cuda_version} torch={args.torch_version}",
            )
    return 0


def command_print_uv_env(args: argparse.Namespace) -> int:
    script = """
set -e
echo "::group::uv python list"
uv python list
echo "::endgroup::"
echo "== python =="
python -VV
which python
which pip || true
echo "== nvcc =="
nvcc --version
echo "::group::pip list"
uv pip list
echo "::endgroup::"
echo "== torch =="
uv pip show torch || true
echo "::group::project files"
ls -ahl
echo "::endgroup::"
echo "::group::git status"
git config --global --add safe.directory "$(pwd)"
git status
echo "::endgroup::"
"""
    subprocess.check_call(["bash", "-lc", script])
    return 0


def command_prepare_test_run(args: argparse.Namespace) -> int:
    if not Path("tests").is_dir():
        print("::error::tests/ directory not found.")
        return 1

    env_name = os.environ.get("ENV_NAME", "")
    if not env_name:
        print("ENV_NAME is not set; falling back to eager test preparation.")
    prepare_stamp = stamp_path(env_name, "prepare_test_run") if env_name else None

    with hold_env_lock(env_name or "fallback_prepare_test_run"):
        if prepare_stamp is not None and prepare_stamp.exists():
            print(f"shared test env already prepared: {env_name}")
            return 0

        run_command(["uv", "pip", "install", "."])
        run_command(["uv", "pip", "install", "-U", "pytest", "datasets", "rouge_score", "sglang", "pybase64"])
        run_command(
            [
                "bash",
                "-lc",
                'echo "::group::pip list"\n'
                "uv pip list\n"
                'echo "::endgroup::"',
            ]
        )

        if prepare_stamp is not None:
            write_stamp(prepare_stamp, "project and test dependencies installed")
    return 0


def command_release_gpu_if_present(args: argparse.Namespace) -> int:
    cuda_visible = args.cuda_visible_devices
    step_timestamp = args.step_timestamp
    if not cuda_visible or not step_timestamp:
        print("Skip GPU release because allocation metadata is missing.")
        return 0

    return run_command(
        [
            "python3",
            ".github/scripts/ci_gpu.py",
            "release",
            "--base-url",
            args.base_url,
            "--run-id",
            args.run_id,
            "--gpu-id",
            cuda_visible,
            "--timestamp",
            step_timestamp,
            "--test",
            args.test_file,
            "--runner",
            args.runner,
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_matrix = subparsers.add_parser("set-matrix-config")
    set_matrix.add_argument("--max-parallel", required=True)
    set_matrix.set_defaults(handler=command_set_matrix_config)

    list_tests = subparsers.add_parser("list-tests")
    list_tests.add_argument("--tests-root", default="tests")
    list_tests.add_argument("--test-regex", default="")
    list_tests.set_defaults(handler=command_list_tests)

    metadata = subparsers.add_parser("set-test-metadata")
    metadata.add_argument("--test-file", required=True)
    metadata.set_defaults(handler=command_set_test_metadata)

    activate_env = subparsers.add_parser("activate-uv-env")
    activate_env.add_argument("--test-file", required=True)
    activate_env.add_argument("--safe-name", required=True)
    activate_env.add_argument("--cuda-version", required=True)
    activate_env.add_argument("--torch-version", required=True)
    activate_env.add_argument("--python-version", required=True)
    activate_env.add_argument("--uv-python", required=True)
    activate_env.add_argument("--env-scope", default="shared")
    activate_env.set_defaults(handler=command_activate_uv_env)

    setup_env = subparsers.add_parser("setup-uv-env")
    setup_env.add_argument("--cuda-version", required=True)
    setup_env.add_argument("--torch-version", required=True)
    setup_env.add_argument("--uv-python", required=True)
    setup_env.set_defaults(handler=command_setup_uv_env)

    print_env = subparsers.add_parser("print-uv-env")
    print_env.set_defaults(handler=command_print_uv_env)

    prepare_run = subparsers.add_parser("prepare-test-run")
    prepare_run.set_defaults(handler=command_prepare_test_run)

    release_gpu = subparsers.add_parser("release-gpu-if-present")
    release_gpu.add_argument("--base-url", required=True)
    release_gpu.add_argument("--run-id", required=True)
    release_gpu.add_argument("--cuda-visible-devices", default="")
    release_gpu.add_argument("--step-timestamp", default="")
    release_gpu.add_argument("--test-file", required=True)
    release_gpu.add_argument("--runner", required=True)
    release_gpu.set_defaults(handler=command_release_gpu_if_present)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
