import argparse
import os
import subprocess

from common import append_github_env, normalize_test_file


TORCHAO_CPU_WHEEL = (
    "https://download.pytorch.org/whl/cpu/"
    "torchao-0.17.0%2Bcpu-py3-none-any.whl"
    "#sha256=6c0ce8b506c72be4efb1f0c6fd1679cb58145efebb20d51ac1adf7a7b3ebb872"
)


def run(cmd: list[str]) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def uv_install(*packages: str, upgrade: bool = False) -> None:
    if not packages:
        return
    cmd = ["uv", "pip", "install"]
    if upgrade:
        cmd.append("-U")
    cmd.extend(packages)
    run(cmd)


def install_flash_attn(uv_python: str, runner: str) -> None:
    if uv_python == "3.14t":
        uv_install(f"http://{runner}/files/flash_attn/flash_attn-2.8.4-cp314-cp314t-linux_x86_64.whl")
        return

    if uv_python == "3.12":
        append_github_env("EVALUTION_SKIP_GIL_CHECK", "1")
        uv_install("tensorrt_llm", upgrade=True)
        uv_install(f"http://{runner}/files/flash_attn/flash_attn-2.8.4-cp312-cp312-linux_x86_64.whl")
        return

    uv_install("flash-attn")
    run(["uv", "pip", "show", "flash-attn"])


def install_test_specific_deps(test_file: str) -> None:
    if test_file != "tests/test_gptqmodel_engine.py":
        return

    uv_install("accelerate", upgrade=True)
    uv_install(TORCHAO_CPU_WHEEL, upgrade=True)

    print("== installing gptqmodel ==")
    uv_install("gptqmodel", upgrade=True)
    run(["uv", "pip", "show", "gptqmodel"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--runner", default=os.environ.get("RUNNER", "10.0.13.31"))
    parser.add_argument("--uv-python", default=os.environ.get("UV_PYTHON", ""))
    parser.add_argument("--install-project", action="store_true")
    args = parser.parse_args()

    test_file = normalize_test_file(args.test_file)

    if args.install_project:
        uv_install(".")
        uv_install("pytest", "datasets", "rouge_score", "sglang", "pybase64", upgrade=True)

    install_flash_attn(args.uv_python, args.runner)
    install_test_specific_deps(test_file)


if __name__ == "__main__":
    main()
