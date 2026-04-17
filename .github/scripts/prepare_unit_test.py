import argparse
import json

from common import append_github_env, append_github_output
from unit_test_config import resolve_unit_test_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", required=True)
    args = parser.parse_args()

    config = resolve_unit_test_config(args.test_file)

    append_github_env("SAFE_NAME", config.safe_name)
    append_github_env("TEST_REQUIRES_GPU", str(config.requires_gpu).lower())
    append_github_env("PYTHON_VERSION", config.python_version)
    append_github_env("UV_PYTHON", config.uv_python)

    append_github_output("safe-name", config.safe_name)
    append_github_output("requires-gpu", str(config.requires_gpu).lower())
    append_github_output("python-version", config.python_version)
    append_github_output("uv-python", config.uv_python)

    print(json.dumps(config.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
