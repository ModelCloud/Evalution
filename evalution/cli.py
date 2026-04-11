# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from typing import Sequence

from evalution.yaml import python_from_yaml, run_yaml


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI entry point for this module. Preserve the fallback order expected by the surrounding caller."""
    normalized_argv = list(argv) if argv is not None else None
    if (
        normalized_argv
        and normalized_argv[0] not in {"run", "emit-python"}
        and not normalized_argv[0].startswith("-")
    ):
        # Treat a bare path as `evalution run <path>` for the common YAML-execution case.
        normalized_argv = ["run", *normalized_argv]

    parser = _build_parser()
    args = parser.parse_args(normalized_argv)

    if args.command == "run":
        # Evalution progress logs currently write to stdout, so redirect them to stderr
        # while the run is active and keep stdout reserved for the final JSON payload.
        with redirect_stdout(sys.stderr):
            result = run_yaml(args.spec)
            payload = json.dumps(result.to_dict(), indent=2, sort_keys=False)
        if args.output is None:
            print(payload)
        else:
            Path(args.output).write_text(payload + "\n", encoding="utf-8")
        return 0

    if args.command == "emit-python":
        print(python_from_yaml(args.spec))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    """Build parser."""
    parser = argparse.ArgumentParser(
        prog="evalution",
        description="Run Evalution YAML specs or emit equivalent Python code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="execute a YAML spec and emit the structured result as JSON",
    )
    run_parser.add_argument("spec", help="path to the Evalution YAML spec")
    run_parser.add_argument(
        "--output",
        help="optional path for the JSON run result; stdout is used when omitted",
    )

    python_parser = subparsers.add_parser(
        "emit-python",
        help="print the equivalent fluent Python script for a YAML spec",
    )
    python_parser.add_argument("spec", help="path to the Evalution YAML spec")
    return parser
