# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import ast
from pathlib import Path


def test_project_python_files_do_not_import_stdlib_regex_module() -> None:
    root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []

    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "re":
                        offenders.append(f"{path.relative_to(root)}:{node.lineno}: import re")
            elif isinstance(node, ast.ImportFrom) and node.module == "re":
                names = ", ".join(alias.name for alias in node.names)
                offenders.append(f"{path.relative_to(root)}:{node.lineno}: from re import {names}")

    assert offenders == []
