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


def test_project_python_files_use_compiled_pcre_execution_apis() -> None:
    root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    forbidden_attrs = {"search", "match", "fullmatch", "findall", "finditer", "sub", "split"}

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if not isinstance(func.value, ast.Name) or func.value.id != "pcre":
                continue
            if func.attr not in forbidden_attrs:
                continue
            offenders.append(
                f"{path.relative_to(root)}:{node.lineno}: pcre.{func.attr}(...) should use a compiled pattern"
            )

    assert offenders == []
