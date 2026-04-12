# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from shutil import move
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile

# Pin the original MeQSum spreadsheet so evaluation never executes the remote BigBio dataset script.
MEQSUM_DATASET_PATH = "bigbio/meqsum"
MEQSUM_DATASET_NAME = "meqsum_source"
MEQSUM_SOURCE_URL = (
    "https://github.com/abachaa/MeQSum/raw/master/"
    "MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx"
)
MEQSUM_SOURCE_SHA256 = "abedd939d5014306ca576522416bf69103e85dc8fcf1668a4099e8b84a39eeea"
_MEQSUM_NAMESPACE = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_MEQSUM_ROW_LOCK = Lock()
_MEQSUM_SPLIT_ALIASES = {
    "train": "train",
    "validation": "train",
    "test": "train",
}


def _default_cache_root() -> Path:
    """Implement default cache root for this module."""
    return Path.home() / ".cache" / "evalution" / "datasets"


def _cache_root(cache_dir: str | None) -> Path:
    """Implement cache root for this module."""
    if cache_dir is None:
        return _default_cache_root()
    return Path(cache_dir) / "evalution-vendored-datasets"


def _sha256_file(path: Path) -> str:
    """Implement sha256 file for this module."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolved_split(split: str) -> str:
    """Implement resolved split for this module."""
    resolved = _MEQSUM_SPLIT_ALIASES.get(split.strip().lower())
    if resolved is None:
        raise ValueError(
            f"unsupported meqsum split: {split!r}; expected one of {tuple(_MEQSUM_SPLIT_ALIASES)}"
        )
    return resolved


def _download_source_xlsx(cache_dir: str | None) -> Path:
    """Implement download source xlsx for this module."""
    cache_root = _cache_root(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    target_path = cache_root / f"meqsum-{MEQSUM_SOURCE_SHA256[:12]}.xlsx"
    if target_path.exists():
        return target_path

    with NamedTemporaryFile(dir=cache_root, suffix=".xlsx", delete=False) as handle:
        temporary_path = Path(handle.name)
        with urlopen(MEQSUM_SOURCE_URL) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    move(str(temporary_path), str(target_path))
    return target_path


def _verified_source_xlsx(cache_dir: str | None) -> Path:
    """Implement verified source xlsx for this module."""
    xlsx_path = _download_source_xlsx(cache_dir)
    actual_sha256 = _sha256_file(xlsx_path)
    if actual_sha256 != MEQSUM_SOURCE_SHA256:
        raise ValueError(
            "meqsum source checksum mismatch: "
            f"expected {MEQSUM_SOURCE_SHA256}, got {actual_sha256}"
        )
    return xlsx_path


def _column_index(cell_reference: str) -> int:
    """Implement column index for this module."""
    column_name = "".join(character for character in cell_reference if character.isalpha())
    index = 0
    for character in column_name:
        index = index * 26 + (ord(character.upper()) - ord("A") + 1)
    return max(index - 1, 0)


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Implement shared strings for this module."""
    shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in shared.findall("a:si", _MEQSUM_NAMESPACE):
        values.append("".join(node.text or "" for node in item.findall(".//a:t", _MEQSUM_NAMESPACE)))
    return values


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    """Implement cell value for this module."""
    if cell.get("t") == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//a:t", _MEQSUM_NAMESPACE))
    value_node = cell.find("a:v", _MEQSUM_NAMESPACE)
    if value_node is None or value_node.text is None:
        return ""
    if cell.get("t") == "s":
        return shared_strings[int(value_node.text)]
    return value_node.text


def _load_source_rows(xlsx_path: Path) -> list[dict[str, str]]:
    """Load source rows. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    with zipfile.ZipFile(xlsx_path) as archive:
        shared = _shared_strings(archive)
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    row_values: list[list[str]] = []
    for row in sheet.findall(".//a:sheetData/a:row", _MEQSUM_NAMESPACE):
        cells: dict[int, str] = {}
        max_index = -1
        for cell in row.findall("a:c", _MEQSUM_NAMESPACE):
            cell_reference = cell.get("r", "")
            index = _column_index(cell_reference)
            cells[index] = _cell_value(cell, shared)
            max_index = max(max_index, index)
        if max_index < 0:
            continue
        row_values.append([cells.get(index, "") for index in range(max_index + 1)])

    if not row_values:
        return []

    headers = [value.strip() for value in row_values[0]]
    rows: list[dict[str, str]] = []
    for row in row_values[1:]:
        item = {
            str(header): str(row[index]).strip() if index < len(row) else ""
            for index, header in enumerate(headers)
            if header
        }
        if item:
            rows.append(item)
    return rows


@lru_cache(maxsize=None)
def _load_rows_cached(*, cache_root_key: str) -> tuple[dict[str, str], ...]:
    """Load rows cached."""
    with _MEQSUM_ROW_LOCK:
        rows = _load_source_rows(_verified_source_xlsx(cache_root_key or None))
    return tuple(rows)


def load_meqsum_dataset(
    dataset_path: str,
    dataset_name: str | None = None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
) -> list[dict[str, str]]:
    """Load meqsum dataset."""
    if dataset_path != MEQSUM_DATASET_PATH:
        raise ValueError(f"meqsum dataset_path must be {MEQSUM_DATASET_PATH!r}, got {dataset_path!r}")
    if dataset_name not in {None, MEQSUM_DATASET_NAME}:
        raise ValueError(f"meqsum dataset_name must be {MEQSUM_DATASET_NAME!r}, got {dataset_name!r}")
    if stream:
        raise ValueError("meqsum vendored loader does not support stream=True")
    _resolved_split(split)
    return list(_load_rows_cached(cache_root_key=cache_dir or ""))
