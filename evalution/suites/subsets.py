# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def normalize_subset_token(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in text.split("_") if part)


def normalize_subset_path(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("subset must be a non-empty string")
    if "." not in text:
        return normalize_subset_token(text)

    tokens = [normalize_subset_token(part) for part in text.split(".")]
    if any(not token for token in tokens):
        raise ValueError(f"subset path {value!r} contains an empty segment")
    return ".".join(tokens)


@dataclass(frozen=True, slots=True)
class ResolvedSubset:
    canonical: str
    path: tuple[str, ...]
    kind: str
    leaf_values: tuple[str, ...]
    leaf_paths: tuple[str, ...]


@dataclass(slots=True)
class _SubsetNode:
    children: dict[str, _SubsetNode]
    leaf_value: str | None = None


class SubsetTree:
    def __init__(self, spec: Mapping[str, Any]):
        self._root = self._build_node(spec)
        self._leaf_path_by_value: dict[str, str] = {}
        self._index_leaf_paths(self._root, ())

    def resolve(self, subset: Any) -> ResolvedSubset:
        canonical = normalize_subset_path(subset)
        if canonical == "all":
            leaf_paths, leaf_values = self._descendant_leaves(self._root, ())
            return ResolvedSubset(
                canonical="all",
                path=("all",),
                kind="all",
                leaf_values=leaf_values,
                leaf_paths=leaf_paths,
            )

        node = self._root
        path_tokens: list[str] = []
        for token in canonical.split("."):
            child = node.children.get(token)
            if child is None:
                allowed = ", ".join(sorted(node.children))
                raise ValueError(
                    f"unknown subset path {canonical!r}: segment {token!r} is not defined"
                    + (f"; expected one of: {allowed}" if allowed else "")
                )
            node = child
            path_tokens.append(token)

        leaf_paths, leaf_values = self._descendant_leaves(node, tuple(path_tokens))
        return ResolvedSubset(
            canonical=canonical,
            path=tuple(path_tokens),
            kind="leaf" if node.leaf_value is not None else "node",
            leaf_values=leaf_values,
            leaf_paths=leaf_paths,
        )

    def leaf_subset(self, value: Any) -> str:
        normalized_value = normalize_subset_token(value)
        subset = self._leaf_path_by_value.get(normalized_value)
        if subset is None:
            raise ValueError(f"leaf subset for {value!r} is not defined")
        return subset

    def leaf_subset_path(self, value: Any) -> list[str]:
        return self.leaf_subset(value).split(".")

    def _build_node(self, spec: Mapping[str, Any]) -> _SubsetNode:
        children: dict[str, _SubsetNode] = {}
        for raw_token, raw_child in spec.items():
            token = normalize_subset_token(raw_token)
            if not token:
                raise ValueError("subset nodes must use non-empty names")
            if token in children:
                raise ValueError(f"duplicate subset token {token!r} in subset tree")
            if isinstance(raw_child, Mapping):
                children[token] = self._build_node(raw_child)
                continue
            children[token] = _SubsetNode(children={}, leaf_value=str(raw_child))
        return _SubsetNode(children=children)

    def _descendant_leaves(
        self,
        node: _SubsetNode,
        prefix: tuple[str, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if node.leaf_value is not None:
            return (".".join(prefix),), (node.leaf_value,)

        leaf_paths: list[str] = []
        leaf_values: list[str] = []
        for token, child in node.children.items():
            child_leaf_paths, child_leaf_values = self._descendant_leaves(child, prefix + (token,))
            leaf_paths.extend(child_leaf_paths)
            leaf_values.extend(child_leaf_values)
        return tuple(leaf_paths), tuple(leaf_values)

    def _index_leaf_paths(
        self,
        node: _SubsetNode,
        prefix: tuple[str, ...],
    ) -> None:
        if node.leaf_value is not None:
            normalized_value = normalize_subset_token(node.leaf_value)
            canonical = ".".join(prefix)
            if normalized_value in self._leaf_path_by_value:
                raise ValueError(f"duplicate leaf subset value {normalized_value!r} in subset tree")
            self._leaf_path_by_value[normalized_value] = canonical
            return

        for token, child in node.children.items():
            self._index_leaf_paths(child, prefix + (token,))
