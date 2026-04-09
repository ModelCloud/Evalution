# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import pcre

_MATHQA_OPTIONS_RE = pcre.compile(r"[abcd] \) .*?, |e \) .*?$")
_CAMEL_BOUNDARY_RE = pcre.compile(r"(?<!^)(?=[A-Z])")
_NON_ALNUM_RE = pcre.compile(r"[^a-z0-9]+")
_PAREN_CONTENT_RE = pcre.compile(r"\(\s*([^\)]*?)\s*\)")
_WHITESPACE_RE = pcre.compile(r"\s+")
_RESULT_JSON_RE = pcre.compile(r"RESULT_JSON_START\n(.*?)\nRESULT_JSON_END", pcre.DOTALL)


def test_pcre_matches_affected_project_patterns() -> None:
    option_label = pcre.compile(r"^\(?([A-Z])\)?(?:[.:：、]|．)?\s*")
    assert option_label.match("(C) choice").group(1) == "C"

    answer_prefix = pcre.compile(r"^(?:A:|Answer:|The answer is)\s*", pcre.IGNORECASE)
    assert answer_prefix.sub("", "Answer: yes") == "yes"
    assert answer_prefix.sub("", "the answer is (B)") == "(B)"

    assert pcre.compile(r"\b(True|False|Yes|No)\b").search("Maybe Yes later").group(1) == "Yes"
    assert pcre.compile(r"\([A-Z]\)").fullmatch("(A)") is not None
    assert pcre.compile(r"[A-Z]").fullmatch("B") is not None

    assert _MATHQA_OPTIONS_RE.findall("a ) 4, b ) 5, c ) 6, d ) 3, e ) 7") == [
        "a ) 4, ",
        "b ) 5, ",
        "c ) 6, ",
        "d ) 3, ",
        "e ) 7",
    ]

    assert _CAMEL_BOUNDARY_RE.sub("_", "DisabilityStatus") == "Disability_Status"
    assert _NON_ALNUM_RE.sub("_", "disability-status").strip("_") == "disability_status"

    assert _PAREN_CONTENT_RE.sub(r"(\1)", "( spaced )") == "(spaced)"
    assert _WHITESPACE_RE.split("a  b\tc") == ["a", "b", "c"]

    split_puncts = pcre.compile(r"[\w]+|[^\s\w]")
    assert split_puncts.findall("Hi, there!") == ["Hi", ",", "there", "!"]

    articles = pcre.compile(r"\b(a|an|the)\b")
    whitespace = pcre.compile(r"\s+")
    assert whitespace.sub(" ", articles.sub(" ", "the quick an fox")).strip() == "quick fox"

    payload = "prefix\nRESULT_JSON_START\n{\"ok\": true}\nRESULT_JSON_END\nsuffix"
    match = _RESULT_JSON_RE.search(payload)
    assert match is not None
    assert match.group(1) == "{\"ok\": true}"
