# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pcre


def test_pcre_matches_affected_project_patterns() -> None:
    option_label = pcre.compile(r"^\(?([A-Z])\)?(?:[.:：、]|．)?\s*")
    assert option_label.match("(C) choice").group(1) == "C"

    answer_prefix = pcre.compile(r"^(?:A:|Answer:|The answer is)\s*", pcre.IGNORECASE)
    assert answer_prefix.sub("", "Answer: yes") == "yes"
    assert answer_prefix.sub("", "the answer is (B)") == "(B)"

    assert pcre.compile(r"\b(True|False|Yes|No)\b").search("Maybe Yes later").group(1) == "Yes"
    assert pcre.compile(r"\([A-Z]\)").fullmatch("(A)") is not None
    assert pcre.compile(r"[A-Z]").fullmatch("B") is not None

    assert pcre.findall(r"[abcd] \) .*?, |e \) .*?$", "a ) 4, b ) 5, c ) 6, d ) 3, e ) 7") == [
        "a ) 4, ",
        "b ) 5, ",
        "c ) 6, ",
        "d ) 3, ",
        "e ) 7",
    ]

    assert pcre.sub(r"(?<!^)(?=[A-Z])", "_", "DisabilityStatus") == "Disability_Status"
    assert pcre.sub(r"[^a-z0-9]+", "_", "disability-status").strip("_") == "disability_status"

    assert pcre.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", "( spaced )") == "(spaced)"
    assert pcre.split(r"\s+", "a  b\tc") == ["a", "b", "c"]

    split_puncts = pcre.compile(r"[\w]+|[^\s\w]")
    assert split_puncts.findall("Hi, there!") == ["Hi", ",", "there", "!"]

    articles = pcre.compile(r"\b(a|an|the)\b")
    whitespace = pcre.compile(r"\s+")
    assert whitespace.sub(" ", articles.sub(" ", "the quick an fox")).strip() == "quick fox"

    payload = "prefix\nRESULT_JSON_START\n{\"ok\": true}\nRESULT_JSON_END\nsuffix"
    match = pcre.search(r"RESULT_JSON_START\n(.*?)\nRESULT_JSON_END", payload, pcre.DOTALL)
    assert match is not None
    assert match.group(1) == "{\"ok\": true}"
