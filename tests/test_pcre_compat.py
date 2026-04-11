# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# GPU=-1
from __future__ import annotations

import importlib

import pcre

# Keep shared test fixtures and expectations explicit at module scope.
_MATHQA_OPTIONS_RE = pcre.compile(r"[abcd] \) .*?, |e \) .*?$")
_CAMEL_BOUNDARY_RE = pcre.compile(r"(?<!^)(?=[A-Z])")
_NON_ALNUM_RE = pcre.compile(r"[^a-z0-9]+")
_PAREN_CONTENT_RE = pcre.compile(r"\(\s*([^\)]*?)\s*\)")
_WHITESPACE_RE = pcre.compile(r"\s+")
_RESULT_JSON_RE = pcre.compile(r"RESULT_JSON_START\n(.*?)\nRESULT_JSON_END", pcre.DOTALL)


def test_pcre_matches_affected_project_patterns() -> None:
    """Verify PCRE matches affected project patterns."""
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


def test_recent_replacements_execute_compatibly_on_compiled_pcre_patterns() -> None:
    """Verify recent replacements execute compatibly on compiled PCRE patterns."""
    xlsum_module = importlib.import_module("evalution.datasets.xlsum")
    qasper_module = importlib.import_module("evalution.benchmarks.qasper")
    longbench_module = importlib.import_module("evalution.scorers.longbench")
    scrolls_module = importlib.import_module("evalution.benchmarks.scrolls")
    mlqa_module = importlib.import_module("evalution.benchmarks.mlqa")
    ruler_module = importlib.import_module("evalution.benchmarks.ruler")

    patterns = (
        xlsum_module._INLINE_SPACES_RE,
        qasper_module._QASPER_ARTICLES_RE,
        longbench_module._EN_ARTICLES_RE,
        longbench_module._DIGIT_RE,
        longbench_module._PARAGRAPH_RE,
        longbench_module._PARAGRAPH_ZH_RE,
        scrolls_module._QUALITY_CHOICE_PATTERN,
        mlqa_module._MLQA_CJK_RE,
        mlqa_module._MLQA_ARTICLES_EN_RE,
        mlqa_module._MLQA_ARTICLES_ES_RE,
        mlqa_module._MLQA_ARTICLES_VI_RE,
        mlqa_module._MLQA_ARTICLES_DE_RE,
        mlqa_module._MLQA_ARTICLES_AR_RE,
        ruler_module._CONTROL_CHARS_RE,
    )
    assert all(isinstance(pattern, pcre.Pattern) for pattern in patterns)

    assert xlsum_module._normalize_inline_spaces("uno   dos") == "uno dos"
    assert qasper_module._normalize_qasper_answer("The, cat!") == "cat"

    paragraph_match = longbench_module._PARAGRAPH_RE.search("Paragraph 42")
    assert paragraph_match is not None
    assert paragraph_match.group(1) == "42"
    paragraph_zh_match = longbench_module._PARAGRAPH_ZH_RE.search("段落7")
    assert paragraph_zh_match is not None
    assert paragraph_zh_match.group(1) == "7"

    choices, passage = scrolls_module._quality_choices_and_context(
        "(A) one (B) two (C) three (D) four\n\nPassage"
    )
    assert choices == ["one", "two", "three", "four"]
    assert passage == "Passage"

    assert mlqa_module._MLQA_CJK_RE.search("漢") is not None
    assert mlqa_module._normalize_mlqa_answer("The, cat!", "en") == "cat"
    assert mlqa_module._normalize_mlqa_answer("el gato", "es") == "gato"
    assert mlqa_module._normalize_mlqa_answer("البيت", "ar") == "بيت"

    assert ruler_module._CONTROL_CHARS_RE.sub(" ", "a\x00b\x1fc") == "a b c"
