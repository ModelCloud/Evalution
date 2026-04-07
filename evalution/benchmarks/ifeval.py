# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubertcloud.ai, x.com/qubertcloud

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import pcre
from datasets import load_dataset

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest, InferenceSession
from evalution.results import SampleResult, TestResult


# IFEval benchmark source and split.
IFEVAL_DATASET_PATH = "google/IFEval"
IFEVAL_SPLIT = "train"

# Regex helpers (project-wide pcre requirement).
_ASTERISK_RE = pcre.compile(r"\*")
_BULLET_LINE_RE = pcre.compile(r"(?m)^\s*[-*]\s+.+$")
_BOLD_RE = pcre.compile(r"\*\*[^*]+\*\*")
_ITALIC_RE = pcre.compile(r"(?<!\*)\*[^*\n]+\*(?!\*)")
_END_QUOTE_RE = pcre.compile(r"\"$")
_TITLE_MARKER_RE = pcre.compile(r"<<([^>]+)>>")
_JSON_RE = pcre.compile(r"^```(?:json|JSON|Json)?\n?(.*?)\n?```$", pcre.DOTALL)
_WORD_RE = pcre.compile(r"[\w\p{L}']+")
_SENTENCE_END_RE = pcre.compile(r"(?<=[.!?])\s+")
_PLACEHOLDER_RE = pcre.compile(r"\[[^\]]+\]")
_POSTSCRIPT_PSD_RE = pcre.compile(r"\s*p\.\s*p?\.\s*s\.?")
_POSTSCRIPT_S_RE = pcre.compile(r"\s*p\.\s*s\.?")
_SECTION_RE_TEMPLATE = r"(?im)(^|\n)\s*{section}\s*\d+\b"
_FIRST_LINE_RE = pcre.compile(r"^[^\n]*\n")
_LAST_LINE_RE = pcre.compile(r"\n[^\n]*$")
_FIRST_WORD_RE = pcre.compile(r"^\W*([\w\p{L}']+)")
_ALPHANUM_RE = pcre.compile(r"[A-Za-z\p{L}]+")
_PUNCTUATION_NO_COMMA_RE = pcre.compile(r",|،")
_PARAGRAPH_SPLIT_RE = pcre.compile(r"\n\s*\n")


def _load_ifeval_dataset(dataset_path: str, dataset_name: str | None = None, **kwargs: Any) -> Any:
    """Load IFEval while ignoring the suite-level `stream` kwarg.

    Evalution uses `stream=` internally; convert it to the Hugging Face `streaming=`
    kwarg at the final loader boundary.
    """
    stream = kwargs.pop("stream", None)
    if stream is not None:
        kwargs["streaming"] = bool(stream)
    if dataset_name is None:
        return load_dataset(dataset_path, **kwargs)
    return load_dataset(dataset_path, dataset_name, **kwargs)
def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _coerce_int(value: Any, *, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _contains_char_in_range(text: str, start: int, end: int) -> bool:
    return any(start <= ord(char) <= end for char in text)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _normalize_language(value: Any) -> str:
    return _coerce_str(value) or "en"


def _build_keyword_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


try:  # pragma: no cover - optional dependency.
    from langdetect import detect as _detect_language
    from langdetect.lang_detect_exception import LangDetectException as _LangDetectException
except Exception:  # pragma: no cover
    _detect_language = None
    _LangDetectException = None


def _is_language_ok(text: str, *, language: str) -> bool:
    if not text.strip():
        return False
    language_code = _normalize_language(language)
    if _detect_language is not None:
        try:
            return _detect_language(text) == language_code
        except _LangDetectException:
            return False
    if language_code in {"en", "es", "fr", "de", "it", "pt", "fi", "sv", "id", "ca", "pl", "ro"}:
        return bool(_ALPHANUM_RE.search(text))
    if language_code == "zh":
        return _contains_char_in_range(text, 0x4E00, 0x9FFF)
    if language_code == "ja":
        return _contains_char_in_range(text, 0x3040, 0x30FF)
    if language_code == "ko":
        return _contains_char_in_range(text, 0xAC00, 0xD7A3)
    if language_code == "ar":
        return _contains_char_in_range(text, 0x0600, 0x06FF)
    if language_code in {"ru", "bg", "sr"}:
        return _contains_char_in_range(text, 0x0400, 0x04FF)
    return True


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(len(_SENTENCE_END_RE.split(stripped)), 1)


def _count_paragraphs(text: str) -> int:
    if not text.strip():
        return 0
    return len([chunk.strip() for chunk in _PARAGRAPH_SPLIT_RE.split(text.strip()) if chunk.strip()])


def _strip_thought(text: str) -> str:
    if not text:
        return text
    trimmed = text.strip()
    if trimmed.startswith("<think>") and "</think>" in trimmed:
        end = trimmed.find("</think>")
        return trimmed[end + len("</think>") :].strip()
    return text


def _remove_first_line(text: str) -> str:
    return _FIRST_LINE_RE.sub("", text, count=1).strip()


def _remove_last_line(text: str) -> str:
    return _LAST_LINE_RE.sub("", text, count=1).strip()


def _candidate_loose_outputs(text: str) -> list[str]:
    # IFEval loose checks retry strict rules after minimal cleanup.
    responses = [
        text,
        _ASTERISK_RE.sub("", text),
        _remove_first_line(text),
        _remove_last_line(text),
        _remove_first_line(_remove_last_line(text)),
    ]
    deduped = []
    for response in responses:
        clean = response.strip()
        if clean not in deduped:
            deduped.append(clean)
        no_stars = _ASTERISK_RE.sub("", response).strip()
        if no_stars and no_stars not in deduped:
            deduped.append(no_stars)
    return deduped


def _relation_check(count: int, *, threshold: int, relation: str) -> bool:
    if relation == "less than":
        return count < threshold
    return count >= threshold


def _build_instruction_checker(
    instruction_id: str,
    *,
    prompt: str,
    kwargs: dict[str, Any],
) -> Callable[[str], bool]:
    if instruction_id == "keywords:existence":
        keywords = _build_keyword_list(kwargs.get("keywords"))
        keyword_patterns = [pcre.compile(pcre.escape(keyword), pcre.IGNORECASE) for keyword in keywords]

        def _check(value: str) -> bool:
            if not keyword_patterns:
                return False
            for pattern in keyword_patterns:
                if pattern.search(value) is None:
                    return False
            return True

        return _check

    if instruction_id == "keywords:forbidden_words":
        forbidden = _build_keyword_list(kwargs.get("forbidden_words"))

        def _check(value: str) -> bool:
            for word in forbidden:
                pattern = pcre.compile(rf"(?<!\w){pcre.escape(word)}(?!\w)", pcre.IGNORECASE)
                if pattern.search(value):
                    return False
            return True

        return _check

    if instruction_id == "keywords:frequency":
        keyword = str(kwargs.get("keyword", "")).strip()
        frequency = _coerce_int(kwargs.get("frequency"), fallback=1)
        relation = _coerce_str(kwargs.get("relation")) or "at least"
        keyword_pattern = pcre.compile(pcre.escape(keyword), pcre.IGNORECASE) if keyword else None

        def _check(value: str) -> bool:
            if keyword_pattern is None:
                return False
            count = len(keyword_pattern.findall(value))
            return _relation_check(count, threshold=frequency, relation=relation)

        return _check

    if instruction_id == "keywords:letter_frequency":
        letter = str(kwargs.get("letter", "")).lower()
        frequency = _coerce_int(kwargs.get("let_frequency"), fallback=1)
        relation = _coerce_str(kwargs.get("let_relation")) or "at least"

        def _check(value: str) -> bool:
            if not letter:
                return False
            return _relation_check(
                value.lower().count(letter),
                threshold=frequency,
                relation=relation,
            )

        return _check

    if instruction_id == "language:response_language":
        language = str(kwargs.get("language", ""))

        def _check(value: str) -> bool:
            return _is_language_ok(value, language=language)

        return _check

    if instruction_id == "length_constraints:number_words":
        num_words = _coerce_int(kwargs.get("num_words"), fallback=1)
        relation = _coerce_str(kwargs.get("relation")) or "at least"

        def _check(value: str) -> bool:
            return _relation_check(
                _count_words(value),
                threshold=num_words,
                relation=relation,
            )

        return _check

    if instruction_id == "length_constraints:number_sentences":
        num_sentences = _coerce_int(kwargs.get("num_sentences"), fallback=1)
        relation = _coerce_str(kwargs.get("relation")) or "at least"

        def _check(value: str) -> bool:
            return _relation_check(
                _count_sentences(value),
                threshold=num_sentences,
                relation=relation,
            )

        return _check

    if instruction_id == "length_constraints:number_paragraphs":
        num_paragraphs = _coerce_int(kwargs.get("num_paragraphs"), fallback=1)

        def _check(value: str) -> bool:
            return _count_paragraphs(value) == num_paragraphs

        return _check

    if instruction_id == "length_constraints:nth_paragraph_first_word":
        num_paragraphs = _coerce_int(kwargs.get("num_paragraphs"), fallback=1)
        nth_paragraph = _coerce_int(kwargs.get("nth_paragraph"), fallback=1)
        first_word = _coerce_str(kwargs.get("first_word")) or ""

        def _check(value: str) -> bool:
            paragraphs = [chunk.strip() for chunk in _PARAGRAPH_SPLIT_RE.split(value) if chunk.strip()]
            if nth_paragraph < 1 or nth_paragraph > len(paragraphs):
                return False
            if num_paragraphs and len(paragraphs) != num_paragraphs:
                return False
            match = _FIRST_WORD_RE.search(paragraphs[nth_paragraph - 1])
            if match is None or not first_word:
                return False
            return match.group(1).lower() == first_word.lower()

        return _check

    if instruction_id == "detectable_content:number_placeholders":
        num_placeholders = _coerce_int(kwargs.get("num_placeholders"), fallback=1)

        def _check(value: str) -> bool:
            return len(_PLACEHOLDER_RE.findall(value)) >= num_placeholders

        return _check

    if instruction_id == "detectable_content:postscript":
        postscript_marker = str(kwargs.get("postscript_marker", "P.S.")).strip()

        def _check(value: str) -> bool:
            if not postscript_marker:
                return False
            lowered = value.lower()
            marker = postscript_marker.lower()
            if marker == "p.s.":
                return _POSTSCRIPT_S_RE.search(lowered) is not None
            if marker == "p.p.s":
                return _POSTSCRIPT_PSD_RE.search(lowered) is not None
            return f"{marker}" in lowered

        return _check

    if instruction_id == "detectable_format:number_bullet_lists":
        num_bullets = _coerce_int(kwargs.get("num_bullets"), fallback=1)

        def _check(value: str) -> bool:
            return len(_BULLET_LINE_RE.findall(value)) == num_bullets

        return _check

    if instruction_id == "detectable_format:number_highlighted_sections":
        num_highlights = _coerce_int(kwargs.get("num_highlights"), fallback=1)

        def _check(value: str) -> bool:
            highlights = [* _BOLD_RE.findall(value), * _ITALIC_RE.findall(value)]
            return len(highlights) >= num_highlights

        return _check

    if instruction_id == "detectable_format:multiple_sections":
        section_spliter = str(kwargs.get("section_spliter", "Section")).strip() or "Section"
        num_sections = _coerce_int(kwargs.get("num_sections"), fallback=1)
        pattern = pcre.compile(_SECTION_RE_TEMPLATE.format(section=pcre.escape(section_spliter)))

        def _check(value: str) -> bool:
            return len(pattern.findall(value)) >= num_sections

        return _check

    if instruction_id == "detectable_format:json_format":
        def _check(value: str) -> bool:
            candidate = value.strip()
            match = _JSON_RE.search(candidate)
            content = match.group(1) if match else candidate
            try:
                json.loads(content)
            except ValueError:
                return False
            return True

        return _check

    if instruction_id == "detectable_format:title":
        def _check(value: str) -> bool:
            return any(title.strip() for title in _TITLE_MARKER_RE.findall(value))

        return _check

    if instruction_id == "detectable_format:constrained_response":
        choices = ("My answer is yes.", "My answer is no.", "My answer is maybe.")

        def _check(value: str) -> bool:
            return any(choice in value for choice in choices)

        return _check

    if instruction_id == "combination:two_responses":
        def _check(value: str) -> bool:
            parts = value.split("******")
            if len(parts) != 2:
                return False
            left = parts[0].strip()
            right = parts[1].strip()
            return bool(left) and bool(right) and left != right

        return _check

    if instruction_id == "combination:repeat_prompt":
        prompt_to_repeat = str(kwargs.get("prompt_to_repeat", "")).strip()

        def _check(value: str) -> bool:
            return bool(prompt_to_repeat) and value.strip().startswith(prompt_to_repeat)

        return _check

    if instruction_id == "startend:end_checker":
        end_phrase = str(kwargs.get("end_phrase", "")).strip().lower()

        def _check(value: str) -> bool:
            if not end_phrase:
                return False
            return value.strip().lower().endswith(end_phrase)

        return _check

    if instruction_id == "change_case:capital_word_frequency":
        capital_frequency = _coerce_int(kwargs.get("capital_frequency"), fallback=0)
        relation = _coerce_str(kwargs.get("capital_relation")) or "at least"

        def _check(value: str) -> bool:
            capitals = [token for token in _WORD_RE.findall(value) if token.isupper() and len(token) > 1]
            return _relation_check(
                len(capitals),
                threshold=capital_frequency,
                relation=relation,
            )

        return _check

    if instruction_id == "change_case:english_capital":
        def _check(value: str) -> bool:
            return bool(value.strip()) and value.isupper() and _is_language_ok(value, language="en")

        return _check

    if instruction_id == "change_case:english_lowercase":
        def _check(value: str) -> bool:
            return bool(value.strip()) and value.islower() and _is_language_ok(value, language="en")

        return _check

    if instruction_id == "punctuation:no_comma":
        def _check(value: str) -> bool:
            return _PUNCTUATION_NO_COMMA_RE.search(value) is None

        return _check

    if instruction_id == "startend:quotation":
        def _check(value: str) -> bool:
            stripped = value.strip()
            return (
                len(stripped) >= 2
                and stripped.startswith('"')
                and _END_QUOTE_RE.search(stripped) is not None
            )

        return _check

    return lambda value: False


def _compute_instruction_scores(
    prompt: str,
    response: str,
    instruction_ids: list[str],
    kwargs_list: list[dict[str, Any]],
) -> dict[str, Any]:
    # Build strict and loose results for prompt-level and instruction-level scores.
    response_clean = _strip_thought(response)
    strict_values: list[bool] = []
    loose_values: list[bool] = []

    for index, instruction_id in enumerate(instruction_ids):
        raw_kwargs = kwargs_list[index] if index < len(kwargs_list) else {}
        if not isinstance(raw_kwargs, dict):
            raw_kwargs = {}
        checker = _build_instruction_checker(
            instruction_id,
            prompt=prompt,
            kwargs=raw_kwargs,
        )
        strict = bool(checker(response_clean))
        loose = False
        for candidate in _candidate_loose_outputs(response_clean):
            if checker(candidate):
                loose = True
                break
        strict_values.append(strict)
        loose_values.append(loose)

    return {
        "prompt_level_strict": all(strict_values),
        "prompt_level_loose": all(loose_values),
        "inst_level_strict": strict_values,
        "inst_level_loose": loose_values,
    }


@dataclass(slots=True)
class IFEval(BaseTestSuite):
    # IFEval benchmark instance with generation and instruction-level scoring.
    dataset_path: str = IFEVAL_DATASET_PATH
    dataset_name: str | None = None
    split: str = IFEVAL_SPLIT
    # Keep streaming enabled by default; the loader translates Evalution's `stream=` kwarg to
    # Hugging Face's `streaming=` argument at the dataset boundary.
    stream: bool = (False)
    max_new_tokens: int = 1280
    do_sample: bool = False
    temperature: float = 0.0
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None

    def task_name(self) -> str:
        return "ifeval"

    def dataset_loader(self) -> Callable[..., Any]:
        return _load_ifeval_dataset

    def result_metadata(self, *, generation_submission_mode: str) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "instruction_following",
            "primary_metric": "prompt_level_strict_acc",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        # One request per IFEval sample using only the prompt text.
        for index, doc in enumerate(docs):
            prompt = str(doc["prompt"]).strip()
            yield PreparedSample(
                index=index,
                doc=doc,
                target=prompt,
                request=GenerationRequest(
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    stop=None,
                ),
            )

    def score_sample(self, prepared_sample: PreparedSample, output: GenerationOutput) -> SampleResult:
        response = output.text
        instruction_id_list = [str(item) for item in prepared_sample.doc.get("instruction_id_list", [])]
        raw_kwargs = prepared_sample.doc.get("kwargs", [])
        kwargs_list = raw_kwargs if isinstance(raw_kwargs, list) else []
        scores = _compute_instruction_scores(
            prompt=prepared_sample.target,
            response=response,
            instruction_ids=instruction_id_list,
            kwargs_list=kwargs_list,
        )
        strict_values = scores["inst_level_strict"]
        loose_values = scores["inst_level_loose"]
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=response,
            extracted={
                "instruction_id_list": instruction_id_list,
                "prompt_level_strict": "1" if scores["prompt_level_strict"] else "0",
                "prompt_level_loose": "1" if scores["prompt_level_loose"] else "0",
                "inst_level_strict": [str(int(value)) for value in strict_values],
                "inst_level_loose": [str(int(value)) for value in loose_values],
            },
            scores={
                "prompt_level_strict_acc": 1.0 if scores["prompt_level_strict"] else 0.0,
                "prompt_level_loose_acc": 1.0 if scores["prompt_level_loose"] else 0.0,
                "inst_level_strict_acc": sum(strict_values) / (len(strict_values) or 1),
                "inst_level_loose_acc": sum(loose_values) / (len(loose_values) or 1),
            },
            metadata={
                "key": str(prepared_sample.doc.get("key", prepared_sample.index)),
                "instruction_count": len(instruction_id_list),
            },
        )

    def evaluate(self, session: InferenceSession) -> TestResult:
        # Recompute metrics so instruction-level accuracy is weighted by total instruction count.
        result = super().evaluate(session)
        samples = list(result.samples)
        if not samples:
            return result

        prompt_level_strict_total = 0.0
        prompt_level_loose_total = 0.0
        strict_correct = 0.0
        strict_total = 0
        loose_correct = 0.0
        loose_total = 0

        for sample in samples:
            prompt_level_strict_total += 1.0 if _to_bool(sample.extracted.get("prompt_level_strict")) else 0.0
            prompt_level_loose_total += 1.0 if _to_bool(sample.extracted.get("prompt_level_loose")) else 0.0
            strict_values = [int(value) for value in sample.extracted.get("inst_level_strict", [])]
            loose_values = [int(value) for value in sample.extracted.get("inst_level_loose", [])]
            strict_correct += sum(strict_values)
            strict_total += len(strict_values)
            loose_correct += sum(loose_values)
            loose_total += len(loose_values)

        result.metrics = {
            "prompt_level_strict_acc": prompt_level_strict_total / len(samples),
            "prompt_level_loose_acc": prompt_level_loose_total / len(samples),
            "inst_level_strict_acc": strict_correct / (strict_total or 1),
            "inst_level_loose_acc": loose_correct / (loose_total or 1),
        }
        return result


def ifeval(**kwargs: Any) -> IFEval:
    return IFEval(**kwargs)
