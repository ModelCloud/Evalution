# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Adapted clean-room from the Google Research ROUGE reference implementation:
# https://github.com/google-research/google-research/tree/master/rouge

from __future__ import annotations

from collections import Counter
from typing import NamedTuple, Protocol

import pcre

# Keep ROUGE tokenization explicit and local so benchmark scoring stays dependency-free.
_NON_ALPHANUM_RE = pcre.compile(r"[^a-z0-9]+")
_SPACES_RE = pcre.compile(r"\s+")
_VALID_TOKEN_RE = pcre.compile(r"^[a-z0-9]+$")
_SENTENCE_SPLIT_RE = pcre.compile(r"(?<=[.!?])\s+")


class Score(NamedTuple):
    """Represent one ROUGE score triple."""

    precision: float
    recall: float
    fmeasure: float


class _Tokenizer(Protocol):
    """Define the minimal tokenizer interface accepted by RougeScorer."""

    def tokenize(self, text: str) -> list[str]:
        """Tokenize input text into normalized ROUGE tokens."""


class _PorterStemmer:
    """Implement a small Porter stemmer compatible with the ROUGE token pipeline."""

    def stem(self, word: str) -> str:
        """Return the Porter stem for one lowercase token."""

        if len(word) <= 2:
            return word

        stemmed = self._step1a(word)
        stemmed = self._step1b(stemmed)
        stemmed = self._step1c(stemmed)
        stemmed = self._step2(stemmed)
        stemmed = self._step3(stemmed)
        stemmed = self._step4(stemmed)
        stemmed = self._step5a(stemmed)
        stemmed = self._step5b(stemmed)
        return stemmed

    def _is_consonant(self, word: str, index: int) -> bool:
        character = word[index]
        if character in "aeiou":
            return False
        if character == "y":
            return index == 0 or not self._is_consonant(word, index - 1)
        return True

    def _measure(self, stem: str) -> int:
        if not stem:
            return 0

        measure = 0
        in_vowel_run = False
        for index in range(len(stem)):
            is_vowel = not self._is_consonant(stem, index)
            if is_vowel:
                in_vowel_run = True
                continue
            if in_vowel_run:
                measure += 1
                in_vowel_run = False
        return measure

    def _contains_vowel(self, stem: str) -> bool:
        return any(not self._is_consonant(stem, index) for index in range(len(stem)))

    def _ends_with_double_consonant(self, word: str) -> bool:
        return (
            len(word) >= 2
            and word[-1] == word[-2]
            and self._is_consonant(word, len(word) - 1)
        )

    def _is_cvc(self, word: str) -> bool:
        return (
            len(word) >= 3
            and self._is_consonant(word, len(word) - 1)
            and not self._is_consonant(word, len(word) - 2)
            and self._is_consonant(word, len(word) - 3)
            and word[-1] not in {"w", "x", "y"}
        )

    def _step1a(self, word: str) -> str:
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step1b(self, word: str) -> str:
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                return stem + "ee"
            return word

        for suffix in ("ed", "ing"):
            if not word.endswith(suffix):
                continue
            stem = word[: -len(suffix)]
            if not self._contains_vowel(stem):
                return word
            return self._step1b_adjustments(stem)
        return word

    def _step1b_adjustments(self, word: str) -> str:
        if word.endswith(("at", "bl", "iz")):
            return word + "e"
        if self._ends_with_double_consonant(word) and word[-1] not in {"l", "s", "z"}:
            return word[:-1]
        if self._measure(word) == 1 and self._is_cvc(word):
            return word + "e"
        return word

    def _step1c(self, word: str) -> str:
        if len(word) > 1 and word.endswith("y") and self._contains_vowel(word[:-1]):
            return word[:-1] + "i"
        return word

    def _step2(self, word: str) -> str:
        replacements = (
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("bli", "ble"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
            ("logi", "log"),
        )
        for suffix, replacement in replacements:
            if not word.endswith(suffix):
                continue
            stem = word[: -len(suffix)]
            if self._measure(stem) > 0:
                return stem + replacement
            return word
        return word

    def _step3(self, word: str) -> str:
        replacements = (
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        )
        for suffix, replacement in replacements:
            if not word.endswith(suffix):
                continue
            stem = word[: -len(suffix)]
            if self._measure(stem) > 0:
                return stem + replacement
            return word
        return word

    def _step4(self, word: str) -> str:
        suffixes = (
            "al",
            "ance",
            "ence",
            "er",
            "ic",
            "able",
            "ible",
            "ant",
            "ement",
            "ment",
            "ent",
            "ou",
            "ism",
            "ate",
            "iti",
            "ous",
            "ive",
            "ize",
        )
        for suffix in suffixes:
            if not word.endswith(suffix):
                continue
            stem = word[: -len(suffix)]
            if self._measure(stem) > 1:
                return stem
            return word

        if word.endswith("ion"):
            stem = word[:-3]
            if self._measure(stem) > 1 and stem.endswith(("s", "t")):
                return stem
        return word

    def _step5a(self, word: str) -> str:
        if not word.endswith("e"):
            return word
        stem = word[:-1]
        measure = self._measure(stem)
        if measure > 1:
            return stem
        if measure == 1 and not self._is_cvc(stem):
            return stem
        return word

    def _step5b(self, word: str) -> str:
        if self._measure(word) > 1 and self._ends_with_double_consonant(word) and word.endswith("l"):
            return word[:-1]
        return word


class _DefaultTokenizer:
    """Tokenize ROUGE inputs the same way the reference implementation expects."""

    def __init__(self, use_stemmer: bool = False) -> None:
        """Initialize the tokenizer."""

        self._stemmer = _PorterStemmer() if use_stemmer else None

    def tokenize(self, text: str) -> list[str]:
        """Normalize input text into lowercase alphanumeric ROUGE tokens."""

        normalized = _NON_ALPHANUM_RE.sub(" ", str(text).lower())
        tokens = _SPACES_RE.split(normalized)
        if self._stemmer is not None:
            tokens = [self._stemmer.stem(token) if len(token) > 3 else token for token in tokens]
        return [token for token in tokens if _VALID_TOKEN_RE.fullmatch(token)]


class RougeScorer:
    """Calculate ROUGE-N, ROUGE-L, and ROUGE-Lsum scores between two text blobs."""

    def __init__(
        self,
        rouge_types: list[str] | tuple[str, ...],
        use_stemmer: bool = False,
        split_summaries: bool = False,
        tokenizer: _Tokenizer | None = None,
    ) -> None:
        """Initialize the scorer."""

        self.rouge_types = tuple(rouge_types)
        self._tokenizer = tokenizer if tokenizer is not None else _DefaultTokenizer(use_stemmer=use_stemmer)
        self._split_summaries = split_summaries

    def score_multi(self, targets: list[str] | tuple[str, ...], prediction: str) -> dict[str, Score]:
        """Return the best score for each ROUGE type across multiple references."""

        score_dicts = [self.score(target, prediction) for target in targets]
        return {
            rouge_type: max(score_dicts, key=lambda score_dict: score_dict[rouge_type].fmeasure)[rouge_type]
            for rouge_type in self.rouge_types
        }

    def score(self, target: str, prediction: str) -> dict[str, Score]:
        """Calculate ROUGE scores between one target and one prediction."""

        target_text = str(target)
        prediction_text = str(prediction)
        if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
            target_tokens = None
            prediction_tokens = None
        else:
            target_tokens = self._tokenizer.tokenize(target_text)
            prediction_tokens = self._tokenizer.tokenize(prediction_text)

        scores: dict[str, Score] = {}
        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                scores[rouge_type] = _score_lcs(target_tokens or [], prediction_tokens or [])
                continue
            if rouge_type == "rougeLsum":
                target_sentences = [self._tokenizer.tokenize(sentence) for sentence in _split_sentences(target_text, self._split_summaries)]
                prediction_sentences = [
                    self._tokenizer.tokenize(sentence)
                    for sentence in _split_sentences(prediction_text, self._split_summaries)
                ]
                scores[rouge_type] = _score_summary_lcs(target_sentences, prediction_sentences)
                continue
            if rouge_type.startswith("rouge") and rouge_type[5:].isdigit():
                ngram_size = int(rouge_type[5:])
                if ngram_size <= 0:
                    raise ValueError(f"rougen requires positive n: {rouge_type}")
                scores[rouge_type] = _score_ngrams(
                    _create_ngrams(target_tokens or [], ngram_size),
                    _create_ngrams(prediction_tokens or [], ngram_size),
                )
                continue
            raise ValueError(f"Invalid rouge type: {rouge_type}")
        return scores


def _split_sentences(text: str, split_summaries: bool) -> list[str]:
    """Split summary text into sentence-like chunks for ROUGE-Lsum."""

    raw_sentences = _SENTENCE_SPLIT_RE.split(str(text)) if split_summaries else str(text).split("\n")
    return [sentence for sentence in raw_sentences if sentence]


def _create_ngrams(tokens: list[str], ngram_size: int) -> Counter[tuple[str, ...]]:
    """Create n-gram counts for one token sequence."""

    return Counter(
        tuple(tokens[index : index + ngram_size])
        for index in range(len(tokens) - ngram_size + 1)
    )


def _score_ngrams(target_ngrams: Counter[tuple[str, ...]], prediction_ngrams: Counter[tuple[str, ...]]) -> Score:
    """Score ROUGE-N overlap."""

    overlap = 0
    for ngram, target_count in target_ngrams.items():
        overlap += min(target_count, prediction_ngrams.get(ngram, 0))
    target_total = sum(target_ngrams.values())
    prediction_total = sum(prediction_ngrams.values())
    precision = overlap / max(prediction_total, 1)
    recall = overlap / max(target_total, 1)
    return Score(precision=precision, recall=recall, fmeasure=_fmeasure(precision, recall))


def _score_lcs(target_tokens: list[str], prediction_tokens: list[str]) -> Score:
    """Score ROUGE-L using the longest common subsequence."""

    if not target_tokens or not prediction_tokens:
        return Score(precision=0.0, recall=0.0, fmeasure=0.0)

    lcs_length = _lcs_length(target_tokens, prediction_tokens)
    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(target_tokens)
    return Score(precision=precision, recall=recall, fmeasure=_fmeasure(precision, recall))


def _score_summary_lcs(reference_sentences: list[list[str]], candidate_sentences: list[list[str]]) -> Score:
    """Score ROUGE-Lsum using union-LCS over reference and candidate sentences."""

    if not reference_sentences or not candidate_sentences:
        return Score(precision=0.0, recall=0.0, fmeasure=0.0)

    reference_token_total = sum(len(sentence) for sentence in reference_sentences)
    candidate_token_total = sum(len(sentence) for sentence in candidate_sentences)
    if reference_token_total == 0 or candidate_token_total == 0:
        return Score(precision=0.0, recall=0.0, fmeasure=0.0)

    remaining_reference = Counter(token for sentence in reference_sentences for token in sentence)
    remaining_candidate = Counter(token for sentence in candidate_sentences for token in sentence)

    hits = 0
    for reference_sentence in reference_sentences:
        union_tokens = _union_lcs_tokens(reference_sentence, candidate_sentences)
        for token in union_tokens:
            if remaining_reference[token] <= 0 or remaining_candidate[token] <= 0:
                continue
            remaining_reference[token] -= 1
            remaining_candidate[token] -= 1
            hits += 1

    precision = hits / candidate_token_total
    recall = hits / reference_token_total
    return Score(precision=precision, recall=recall, fmeasure=_fmeasure(precision, recall))


def _union_lcs_tokens(reference_sentence: list[str], candidate_sentences: list[list[str]]) -> list[str]:
    """Return the union of one reference sentence's LCS tokens over all candidate sentences."""

    lcs_indices = [_backtrack_lcs_indices(reference_sentence, candidate_sentence) for candidate_sentence in candidate_sentences]
    union_indices = sorted(set().union(*lcs_indices))
    return [reference_sentence[index] for index in union_indices]


def _lcs_length(left: list[str], right: list[str]) -> int:
    """Return the LCS length for two token sequences."""

    return _lcs_table(left, right)[-1][-1]


def _lcs_table(left: list[str], right: list[str]) -> list[list[int]]:
    """Build a dynamic-programming table for the longest common subsequence."""

    rows = len(left)
    cols = len(right)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for left_index in range(1, rows + 1):
        for right_index in range(1, cols + 1):
            if left[left_index - 1] == right[right_index - 1]:
                table[left_index][right_index] = table[left_index - 1][right_index - 1] + 1
            else:
                table[left_index][right_index] = max(table[left_index - 1][right_index], table[left_index][right_index - 1])
    return table


def _backtrack_lcs_indices(reference: list[str], candidate: list[str]) -> list[int]:
    """Recover one LCS path as indices into the reference sentence."""

    table = _lcs_table(reference, candidate)
    left_index = len(reference)
    right_index = len(candidate)
    indices: list[int] = []
    while left_index > 0 and right_index > 0:
        if reference[left_index - 1] == candidate[right_index - 1]:
            indices.insert(0, left_index - 1)
            left_index -= 1
            right_index -= 1
            continue
        if table[left_index][right_index - 1] > table[left_index - 1][right_index]:
            right_index -= 1
        else:
            left_index -= 1
    return indices


def _fmeasure(precision: float, recall: float) -> float:
    """Compute the harmonic mean of precision and recall."""

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


__all__ = ["RougeScorer", "Score"]
