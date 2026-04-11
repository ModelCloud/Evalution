# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import json
from pathlib import Path
import random
import string
from typing import Any
from urllib.request import urlopen
import uuid

from datasets import Dataset
import pcre
from transformers import AutoTokenizer

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest, InferenceSession
from evalution.results import SampleResult

# Preserve the upstream public task ids so CLI, YAML, and Python stay aligned with lm-eval.
RULER_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "ruler_vt",
    "ruler_cwe",
    "ruler_fwe",
    "ruler_qa_squad",
    "ruler_qa_hotpot",
)

_DEFAULT_SAMPLE_COUNT = 500
_RULER_RANDOM_SEED = 42
_SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
_HOTPOT_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
_CONTROL_CHARS_RE = pcre.compile(r"[\x00-\x1f]+")
_DOCUMENT_PROMPT = "Document {i}:\n{document}"
_DEFAULT_ESSAY_SENTENCES = (
    "The city archive opens before sunrise and closes after the river traffic slows.",
    "Engineers describe old bridges in careful detail because every repair changes the way the structure carries weight.",
    "A long research memo often hides its most useful observation inside a routine paragraph about earlier experiments.",
    "Writers return to the same anecdote when they want a familiar event to carry a different argument.",
    "The museum catalog notes every small revision so later readers can reconstruct how the exhibit changed over time.",
    "Experienced pilots repeat checklists even on calm mornings because routine prevents avoidable mistakes.",
    "A field notebook is most valuable when it records ordinary observations with the same care as surprising ones.",
    "Teams with clear ownership tend to move faster because each handoff leaves less ambiguity behind.",
    "The committee approved the plan only after the cost estimates were rewritten in plain language.",
    "When the storm passed, the harbor sounded normal again, but every crew remembered the unusual silence beforehand.",
    "Historians compare editions line by line because missing punctuation can signal a larger editorial choice.",
    "The laboratory stored spare components in labeled bins so night shifts could repair instruments without delay.",
    "Travelers judge a station by its timetable, but operators judge it by how quickly congestion clears after noon.",
    "An architect can explain a building in drawings, yet the real lesson often appears in the maintenance log.",
    "Auditors prefer simple evidence trails because complicated stories usually hide unnecessary risk.",
    "A patient explanation written once can save hundreds of repeated clarifications later.",
    "The server room stayed cool, but the incident report focused on the switch that failed before the alarm triggered.",
    "Editors cut elegant paragraphs when those paragraphs distract from the sentence the reader actually needs.",
    "Most experiments succeed only after the team learns which measurements can be ignored safely.",
    "A reliable schedule is built from small buffers that look wasteful until the first unexpected delay arrives.",
)
_FIXED_WORDS = (
    "amber-otter",
    "ardent-maple",
    "brisk-harbor",
    "calm-orchid",
    "cedar-lantern",
    "clear-falcon",
    "copper-meadow",
    "crisp-anchor",
    "daring-valley",
    "echo-canyon",
    "ember-sparrow",
    "fable-garden",
    "frost-mariner",
    "gentle-radar",
    "gloss-river",
    "golden-compass",
    "granite-bridge",
    "harbor-birch",
    "hazel-forge",
    "ivory-circuit",
    "jade-mirror",
    "lunar-thicket",
    "marble-signal",
    "mint-orbit",
    "navy-cedar",
    "opal-station",
    "plaid-comet",
    "quartz-harbor",
    "rapid-orchid",
    "silver-atlas",
    "sunlit-voyage",
    "tidal-raven",
    "velvet-grove",
    "verdant-cinder",
    "violet-beacon",
    "winter-keystone",
)
_NIAH_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
    "{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
)
_VT_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
    "{context}\n"
    "Question: Find all variables that are assigned the value {query} in the text above."
)
_VT_ANSWER_PREFIX = (
    "Answer: According to the chain(s) of variable assignment in the text above, "
    "{num_v} variables are assgined the value {query}, they are:"
)
_CWE_TEMPLATE = (
    "Below is a numbered list of words. In these words, some appear more often than others. "
    "Memorize the ones that appear most often.\n"
    "{context}\n"
    "Question: What are the 10 most common words in the above list?"
)
_CWE_ANSWER_PREFIX = "Answer: The top 10 words that appear most often in the list are:"
_FWE_TEMPLATE = (
    "Read the following coded text and track the frequency of each coded word. "
    "Find the three most frequently appeared coded words. {context}\n"
    "Question: Do not provide any explanation. Please ignore the dots '...'. "
    "What are the three most frequently appeared words in the above coded text?"
)
_FWE_ANSWER_PREFIX = (
    "Answer: According to the coded text above, the three most frequently appeared words are:"
)
_QA_TEMPLATE = (
    "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\n"
    "The following are given documents.\n\n"
    "{context}\n\n"
    "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\n"
    "Question: {query}"
)
_VARIANT_CONFIG: dict[str, dict[str, Any]] = {
    "niah_single_1": {
        "kind": "niah",
        "haystack": "repeat",
        "needle_key_type": "words",
        "needle_value_type": "numbers",
        "num_keys": 1,
        "num_values": 1,
        "num_queries": 1,
        "max_new_tokens": 128,
        "step": 25,
    },
    "niah_single_2": {
        "kind": "niah",
        "haystack": "essay",
        "needle_key_type": "words",
        "needle_value_type": "numbers",
        "num_keys": 1,
        "num_values": 1,
        "num_queries": 1,
        "max_new_tokens": 128,
        "step": 8,
    },
    "niah_single_3": {
        "kind": "niah",
        "haystack": "essay",
        "needle_key_type": "words",
        "needle_value_type": "uuids",
        "num_keys": 1,
        "num_values": 1,
        "num_queries": 1,
        "max_new_tokens": 128,
        "step": 8,
    },
    "niah_multikey_1": {
        "kind": "niah",
        "haystack": "essay",
        "needle_key_type": "words",
        "needle_value_type": "numbers",
        "num_keys": 4,
        "num_values": 1,
        "num_queries": 1,
        "max_new_tokens": 128,
        "step": 8,
    },
    "niah_multikey_2": {
        "kind": "niah",
        "haystack": "needle",
        "needle_key_type": "words",
        "needle_value_type": "numbers",
        "num_keys": 1,
        "num_values": 1,
        "num_queries": 1,
        "max_new_tokens": 128,
        "step": 25,
    },
    "niah_multikey_3": {
        "kind": "niah",
        "haystack": "needle",
        "needle_key_type": "uuids",
        "needle_value_type": "uuids",
        "num_keys": 1,
        "num_values": 1,
        "num_queries": 1,
        "max_new_tokens": 128,
        "step": 25,
    },
    "niah_multiquery": {
        "kind": "niah",
        "haystack": "essay",
        "needle_key_type": "words",
        "needle_value_type": "numbers",
        "num_keys": 4,
        "num_values": 1,
        "num_queries": 4,
        "max_new_tokens": 128,
        "step": 8,
    },
    "niah_multivalue": {
        "kind": "niah",
        "haystack": "essay",
        "needle_key_type": "words",
        "needle_value_type": "numbers",
        "num_keys": 1,
        "num_values": 4,
        "num_queries": 1,
        "max_new_tokens": 128,
        "step": 8,
    },
    "ruler_vt": {
        "kind": "vt",
        "max_new_tokens": 30,
        "step": 10,
    },
    "ruler_cwe": {
        "kind": "cwe",
        "max_new_tokens": 120,
        "step": 10,
    },
    "ruler_fwe": {
        "kind": "fwe",
        "max_new_tokens": 50,
        "step": 50,
    },
    "ruler_qa_squad": {
        "kind": "qa",
        "dataset": "squad",
        "max_new_tokens": 32,
        "step": 1,
    },
    "ruler_qa_hotpot": {
        "kind": "qa",
        "dataset": "hotpot",
        "max_new_tokens": 32,
        "step": 1,
    },
}


def _normalize_prediction(text: str) -> str:
    return _CONTROL_CHARS_RE.sub("\n", text).strip()


def _contains_fraction(prediction: str, outputs: list[str]) -> float:
    if not outputs:
        return 0.0
    lowered_prediction = prediction.lower()
    return sum(1.0 if output.lower() in lowered_prediction else 0.0 for output in outputs) / len(outputs)


def _compose_prompt(input_text: str, gen_prefix: str) -> str:
    stripped_input = input_text.rstrip()
    stripped_prefix = gen_prefix.strip()
    if stripped_input.endswith(stripped_prefix):
        return stripped_input
    return f"{stripped_input} {stripped_prefix}".strip()


def _token_length(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else getattr(encoded, "input_ids")
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _ruler_cache_root(cache_dir: str | None) -> Path:
    if cache_dir:
        return Path(cache_dir) / "ruler"
    return Path.home() / ".cache" / "evalution" / "ruler"


def _download_json(url: str, *, cache_dir: str | None, file_name: str) -> Any:
    cache_root = _ruler_cache_root(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    target_path = cache_root / file_name
    if not target_path.exists():
        with urlopen(url) as response:
            payload = json.loads(response.read().decode("utf-8"))
        target_path.write_text(json.dumps(payload), encoding="utf-8")
    return json.loads(target_path.read_text(encoding="utf-8"))


def _random_value(kind: str, rng: random.Random) -> str:
    if kind == "numbers":
        return str(rng.randint(1_000_000, 9_999_999))
    if kind == "words":
        return rng.choice(_FIXED_WORDS)
    if kind == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    raise ValueError(f"unsupported ruler token type: {kind!r}")


def _sample_unique_values(kind: str, count: int, rng: random.Random) -> list[str]:
    values: list[str] = []
    while len(values) < count:
        candidate = _random_value(kind, rng)
        if candidate not in values:
            values.append(candidate)
    return values


def _render_niah_context(
    unit_count: int,
    *,
    haystack_kind: str,
    needles: list[str],
    needle_key_type: str,
    needle_value_type: str,
    rng: random.Random,
) -> str:
    if haystack_kind == "essay":
        sentences = [_DEFAULT_ESSAY_SENTENCES[index % len(_DEFAULT_ESSAY_SENTENCES)] for index in range(unit_count)]
        insert_positions = sorted(rng.sample(range(len(sentences) + 1), k=len(needles)))
        offset = 0
        for position, needle in zip(insert_positions, needles, strict=True):
            sentences.insert(position + offset, needle)
            offset += 1
        return " ".join(sentences)
    if haystack_kind == "repeat":
        repeated = [
            "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
            for _ in range(unit_count)
        ]
        insert_positions = sorted(rng.sample(range(len(repeated) + 1), k=len(needles)))
        offset = 0
        for position, needle in zip(insert_positions, needles, strict=True):
            repeated.insert(position + offset, needle)
            offset += 1
        return "\n".join(repeated)
    if haystack_kind == "needle":
        distractors = [
            (
                "One of the special magic "
                f"{needle_value_type} for {_random_value(needle_key_type, rng)} is: "
                f"{_random_value(needle_value_type, rng)}."
            )
            for _ in range(unit_count)
        ]
        insert_positions = sorted(rng.sample(range(len(distractors) + 1), k=len(needles)))
        offset = 0
        for position, needle in zip(insert_positions, needles, strict=True):
            distractors.insert(position + offset, needle)
            offset += 1
        return "\n".join(distractors)
    raise ValueError(f"unsupported ruler haystack kind: {haystack_kind!r}")


def _build_niah_row(sample_index: int, unit_count: int, config: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(_RULER_RANDOM_SEED + sample_index)
    num_keys = max(int(config["num_keys"]), int(config["num_queries"]))
    keys = _sample_unique_values(str(config["needle_key_type"]), num_keys, rng)
    values_by_key = [
        _sample_unique_values(str(config["needle_value_type"]), int(config["num_values"]), rng)
        for _ in keys
    ]
    needles = [
        f"One of the special magic {config['needle_value_type']} for {key} is: {value}."
        for key, values in zip(keys, values_by_key, strict=True)
        for value in values
    ]
    rng.shuffle(needles)
    context = _render_niah_context(
        unit_count,
        haystack_kind=str(config["haystack"]),
        needles=needles,
        needle_key_type=str(config["needle_key_type"]),
        needle_value_type=str(config["needle_value_type"]),
        rng=rng,
    )
    query_indices = sorted(rng.sample(range(len(keys)), k=int(config["num_queries"])))
    queries = [keys[index] for index in query_indices]
    outputs = [value for index in query_indices for value in values_by_key[index]]
    query = ", ".join(queries[:-1]) + f", and {queries[-1]}" if len(queries) > 1 else queries[0]
    template = _NIAH_TEMPLATE
    value_label = str(config["needle_value_type"])
    if len(outputs) == 1:
        template = template.replace("Some", "A")
        template = template.replace("are all", "is")
        template = template.replace(" are ", " is ")
        value_label = value_label[:-1]
    input_text = template.format(
        type_needle_v=value_label,
        context=context,
        query=query,
    )
    gen_prefix = (
        f"The special magic {value_label} for {query} mentioned in the provided text is"
        if len(outputs) == 1
        else f"The special magic {value_label} for {query} mentioned in the provided text are"
    )
    return {
        "index": sample_index,
        "input": input_text,
        "outputs": outputs,
        "gen_prefix": gen_prefix,
    }


def _generate_variable_chain(rng: random.Random, *, num_hops: int = 4) -> tuple[list[str], list[str], str]:
    variables = _sample_unique_values("uuids", num_hops + 1, rng)
    variables = [variable.replace("-", "")[:5].upper() for variable in variables]
    value = str(rng.randint(10_000, 99_999))
    chain = [f"VAR {variables[0]} = {value}"]
    for index in range(num_hops):
        chain.append(f"VAR {variables[index + 1]} = VAR {variables[index]}")
    return variables, chain, value


def _build_vt_row(sample_index: int, unit_count: int) -> dict[str, Any]:
    rng = random.Random(_RULER_RANDOM_SEED + sample_index)
    variables, chain, value = _generate_variable_chain(rng)
    noise_sentences = [
        "The grass is green.",
        "The sky is blue.",
        "The sun is yellow.",
        "Here we go.",
        "There and back again.",
    ]
    context_sentences = [noise_sentences[index % len(noise_sentences)] for index in range(unit_count)]
    insert_positions = sorted(rng.sample(range(len(context_sentences) + 1), k=len(chain)))
    offset = 0
    for position, chain_step in zip(insert_positions, chain, strict=True):
        context_sentences.insert(position + offset, chain_step)
        offset += 1
    context = " ".join(context_sentences)
    input_text = _VT_TEMPLATE.format(context=context, query=value)
    gen_prefix = _VT_ANSWER_PREFIX.format(num_v=len(variables), query=value)
    return {
        "index": sample_index,
        "input": input_text,
        "outputs": variables,
        "gen_prefix": gen_prefix,
    }


def _build_cwe_row(sample_index: int, unit_count: int) -> dict[str, Any]:
    rng = random.Random(_RULER_RANDOM_SEED + sample_index)
    selected_words = rng.sample(list(_FIXED_WORDS), min(unit_count, len(_FIXED_WORDS)))
    common = selected_words[:10]
    uncommon = selected_words[10:]
    common_repeat = 30 if unit_count >= 20 else 6
    uncommon_repeat = 3 if unit_count >= 20 else 1
    repeated_words = common * common_repeat + uncommon * uncommon_repeat
    rng.shuffle(repeated_words)
    context = " ".join(f"{index + 1}. {word}" for index, word in enumerate(repeated_words))
    return {
        "index": sample_index,
        "input": _CWE_TEMPLATE.format(context=context),
        "outputs": common,
        "gen_prefix": _CWE_ANSWER_PREFIX,
    }


def _random_coded_word(rng: random.Random, *, length: int = 6) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def _build_fwe_row(sample_index: int, unit_count: int) -> dict[str, Any]:
    rng = random.Random(_RULER_RANDOM_SEED + sample_index)
    outputs = [_random_coded_word(rng) for _ in range(3)]
    distractors: list[str] = []
    while len(distractors) < max(10, unit_count // 8):
        candidate = _random_coded_word(rng)
        if candidate not in outputs and candidate not in distractors:
            distractors.append(candidate)
    coded_words = ["..."] * max(1, unit_count // 20)
    coded_words.extend([outputs[0]] * max(4, unit_count // 3))
    coded_words.extend([outputs[1]] * max(3, unit_count // 4))
    coded_words.extend([outputs[2]] * max(2, unit_count // 5))
    for distractor in distractors:
        coded_words.extend([distractor] * 2)
    rng.shuffle(coded_words)
    return {
        "index": sample_index,
        "input": _FWE_TEMPLATE.format(context=" ".join(coded_words)),
        "outputs": outputs,
        "gen_prefix": _FWE_ANSWER_PREFIX,
    }


def _squad_material(cache_dir: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    payload = _download_json(_SQUAD_DEV_URL, cache_dir=cache_dir, file_name="squad-dev-v2.0.json")
    docs = sorted(
        {
            str(paragraph["context"]).strip()
            for article in payload["data"]
            for paragraph in article["paragraphs"]
        }
    )
    doc_to_index = {doc: index for index, doc in enumerate(docs)}
    qas: list[dict[str, Any]] = []
    for article in payload["data"]:
        more_docs = [doc_to_index[str(paragraph["context"]).strip()] for paragraph in article["paragraphs"]]
        for paragraph in article["paragraphs"]:
            context_index = doc_to_index[str(paragraph["context"]).strip()]
            for qa in paragraph["qas"]:
                if qa["is_impossible"]:
                    continue
                qas.append(
                    {
                        "query": str(qa["question"]).strip(),
                        "outputs": [str(answer["text"]).strip() for answer in qa["answers"] if str(answer["text"]).strip()],
                        "context": [context_index],
                        "more_context": [index for index in more_docs if index != context_index],
                    }
                )
    return qas, docs


def _hotpot_material(cache_dir: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    payload = _download_json(_HOTPOT_DEV_URL, cache_dir=cache_dir, file_name="hotpot-dev-distractor-v1.json")
    docs = sorted(
        {
            f"{title}\n{''.join(sentences)}"
            for item in payload
            for title, sentences in item["context"]
        }
    )
    doc_to_index = {doc: index for index, doc in enumerate(docs)}
    qas: list[dict[str, Any]] = []
    for item in payload:
        qas.append(
            {
                "query": str(item["question"]).strip(),
                "outputs": [str(item["answer"]).strip()],
                "context": [doc_to_index[f"{title}\n{''.join(sentences)}"] for title, sentences in item["context"]],
                "more_context": [],
            }
        )
    return qas, docs


def _qa_material(dataset: str, cache_dir: str | None) -> tuple[list[dict[str, Any]], list[str]]:
    if dataset == "squad":
        return _squad_material(cache_dir)
    if dataset == "hotpot":
        return _hotpot_material(cache_dir)
    raise ValueError(f"unsupported ruler qa dataset: {dataset!r}")


def _build_qa_row(sample_index: int, num_docs: int, *, dataset: str, cache_dir: str | None) -> dict[str, Any]:
    rng = random.Random(_RULER_RANDOM_SEED + sample_index)
    qas, docs = _qa_material(dataset, cache_dir)
    qa = qas[sample_index % len(qas)]
    selected_doc_indices = list(qa["context"])
    for index in qa.get("more_context", []):
        if len(selected_doc_indices) >= num_docs:
            break
        if index not in selected_doc_indices:
            selected_doc_indices.append(index)
    remaining_pool = [index for index in range(len(docs)) if index not in selected_doc_indices]
    while len(selected_doc_indices) < num_docs and remaining_pool:
        choice = remaining_pool.pop(rng.randrange(len(remaining_pool)))
        selected_doc_indices.append(choice)
    rendered_docs = [docs[index] for index in selected_doc_indices]
    rng.shuffle(rendered_docs)
    context = "\n\n".join(
        _DOCUMENT_PROMPT.format(i=index + 1, document=document)
        for index, document in enumerate(rendered_docs)
    )
    return {
        "index": sample_index,
        "input": _QA_TEMPLATE.format(context=context, query=qa["query"]),
        "outputs": list(qa["outputs"]),
        "gen_prefix": "Answer:",
    }


def _fit_units(
    builder: Any,
    *,
    step: int,
    max_length: int,
    max_new_tokens: int,
    tokenizer: Any,
    minimum_units: int = 1,
    maximum_units: int | None = None,
) -> int:
    best_units = minimum_units
    units = minimum_units
    while True:
        row = builder(units)
        prompt = _compose_prompt(str(row["input"]), str(row["gen_prefix"]))
        answer_text = " ".join(str(output) for output in row["outputs"])
        total_length = _token_length(tokenizer, f"{prompt} {answer_text}") + max_new_tokens
        if total_length > max_length:
            return best_units
        best_units = units
        units += step
        if maximum_units is not None and units > maximum_units:
            return best_units


def _generate_rows(
    variant: str,
    *,
    tokenizer: Any,
    max_length: int,
    sample_count: int,
    cache_dir: str | None,
) -> list[dict[str, Any]]:
    config = _VARIANT_CONFIG[variant]
    kind = str(config["kind"])
    max_new_tokens = int(config["max_new_tokens"])
    step = int(config["step"])
    if kind == "niah":
        fitted_units = _fit_units(
            lambda unit_count: _build_niah_row(0, unit_count, config),
            step=step,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
            minimum_units=step,
        )
        rows = [_build_niah_row(sample_index, fitted_units, config) for sample_index in range(sample_count)]
    elif kind == "vt":
        fitted_units = _fit_units(
            lambda unit_count: _build_vt_row(0, unit_count),
            step=step,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
            minimum_units=step,
        )
        rows = [_build_vt_row(sample_index, fitted_units) for sample_index in range(sample_count)]
    elif kind == "cwe":
        fitted_units = _fit_units(
            lambda unit_count: _build_cwe_row(0, unit_count),
            step=step,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
            minimum_units=10,
            maximum_units=len(_FIXED_WORDS),
        )
        rows = [_build_cwe_row(sample_index, fitted_units) for sample_index in range(sample_count)]
    elif kind == "fwe":
        fitted_units = _fit_units(
            lambda unit_count: _build_fwe_row(0, unit_count),
            step=step,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
            minimum_units=50,
        )
        rows = [_build_fwe_row(sample_index, fitted_units) for sample_index in range(sample_count)]
    elif kind == "qa":
        dataset = str(config["dataset"])
        fitted_docs = _fit_units(
            lambda doc_count: _build_qa_row(0, doc_count, dataset=dataset, cache_dir=cache_dir),
            step=step,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            tokenizer=tokenizer,
            minimum_units=1,
        )
        rows = [
            _build_qa_row(sample_index, fitted_docs, dataset=dataset, cache_dir=cache_dir)
            for sample_index in range(sample_count)
        ]
    else:
        raise ValueError(f"unsupported ruler variant kind: {kind!r}")
    return [
        {
            **row,
            "length": _token_length(tokenizer, _compose_prompt(str(row["input"]), str(row["gen_prefix"])))
            + max_new_tokens,
            "max_length": max_length,
        }
        for row in rows
    ]


def _load_ruler_dataset(
    dataset_path: str,
    dataset_name: str | None,
    *,
    split: str,
    cache_dir: str | None = None,
    stream: bool = False,
    variant: str,
    tokenizer: Any,
    max_length: int,
    sample_count: int,
) -> Dataset:
    del dataset_name
    if dataset_path != "NVIDIA/RULER":
        raise ValueError(f"unsupported ruler dataset path: {dataset_path!r}")
    if split != "test":
        raise ValueError(f"unsupported ruler split: {split!r}")
    if stream:
        raise ValueError("ruler does not support stream=True")
    return Dataset.from_list(
        _generate_rows(
            variant,
            tokenizer=tokenizer,
            max_length=max_length,
            sample_count=sample_count,
            cache_dir=cache_dir,
        )
    )


def _session_tokenizer(session: InferenceSession) -> Any | None:
    return getattr(session, "prepare_tokenizer", None) or getattr(session, "tokenizer", None)


@dataclass(slots=True)
class RULER(BaseTestSuite):
    # Generate deterministic tokenizer-sized synthetic RULER samples for one upstream task id.
    dataset_path: str = "NVIDIA/RULER"
    dataset_name: str | None = "niah_single_1"
    split: str = "test"
    stream: bool = False
    variant: str = "niah_single_1"
    max_length: int = 4096
    num_samples: int = _DEFAULT_SAMPLE_COUNT
    tokenizer_path: str | None = None
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict, repr=False)
    _generation_tokenizer: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.variant not in RULER_TASKS:
            raise ValueError(f"unsupported ruler variant: {self.variant!r}")
        if self.dataset_name in {None, self.variant}:
            self.dataset_name = self.variant
        else:
            raise ValueError("ruler dataset_name must match the configured variant")

    def _effective_sample_count(self) -> int:
        if self.max_rows is None:
            return self.num_samples
        return min(self.num_samples, self.max_rows)

    def _resolve_generation_tokenizer(self, session: InferenceSession | None = None) -> Any:
        if self._generation_tokenizer is not None:
            return self._generation_tokenizer
        tokenizer = _session_tokenizer(session) if session is not None else None
        if tokenizer is None and self.tokenizer_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
                **self.tokenizer_kwargs,
            )
        if tokenizer is None:
            raise ValueError(
                "ruler requires a tokenizer for synthetic length fitting; "
                "use an engine session exposing `tokenizer` or set `tokenizer_path`"
            )
        self._generation_tokenizer = tokenizer
        return tokenizer

    def dataset_loader(self) -> Any:
        return partial(
            _load_ruler_dataset,
            variant=self.variant,
            tokenizer=self._resolve_generation_tokenizer(),
            max_length=self.max_length,
            sample_count=self._effective_sample_count(),
        )

    def task_name(self) -> str:
        return self.variant

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "variant": self.variant,
            "max_length": self.max_length,
            "scoring_mode": "generated_contains_fraction",
            "primary_metric": "contains_fraction",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        max_new_tokens = int(_VARIANT_CONFIG[self.variant]["max_new_tokens"])
        for index, doc in enumerate(docs):
            outputs = [str(output).strip() for output in doc["outputs"]]
            yield PreparedSample(
                index=index,
                doc=doc,
                target=" | ".join(outputs),
                request=GenerationRequest(
                    prompt=_compose_prompt(str(doc["input"]), str(doc["gen_prefix"])),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        normalized_prediction = _normalize_prediction(output.text)
        outputs = [str(item).strip() for item in prepared_sample.doc["outputs"]]
        score = _contains_fraction(normalized_prediction, outputs)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-normalized": normalized_prediction,
                "outputs": outputs,
                "matched_outputs": [output_item for output_item in outputs if output_item.lower() in normalized_prediction.lower()],
            },
            scores={"contains_fraction": score},
            metadata={
                "variant": self.variant,
                "gen_prefix": str(prepared_sample.doc["gen_prefix"]),
                "max_length": int(prepared_sample.doc["max_length"]),
                "length": int(prepared_sample.doc["length"]),
            },
        )

    def evaluate(self, session: InferenceSession) -> Any:
        self._resolve_generation_tokenizer(session)
        return super().evaluate(session)


def ruler(*, variant: str = "niah_single_1", **kwargs: Any) -> RULER:
    kwargs.setdefault("dataset_name", variant)
    return RULER(variant=variant, **kwargs)


def _make_ruler_factory(variant: str) -> Any:
    def factory(**kwargs: Any) -> RULER:
        return ruler(variant=variant, **kwargs)

    factory.__name__ = variant
    return factory


for _variant in RULER_TASKS:
    globals()[_variant] = _make_ruler_factory(_variant)

del _variant
