# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from evalution.benchmarks.base import TestSuite
from evalution.benchmarks.data import doc_count, limit_docs, load_suite_dataset
from evalution.engines.base import InferenceSession, RollingLoglikelihoodRequest
from evalution.logbar import get_logger
from evalution.results import SampleResult, TestResult


def _wikitext_detokenizer(doc: dict[str, Any]) -> str:
    string = str(doc["page"])
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def _wikitext_word_count(page: str) -> int:
    return len(re.split(r"\s+", page))


def _wikitext_byte_count(page: str) -> int:
    return len(page.encode("utf-8"))


def _wikitext_preview(text: str, *, limit: int = 160) -> str:
    preview = text.replace("\n", "\\n")
    if len(preview) <= limit:
        return preview
    return f"{preview[:limit]}..."


@dataclass(slots=True)
class WikiText(TestSuite):
    dataset_path: str = "EleutherAI/wikitext_document_level"
    dataset_name: str | None = "wikitext-2-raw-v1"
    split: str = "test"
    max_rows: int | None = None
    batch_size: int | None = None
    cache_dir: str | None = None
    streaming: bool = False

    def dataset_loader(self) -> Any:
        return load_dataset

    def task_name(self) -> str:
        return "wikitext"

    def result_metadata(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "streaming": self.streaming,
            "scoring_mode": "rolling_loglikelihood_perplexity",
            "primary_metric": "word_perplexity",
        }

    def evaluate(self, session: InferenceSession) -> TestResult:
        task_name = self.task_name()
        logger = get_logger()
        loaded_docs, _dataset_load_wall_s = load_suite_dataset(
            self.dataset_loader(),
            task_name=task_name,
            dataset_path=self.dataset_path,
            dataset_name=self.dataset_name,
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
        )

        docs = limit_docs(loaded_docs, self.max_rows)
        if not isinstance(docs, list):
            docs = list(docs)

        total = doc_count(
            docs,
            loaded_docs=loaded_docs,
            max_rows=self.max_rows,
            split=self.split,
        )
        logger.info("%s: evaluating %d sample(s)", task_name, total)

        sample_payloads: list[dict[str, Any]] = []
        requests: list[RollingLoglikelihoodRequest] = []
        for index, doc in enumerate(docs):
            page = str(doc["page"])
            text = _wikitext_detokenizer(doc)
            word_count = _wikitext_word_count(page)
            byte_count = _wikitext_byte_count(page)
            sample_payloads.append(
                {
                    "index": index,
                    "page": page,
                    "text": text,
                    "word_count": word_count,
                    "byte_count": byte_count,
                }
            )
            requests.append(RollingLoglikelihoodRequest(text=text))

        outputs = session.loglikelihood_rolling(requests, batch_size=self.batch_size)
        logger.info("%s: executed %d/%d sample(s)", task_name, len(sample_payloads), total)

        sample_results: list[SampleResult] = []
        total_logprob = 0.0
        total_words = 0
        total_bytes = 0
        for sample_payload, output in zip(sample_payloads, outputs, strict=True):
            word_count = sample_payload["word_count"]
            byte_count = sample_payload["byte_count"]
            total_logprob += output.logprob
            total_words += word_count
            total_bytes += byte_count
            sample_results.append(
                SampleResult(
                    index=sample_payload["index"],
                    prompt="",
                    target="[document]",
                    prediction="[rolling-loglikelihood]",
                    extracted={
                        "token_count": str(output.token_count),
                        "word_count": str(word_count),
                        "byte_count": str(byte_count),
                    },
                    scores={
                        "word_perplexity": math.exp(-(output.logprob / word_count)),
                        "byte_perplexity": math.exp(-(output.logprob / byte_count)),
                        "bits_per_byte": -(output.logprob / byte_count) / math.log(2),
                    },
                    metadata={
                        "page_preview": _wikitext_preview(sample_payload["page"]),
                        "detokenized_preview": _wikitext_preview(sample_payload["text"]),
                        "page_char_count": len(sample_payload["page"]),
                        "logprob": output.logprob,
                        "token_count": output.token_count,
                    },
                )
            )

        metrics = {
            "word_perplexity": math.exp(-(total_logprob / total_words)),
            "byte_perplexity": math.exp(-(total_logprob / total_bytes)),
            "bits_per_byte": -(total_logprob / total_bytes) / math.log(2),
        }
        return TestResult(
            name=task_name,
            metrics=metrics,
            samples=sample_results,
            metadata=self.result_metadata(),
        )


def wikitext(**kwargs: Any) -> WikiText:
    return WikiText(**kwargs)
