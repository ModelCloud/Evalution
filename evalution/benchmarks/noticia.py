# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import string
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from datasets import load_dataset
from rouge_score import rouge_scorer

from evalution.benchmarks.base import BaseTestSuite
from evalution.benchmarks.execution import PreparedSample
from evalution.engines.base import GenerationOutput, GenerationRequest
from evalution.results import SampleResult

# Mirror the upstream NoticIA one-sentence debunking prompt and stop strings.
_NOTICIA_STOP_STRINGS = ("\n\n", "\n")


def _noticia_prompt(headline: str, body: str) -> str:
    """Implement noticia prompt for this module."""
    return (
        "Ahora eres una Inteligencia Artificial experta en desmontar titulares "
        "sensacionalistas o clickbait. Tu tarea consiste en analizar noticias con "
        "titulares sensacionalistas y generar un resumen de una sola frase que revele "
        "la verdad detrás del titular.\n"
        f"Este es el titular de la noticia: {headline}\n"
        "El titular plantea una pregunta o proporciona información incompleta. "
        "Debes buscar en el cuerpo de la noticia una frase que responda lo que se "
        "sugiere en el título. Siempre que puedas cita el texto original, especialmente "
        "si se trata de una frase que alguien ha dicho. Si citas una frase que alguien "
        "ha dicho, usa comillas para indicar que es una cita. Usa siempre las mínimas "
        "palabras posibles. No es necesario que la respuesta sea una oración completa, "
        "puede ser sólo el foco de la pregunta. Recuerda responder siempre en Español.\n"
        f"Este es el cuerpo de la noticia:\n{body}"
    )


def _clean_noticia_text(text: str) -> str:
    """Implement clean noticia text for this module."""
    cleaned = text.translate(str.maketrans("", "", string.punctuation))
    cleaned = cleaned.replace("\n", " ").strip()
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned.lower()


@lru_cache(maxsize=1)
def _noticia_rouge1_scorer() -> rouge_scorer.RougeScorer:
    # Score the cleaned one-sentence summaries without stemming to match the upstream task intent.
    """Implement noticia rouge1 scorer for this module. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
    return rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)


def _noticia_rouge1(prediction: str, reference: str) -> float:
    """Implement noticia rouge1 for this module."""
    cleaned_prediction = _clean_noticia_text(prediction)
    cleaned_reference = _clean_noticia_text(reference)
    score = _noticia_rouge1_scorer().score(cleaned_reference, cleaned_prediction)
    return float(score["rouge1"].fmeasure)


@dataclass(slots=True)
class Noticia(BaseTestSuite):
    # NoticIA checks whether the model can compress a clickbait headline into the article's true core fact.
    """Implement the noticia benchmark suite."""
    dataset_path: str = "Iker/NoticIA"
    dataset_name: str | None = None
    split: str = "test"
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0

    def dataset_loader(self) -> Any:
        """Return the dataset loader bound to this suite."""
        return load_dataset

    def task_name(self) -> str:
        """Return the exported task name for this suite."""
        return "noticia"

    def result_metadata(
        self,
        *,
        generation_submission_mode: str,
    ) -> dict[str, Any]:
        """Return the result metadata emitted for this suite."""
        return {
            **self.base_result_metadata(generation_submission_mode=generation_submission_mode),
            "scoring_mode": "generated_clickbait_truth_summary",
            "primary_metric": "rouge1",
            "prompt_variant": "headline_body_to_truth_summary",
        }

    def iter_prepared_samples(self, docs: list[dict[str, Any]] | Any) -> Any:
        """Yield prepared samples for the current dataset rows."""
        for index, doc in enumerate(docs):
            headline = str(doc["web_headline"])
            body = str(doc["web_text"])
            target = str(doc["summary"])
            yield PreparedSample(
                index=index,
                doc=doc,
                target=target,
                request=GenerationRequest(
                    prompt=_noticia_prompt(headline, body),
                    stop=list(_NOTICIA_STOP_STRINGS),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                ),
            )

    def score_sample(
        self,
        prepared_sample: PreparedSample,
        output: GenerationOutput,
    ) -> SampleResult:
        """Score one sample against its expected outputs. Keep the scoring path explicit so benchmark-specific behavior stays auditable."""
        cleaned_prediction = _clean_noticia_text(output.text)
        cleaned_reference = _clean_noticia_text(prepared_sample.target)
        return SampleResult(
            index=prepared_sample.index,
            prompt=output.prompt,
            target=prepared_sample.target,
            prediction=output.text,
            extracted={
                "prediction-clean": cleaned_prediction,
                "reference-clean": cleaned_reference,
            },
            scores={
                "rouge1": _noticia_rouge1(output.text, prepared_sample.target),
                "average_len": float(len(cleaned_prediction.split())),
            },
            metadata={
                "web_url": str(prepared_sample.doc["web_url"]),
                "web_headline": str(prepared_sample.doc["web_headline"]),
                "web_text_chars": len(str(prepared_sample.doc["web_text"])),
            },
        )


def noticia(**kwargs: Any) -> Noticia:
    """Implement noticia for this module."""
    return Noticia(**kwargs)
