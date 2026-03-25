# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import Any

from .lambada import LAMBADA

# Keep the task surface aligned to the available StableLM multilingual LAMBADA languages.
LAMBADA_OPENAI_MT_STABLELM_LANGUAGES = ("de", "en", "es", "fr", "it", "nl", "pt")

# Keep task names stable and discovery-friendly for test and model baselines.
LAMBADA_OPENAI_MT_STABLELM_TASKS = tuple(
    f"lambada_openai_mt_stablelm_{language}" for language in LAMBADA_OPENAI_MT_STABLELM_LANGUAGES
)


def lambada_openai_mt_stablelm(language: str, **kwargs: Any) -> LAMBADA:
    # Route each language through the multilingual StableLM-backed LAMBADA dataset.
    if language not in LAMBADA_OPENAI_MT_STABLELM_LANGUAGES:
        raise ValueError(f"unsupported lambada_openai_mt_stablelm language: {language!r}")
    return LAMBADA(
        dataset_path="EleutherAI/lambada_multilingual_stablelm",
        dataset_name=language,
        variant_name=f"openai_mt_stablelm_{language}",
        **kwargs,
    )


def lambada_openai_mt_stablelm_de(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt_stablelm("de", **kwargs)


def lambada_openai_mt_stablelm_en(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt_stablelm("en", **kwargs)


def lambada_openai_mt_stablelm_es(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt_stablelm("es", **kwargs)


def lambada_openai_mt_stablelm_fr(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt_stablelm("fr", **kwargs)


def lambada_openai_mt_stablelm_it(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt_stablelm("it", **kwargs)


def lambada_openai_mt_stablelm_nl(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt_stablelm("nl", **kwargs)


def lambada_openai_mt_stablelm_pt(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt_stablelm("pt", **kwargs)
