# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import Any

from .lambada import LAMBADA

# Mirror the lm-eval multilingual LAMBADA task family over the public OpenAI-translated dataset.
LAMBADA_OPENAI_MT_LANGUAGES = ("de", "en", "es", "fr", "it")
LAMBADA_OPENAI_MT_TASKS = tuple(
    f"lambada_openai_mt_{language}" for language in LAMBADA_OPENAI_MT_LANGUAGES
)


def lambada_openai_mt(language: str, **kwargs: Any) -> LAMBADA:
    if language not in LAMBADA_OPENAI_MT_LANGUAGES:
        raise ValueError(f"unsupported lambada_openai_mt language: {language!r}")
    return LAMBADA(
        dataset_name=language,
        variant_name=f"openai_mt_{language}",
        **kwargs,
    )


def lambada_openai_mt_de(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt("de", **kwargs)


def lambada_openai_mt_en(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt("en", **kwargs)


def lambada_openai_mt_es(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt("es", **kwargs)


def lambada_openai_mt_fr(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt("fr", **kwargs)


def lambada_openai_mt_it(**kwargs: Any) -> LAMBADA:
    return lambada_openai_mt("it", **kwargs)
