"""LLM factory.

Every model handle in the project is created here so that switching provider —
OpenAI today, a local vLLM/Ollama server on the workstation GPU tomorrow — is a
change to ``OPENAI_BASE_URL`` and nothing else.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import get_settings


@lru_cache(maxsize=8)
def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """Return a chat model handle, cached per temperature."""
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=90,
        max_retries=2,
    )
