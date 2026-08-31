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
def get_llm(temperature: float | None = None, backend: str = "cloud") -> ChatOpenAI:
    """Return a chat model handle, cached per (temperature, backend).

    ``backend="cloud"`` is the configured hosted model; ``backend="local"``
    targets ``LOCAL_LLM_BASE_URL`` — the same graph on local hardware, with a
    longer timeout because a 30B model on one GPU thinks in tens of seconds.
    """
    settings = get_settings()
    resolved = settings.llm_temperature if temperature is None else temperature
    if backend == "local":
        if not settings.local_llm_base_url:
            raise RuntimeError("LOCAL_LLM_BASE_URL is not set; the local backend is disabled.")
        return ChatOpenAI(
            model=settings.local_llm_model,
            temperature=resolved,
            api_key=settings.local_llm_api_key,
            base_url=settings.local_llm_base_url,
            timeout=180,
            max_retries=1,
        )
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=resolved,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=90,
        max_retries=2,
    )
