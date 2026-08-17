"""Central configuration, loaded from environment variables / ``.env``.

Everything that could reasonably change between environments lives here so the
rest of the codebase never reads ``os.environ`` directly.  The LLM is addressed
through an OpenAI-*compatible* base URL, which means pointing ``OPENAI_BASE_URL``
at a local vLLM or Ollama server is enough to run the whole system offline.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings for the agents, the retriever and the web server."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---------------------------------------------------------------
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-5-mini"
    llm_temperature: float = 0.2

    # --- Embeddings --------------------------------------------------------
    embedding_backend: Literal["local", "openai"] = "local"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "auto"

    # --- Retrieval ---------------------------------------------------------
    knowledge_base_path: Path = Path("knowledge_base.txt")
    retrieval_top_k: int = 4
    rrf_k: int = 60

    # --- Server ------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8100
    chat_db_path: Path = Path("chat_history.db")

    @property
    def knowledge_base_file(self) -> Path:
        """Absolute path to the knowledge base, resolved against the repo root."""
        path = self.knowledge_base_path
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def chat_db_file(self) -> Path:
        path = self.chat_db_path
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def cache_dir(self) -> Path:
        directory = PROJECT_ROOT / ".cache"
        directory.mkdir(exist_ok=True)
        return directory

    def resolve_device(self) -> str:
        """Turn ``EMBEDDING_DEVICE=auto`` into a concrete torch device.

        Keeps the project GPU-ready without forcing a CUDA install: the same
        code runs on CPU today and picks up an RTX 3090 automatically later.
        """
        if self.embedding_device != "auto":
            return self.embedding_device
        try:
            import warnings

            import torch

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover - torch is a hard dependency
            return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
