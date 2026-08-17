"""Embedding backends for the semantic half of the retriever.

Two interchangeable implementations sit behind one protocol:

``local``   sentence-transformers, downloaded once and then run offline.  This
            is the default because it costs nothing per query and moves to a
            local GPU by changing a single environment variable.
``openai``  ``text-embedding-3-small`` over the API, for machines where a
            ~500 MB model download is not wanted.

E5-family models are trained with asymmetric ``query:`` / ``passage:`` prefixes;
omitting them measurably degrades retrieval, so the prefixes are applied here
rather than left to the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from src.config import Settings


@runtime_checkable
class Embedder(Protocol):
    """Minimal interface the retriever depends on."""

    name: str
    dimension: int

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalise rows so that a dot product is the cosine similarity."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class LocalEmbedder:
    """sentence-transformers backend (CPU today, CUDA when a driver allows)."""

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"local:{model_name}@{device}"
        self._is_e5 = "e5" in model_name.lower()
        self._model = SentenceTransformer(model_name, device=device)
        get_dimension = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dimension = int(get_dimension())

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        payload = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
        vectors = self._model.encode(
            payload, batch_size=16, convert_to_numpy=True, show_progress_bar=False
        )
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "passage: " if self._is_e5 else "")

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text], "query: " if self._is_e5 else "")[0]


class OpenAIEmbedder:
    """API backend, used when ``EMBEDDING_BACKEND=openai``."""

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        model = settings.embedding_model
        if model.startswith("intfloat/"):  # a local model name is meaningless here
            model = "text-embedding-3-small"
        self.name = f"openai:{model}"
        self._model = model
        self._client = OpenAI(
            api_key=settings.openai_api_key, base_url=settings.openai_base_url
        )
        self.dimension = 1536

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        return _l2_normalize(vectors)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]


def get_embedder(settings: Settings) -> Embedder:
    """Build the embedder selected by ``EMBEDDING_BACKEND``."""
    if settings.embedding_backend == "openai":
        return OpenAIEmbedder(settings)
    return LocalEmbedder(settings.embedding_model, settings.resolve_device())
