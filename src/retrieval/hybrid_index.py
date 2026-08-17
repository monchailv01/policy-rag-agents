"""Hybrid retriever: BM25 + dense embeddings fused with Reciprocal Rank Fusion.

Neither half is sufficient on its own for this knowledge base:

* BM25 alone misses paraphrases and cannot bridge languages — a Thai question
  about ``เบี้ยเลี้ยง`` shares no token with the English ``per diem`` section.
* Embeddings alone are unreliable on rare literals such as ``POL-FIN-021`` or
  ``Andaman Travel Services``, which lexical matching nails.

RRF combines the two by rank rather than by score, so no score normalisation is
needed and one ranker cannot dominate the other through raw magnitude.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import Settings, get_settings
from src.retrieval.bm25 import BM25Index
from src.retrieval.chunker import PolicyChunk, load_chunks
from src.retrieval.embedder import Embedder, get_embedder
from src.retrieval.tokenizer import tokenize


@dataclass(frozen=True, slots=True)
class RetrievedSnippet:
    """One policy section, with the full scoring trail that selected it."""

    chunk: PolicyChunk
    fused_score: float
    bm25_score: float
    bm25_rank: int
    dense_score: float
    dense_rank: int
    matched_language: str = ""
    language_swapped: bool = False

    def as_dict(self) -> dict:
        return {
            **self.chunk.as_dict(),
            "fused_score": round(self.fused_score, 5),
            "bm25_score": round(self.bm25_score, 4),
            "bm25_rank": self.bm25_rank,
            "dense_score": round(self.dense_score, 4),
            "dense_rank": self.dense_rank,
            "matched_language": self.matched_language or self.chunk.language,
            "language_swapped": self.language_swapped,
        }


class HybridRetriever:
    """Rank every policy section against a query, then fuse and de-duplicate."""

    def __init__(
        self,
        chunks: Sequence[PolicyChunk],
        embedder: Embedder,
        *,
        rrf_k: int = 60,
        default_top_k: int = 4,
        embeddings: np.ndarray | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.default_top_k = default_top_k

        self._bm25 = BM25Index([tokenize(c.text) for c in self.chunks])
        self._embeddings = (
            embeddings
            if embeddings is not None
            else embedder.encode_documents([c.text for c in self.chunks])
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "HybridRetriever":
        """Build the retriever, reusing cached document embeddings when possible."""
        settings = settings or get_settings()
        chunks = load_chunks(settings.knowledge_base_file)
        embedder = get_embedder(settings)

        cache_file = cls._cache_path(settings, chunks, embedder)
        embeddings = None
        if cache_file.exists():
            try:
                embeddings = np.load(cache_file)["embeddings"]
            except (OSError, KeyError, ValueError):
                embeddings = None  # corrupt cache: fall through and re-encode

        retriever = cls(
            chunks,
            embedder,
            rrf_k=settings.rrf_k,
            default_top_k=settings.retrieval_top_k,
            embeddings=embeddings,
        )
        if embeddings is None:
            np.savez_compressed(cache_file, embeddings=retriever._embeddings)
        return retriever

    @staticmethod
    def _cache_path(
        settings: Settings, chunks: Sequence[PolicyChunk], embedder: Embedder
    ) -> Path:
        """Cache key covers both the corpus content and the embedding model."""
        digest = hashlib.sha256(
            json.dumps([c.text for c in chunks], ensure_ascii=False).encode("utf-8")
            + embedder.name.encode("utf-8")
        ).hexdigest()[:16]
        return settings.cache_dir / f"embeddings-{digest}.npz"

    # -- search -------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        prefer_language: str | None = None,
    ) -> list[RetrievedSnippet]:
        """Return the best policy sections for ``query``, one per policy."""
        top_k = top_k or self.default_top_k

        bm25_scores = self._bm25.score(tokenize(query))
        dense_scores = self._embeddings @ self.embedder.encode_query(query)

        bm25_ranks = _ranks_from_scores(bm25_scores)
        dense_ranks = _ranks_from_scores(dense_scores)
        fused = 1.0 / (self.rrf_k + bm25_ranks + 1) + 1.0 / (self.rrf_k + dense_ranks + 1)

        ordered = np.argsort(-fused, kind="stable")
        by_policy = {c.policy_id: {} for c in self.chunks}
        for index, chunk in enumerate(self.chunks):
            by_policy[chunk.policy_id][chunk.language] = index

        results: list[RetrievedSnippet] = []
        seen: set[str] = set()
        for index in ordered:
            matched = self.chunks[index]
            if matched.policy_id in seen:
                continue
            seen.add(matched.policy_id)

            # The EN and TH editions of a policy are translations of each other,
            # so returning both would only pad the report agent's context with
            # duplicate facts. Keep one, preferring the reader's language.
            #
            # The reported scores always describe the edition that actually won
            # the ranking, so the returned list stays ordered by fused score;
            # ``language_swapped`` flags that a sibling edition is shown instead.
            presented, swapped = matched, False
            if prefer_language and matched.language != prefer_language:
                sibling = by_policy[matched.policy_id].get(prefer_language)
                if sibling is not None:
                    presented, swapped = self.chunks[sibling], True

            results.append(
                RetrievedSnippet(
                    chunk=presented,
                    fused_score=float(fused[index]),
                    bm25_score=float(bm25_scores[index]),
                    bm25_rank=int(bm25_ranks[index]) + 1,
                    dense_score=float(dense_scores[index]),
                    dense_rank=int(dense_ranks[index]) + 1,
                    matched_language=matched.language,
                    language_swapped=swapped,
                )
            )
            if len(results) == top_k:
                break
        return results

    def describe(self) -> dict:
        """Index statistics, surfaced by the web UI and the CLI banner."""
        return {
            "chunks": len(self.chunks),
            "policies": len({c.policy_id for c in self.chunks}),
            "languages": sorted({c.language for c in self.chunks}),
            "embedder": self.embedder.name,
            "dimension": self.embedder.dimension,
            "vocabulary": self._bm25.vocabulary_size,
            "rrf_k": self.rrf_k,
        }


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Convert scores into 0-based ranks (rank 0 = highest score)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks
