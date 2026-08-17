"""A small, dependency-free Okapi BM25 implementation.

The knowledge base is a handful of policy sections, so a dense term-frequency
matrix is both simpler and faster than a sparse index or an external library.
Scoring every document at once with NumPy keeps ``score()`` a single vectorised
expression, which is what the fusion step needs: a full ranking, not a top-k.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class BM25Index:
    """Okapi BM25 over a fixed, pre-tokenised corpus."""

    def __init__(
        self,
        corpus_tokens: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not corpus_tokens:
            raise ValueError("BM25Index requires a non-empty corpus")
        self.k1 = k1
        self.b = b

        vocabulary: dict[str, int] = {}
        for tokens in corpus_tokens:
            for token in tokens:
                vocabulary.setdefault(token, len(vocabulary))
        self._vocabulary = vocabulary

        n_docs, n_terms = len(corpus_tokens), len(vocabulary)
        freqs = np.zeros((n_docs, n_terms), dtype=np.float32)
        for row, tokens in enumerate(corpus_tokens):
            for token in tokens:
                freqs[row, vocabulary[token]] += 1.0
        self._freqs = freqs

        self._doc_lengths = freqs.sum(axis=1)
        self._avg_doc_length = float(self._doc_lengths.mean()) or 1.0

        doc_freq = (freqs > 0).sum(axis=0).astype(np.float32)
        self._idf = np.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        # Denominator term that depends only on the document, not the query.
        self._length_norm = self.k1 * (
            1.0 - self.b + self.b * self._doc_lengths / self._avg_doc_length
        )

    @property
    def n_docs(self) -> int:
        return self._freqs.shape[0]

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)

    def score(self, query_tokens: Sequence[str]) -> np.ndarray:
        """Return a BM25 score for every document in the corpus."""
        columns = [self._vocabulary[t] for t in query_tokens if t in self._vocabulary]
        if not columns:
            return np.zeros(self.n_docs, dtype=np.float32)

        freqs = self._freqs[:, columns]
        numerator = freqs * (self.k1 + 1.0)
        denominator = freqs + self._length_norm[:, None]
        return (self._idf[columns] * numerator / denominator).sum(axis=1)
