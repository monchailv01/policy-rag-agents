"""Tests for the retrieval layer.

Everything here is deterministic and offline: the chunker, the tokeniser, BM25
and the fusion logic are exercised without an LLM or an embedding download, so
the suite runs in under a second and needs no API key.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import get_settings
from src.retrieval.bm25 import BM25Index
from src.retrieval.chunker import load_chunks
from src.retrieval.hybrid_index import HybridRetriever, _ranks_from_scores
from src.retrieval.tokenizer import tokenize
from src.utils import detect_language


class StubEmbedder:
    """Deterministic bag-of-tokens embedder, so tests never hit the network."""

    name = "stub"
    dimension = 32

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            vector[hash(token) % self.dimension] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode_documents(self, texts):
        return np.vstack([self._vector(t) for t in texts])

    def encode_query(self, text):
        return self._vector(text)


@pytest.fixture(scope="module")
def chunks():
    return load_chunks(get_settings().knowledge_base_file)


@pytest.fixture(scope="module")
def retriever(chunks):
    return HybridRetriever(chunks, StubEmbedder(), rrf_k=60, default_top_k=4)


# --- chunker ---------------------------------------------------------------


def test_every_policy_has_both_language_editions(chunks):
    editions: dict[str, set[str]] = {}
    for chunk in chunks:
        editions.setdefault(chunk.policy_id, set()).add(chunk.language)
    assert editions, "knowledge base produced no chunks"
    assert all(langs == {"en", "th"} for langs in editions.values()), editions


def test_chunk_text_carries_its_identifier(chunks):
    chunk = chunks[0]
    assert chunk.text.startswith(chunk.policy_id)
    assert chunk.body in chunk.text
    assert chunk.chunk_id == f"{chunk.policy_id}#{chunk.language}"


def test_file_level_comments_are_not_chunked(chunks):
    assert not any(c.body.lstrip().startswith("# Siam Horizon") for c in chunks)


# --- tokenizer -------------------------------------------------------------


def test_thai_text_is_segmented_into_words():
    tokens = tokenize("เบี้ยเลี้ยงเดินทางต่างประเทศ")
    assert len(tokens) > 1, "Thai run collapsed into a single token"
    assert "เบี้ยเลี้ยง" in tokens


def test_policy_identifier_survives_as_one_token():
    assert "pol-hr-014" in tokenize("Please explain POL-HR-014 to me")


def test_stopwords_are_dropped():
    assert "the" not in tokenize("the travel policy")


def test_mixed_script_input_yields_both_languages():
    tokens = tokenize("ใช้ ChatGPT กับข้อมูลลูกค้า")
    assert "chatgpt" in tokens
    assert any("฀" <= t[0] <= "๿" for t in tokens)


# --- BM25 ------------------------------------------------------------------


def test_bm25_prefers_the_document_containing_the_term():
    index = BM25Index([["per", "diem", "tokyo"], ["fire", "drill", "evacuation"]])
    scores = index.score(["per", "diem"])
    assert scores[0] > scores[1] == 0.0


def test_bm25_returns_zeros_for_unknown_terms():
    index = BM25Index([["alpha"], ["beta"]])
    assert np.all(index.score(["gamma"]) == 0.0)


def test_bm25_rejects_an_empty_corpus():
    with pytest.raises(ValueError):
        BM25Index([])


# --- fusion ----------------------------------------------------------------


def test_ranks_are_zero_based_and_descending():
    assert list(_ranks_from_scores(np.array([0.1, 0.9, 0.5]))) == [2, 0, 1]


def test_search_returns_one_snippet_per_policy(retriever):
    hits = retriever.search("international travel approval", top_k=4)
    assert len(hits) == 4
    assert len({hit.chunk.policy_id for hit in hits}) == 4


def test_results_are_ordered_by_fused_score(retriever):
    scores = [hit.fused_score for hit in retriever.search("travel expenses", top_k=4)]
    assert scores == sorted(scores, reverse=True)


def test_preferred_language_wins_the_edition(retriever):
    for hit in retriever.search("travel", top_k=3, prefer_language="th"):
        assert hit.chunk.language == "th"
        assert hit.matched_language in {"en", "th"}


def test_language_swap_is_flagged(retriever):
    hits = retriever.search("international business travel", top_k=3, prefer_language="th")
    swapped = [hit for hit in hits if hit.language_swapped]
    assert all(hit.matched_language == "en" for hit in swapped)


def test_exact_policy_code_ranks_first(retriever):
    top = retriever.search("POL-IT-019", top_k=1)[0]
    assert top.chunk.policy_id == "POL-IT-019"


# --- language detection -----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What is the travel policy?", "en"),
        ("ขอลาพักร้อนกี่วัน", "th"),
        ("ใช้ ChatGPT ได้ไหม", "th"),
        ("POL-HR-014", "en"),
        ("", "en"),
    ],
)
def test_detect_language(text, expected):
    assert detect_language(text) == expected
