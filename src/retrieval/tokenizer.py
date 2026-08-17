"""Bilingual tokenisation for the lexical (BM25) half of the retriever.

Thai is written without spaces, so a naive ``text.split()`` collapses an entire
Thai sentence into one token and BM25 degenerates to exact-sentence matching.
Here each Thai run is segmented with PyThaiNLP's ``newmm`` dictionary matcher
while Latin text falls back to a word regex.  Policy identifiers such as
``POL-HR-014`` are additionally emitted whole so an exact-code query still
scores a direct hit.
"""

from __future__ import annotations

import re
from functools import lru_cache

_THAI_RUN = re.compile(r"[\u0E00-\u0E7F]+")
_LATIN_WORD = re.compile(r"[a-z0-9]+")
_POLICY_ID = re.compile(r"\b[a-z]{2,4}-[a-z]{2,4}-\d{2,4}\b")

_EN_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the
    their there they this to was were what when where which who why will with your you
    do does did can could should would may might must not no if then than about into
    over under between per each any all more most such other some own same so""".split()
)


@lru_cache(maxsize=1)
def _thai_stopwords() -> frozenset[str]:
    from pythainlp.corpus.common import thai_stopwords

    return frozenset(thai_stopwords())


@lru_cache(maxsize=1)
def _thai_segmenter():
    from pythainlp.tokenize import word_tokenize

    return word_tokenize


def tokenize(text: str) -> list[str]:
    """Split mixed Thai/English text into lowercase, stopword-filtered tokens."""
    lowered = text.lower()
    tokens: list[str] = _POLICY_ID.findall(lowered)

    segment = _thai_segmenter()
    cursor = 0
    for run in _THAI_RUN.finditer(lowered):
        tokens += _LATIN_WORD.findall(lowered[cursor : run.start()])
        tokens += segment(run.group(), engine="newmm", keep_whitespace=False)
        cursor = run.end()
    tokens += _LATIN_WORD.findall(lowered[cursor:])

    stop = _EN_STOPWORDS | _thai_stopwords()
    return [t for t in (tok.strip() for tok in tokens) if t and t not in stop]
