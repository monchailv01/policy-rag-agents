#!/usr/bin/env python3
"""Inspect the retriever on its own, with no LLM in the loop.

Running this is the fastest way to see *why* a snippet was selected: it prints
the BM25 rank, the dense rank and the fused RRF score for every hit.

    python scripts/inspect_retrieval.py
    python scripts/inspect_retrieval.py "เบิกค่าเดินทางได้เท่าไหร่"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import HybridRetriever  # noqa: E402
from src.utils import detect_language  # noqa: E402

DEMO_QUERIES = [
    "What is the policy on international travel?",
    "How much per diem do I get in Tokyo?",
    "เบี้ยเลี้ยงเดินทางไปญี่ปุ่นได้วันละเท่าไหร่",
    "ใช้ ChatGPT กับข้อมูลลูกค้าได้ไหม",
    "POL-HR-022",
    "Can I accept a gift from a supplier?",
]


def main() -> None:
    queries = sys.argv[1:] or DEMO_QUERIES
    retriever = HybridRetriever.from_settings()

    stats = retriever.describe()
    print("Index:", ", ".join(f"{k}={v}" for k, v in stats.items()))

    for query in queries:
        language = detect_language(query)
        print(f"\n{'=' * 78}\nQUERY [{language}]  {query}\n{'-' * 78}")
        for position, hit in enumerate(
            retriever.search(query, prefer_language=language), start=1
        ):
            flag = f" (matched {hit.matched_language} edition)" if hit.language_swapped else ""
            print(
                f"{position}. [{hit.chunk.policy_id} · {hit.chunk.language}] {hit.chunk.title}{flag}\n"
                f"   rrf={hit.fused_score:.5f}  "
                f"bm25=#{hit.bm25_rank} ({hit.bm25_score:.3f})  "
                f"dense=#{hit.dense_rank} ({hit.dense_score:.3f})"
            )


if __name__ == "__main__":
    main()
