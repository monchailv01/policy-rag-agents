#!/usr/bin/env python3
"""Command-line entry point for the two-agent policy assistant.

    python main.py                                  interactive chat (remembers context)
    python main.py "What is the policy on international travel?"
    python main.py --demo                           run the built-in sample queries
    python main.py --trace "..."                    also show the retrieval scoring trail

Interactive mode persists its history to SQLite, so closing the process and
starting it again with the same ``--thread`` continues the same conversation.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.rule import Rule  # noqa: E402
from rich.table import Table  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.graph import build_graph, get_retriever  # noqa: E402

DEMO_QUERIES = [
    "What is the policy on international travel?",
    "How much per diem do I get for a five-day trip to Tokyo, and by when must I file the claim?",
    "ขอลาพักร้อนติดกัน 5 วัน ต้องยื่นล่วงหน้ากี่วัน และถ้าลาป่วยต้องใช้ใบรับรองแพทย์เมื่อไหร่",
    "Can I paste a customer's account number into ChatGPT to summarise it?",
    "What is the company's pet insurance benefit?",
]

NODE_LABELS = {
    "contextualize": "contextualize      · rewrite + detect language",
    "data_retriever": "data_retriever     · Agent 1 (RAG)",
    "retrieval_tools": "retrieval_tools    · search_knowledge_base",
    "handoff": "handoff            · pass snippets to Agent 2",
    "report_generator": "report_generator   · Agent 2 (synthesis)",
}


def render_banner(console: Console) -> None:
    settings = get_settings()
    stats = get_retriever().describe()
    console.print(
        Panel(
            f"[bold]Siam Horizon policy assistant[/bold]  ·  LangGraph two-agent RAG\n"
            f"LLM        {settings.llm_model}\n"
            f"Embeddings {stats['embedder']}  ({stats['dimension']}d)\n"
            f"Index      {stats['policies']} policies / {stats['chunks']} sections "
            f"/ langs {'+'.join(stats['languages'])} / BM25 vocab {stats['vocabulary']}",
            border_style="cyan",
        )
    )


def render_trace(console: Console, state: dict) -> None:
    """Show which searches were issued and how each snippet was scored."""
    searches = state.get("searches") or []
    if searches:
        console.print(
            "[dim]searches issued:[/dim] "
            + "; ".join(f"{s['query']!r} -> {s['hits']} hits" for s in searches)
        )

    snippets = state.get("snippets") or []
    if not snippets:
        return
    table = Table(title="Retrieved snippets", title_justify="left", header_style="bold")
    for column in ("policy", "lang", "title", "RRF", "BM25", "dense"):
        table.add_column(column, overflow="fold")
    for snippet in snippets:
        table.add_row(
            snippet["policy_id"],
            snippet["language"] + ("*" if snippet["language_swapped"] else ""),
            snippet["title"],
            f"{snippet['fused_score']:.5f}",
            f"#{snippet['bm25_rank']} ({snippet['bm25_score']:.2f})",
            f"#{snippet['dense_rank']} ({snippet['dense_score']:.2f})",
        )
    console.print(table)

    if state.get("handoff_note"):
        console.print(
            Panel(
                state["handoff_note"],
                title="Data Retriever's handoff note",
                border_style="yellow",
            )
        )


def run_turn(
    graph, question: str, thread_id: str, console: Console, *, trace: bool
) -> dict:
    """Stream one question through the graph, printing each node as it runs."""
    console.print(Rule(f"[bold cyan]{question}"))

    state: dict = {}
    for update in graph.stream(
        {"messages": [HumanMessage(question)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        for node, payload in update.items():
            console.print(f"  [dim]->[/dim] [green]{NODE_LABELS.get(node, node)}[/green]")
            if not payload:
                continue
            state.update(payload)
            for message in payload.get("retriever_messages", []):
                for call in getattr(message, "tool_calls", None) or []:
                    console.print(
                        f"     [dim]search_knowledge_base({call['args'].get('query', '')!r})[/dim]"
                    )

    if state.get("standalone_question") not in (None, question):
        console.print(f"  [dim]rewritten as:[/dim] {state['standalone_question']}")
    if trace:
        render_trace(console, state)

    console.print(
        Panel(
            Markdown(state.get("answer", "(no answer produced)")),
            title="Report Generator",
            border_style="green",
        )
    )
    return state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-agent RAG assistant over the company policy handbook."
    )
    parser.add_argument("question", nargs="*", help="a single question to answer")
    parser.add_argument("--demo", action="store_true", help="run the sample queries")
    parser.add_argument(
        "--trace", action="store_true", help="show retrieval scores and the handoff note"
    )
    parser.add_argument(
        "--thread", default=None, help="conversation id to resume (default: new)"
    )
    args = parser.parse_args()

    console = Console()
    render_banner(console)

    with SqliteSaver.from_conn_string(str(get_settings().chat_db_file)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        thread_id = args.thread or f"cli-{uuid.uuid4().hex[:8]}"

        if args.demo:
            for question in DEMO_QUERIES:
                run_turn(
                    graph, question, f"{thread_id}-{uuid.uuid4().hex[:4]}", console, trace=args.trace
                )
            return

        if args.question:
            run_turn(graph, " ".join(args.question), thread_id, console, trace=args.trace)
            return

        console.print(f"[dim]thread {thread_id} · blank line or 'exit' to quit[/dim]")
        while True:
            try:
                question = console.input("\n[bold]you >[/bold] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() in {"exit", "quit"}:
                break
            run_turn(graph, question, thread_id, console, trace=args.trace)


if __name__ == "__main__":
    main()
