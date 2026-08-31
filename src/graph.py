"""LangGraph orchestration of the two-agent workflow.

    START
      -> contextualize      rewrite a follow-up into a standalone question
      -> data_retriever <-> retrieval_tools     (agentic search loop)
      -> handoff            collect the snippets the retriever selected
      -> report_generator   synthesise the final answer
      -> END

The two agents never share a message list.  The retriever works in its own
scratchpad, and only the material it hands over reaches the report generator,
which is what makes this a real handoff rather than one long chain of thought.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode

from src.agents.prompts import CONTEXTUALIZE_PROMPT
from src.agents.reporter import (
    build_report_generator,
    render_handoff,
    render_system_prompt,
)
from src.agents.retriever import MAX_SEARCH_ROUNDS, build_data_retriever
from src.agents.tools import build_search_tool
from src.llm import get_llm
from src.retrieval import HybridRetriever
from src.utils import detect_language

#: How many prior messages of the conversation each agent may see.
HISTORY_WINDOW = 6

#: Cap on the snippets handed to the report generator, so that several searches
#: cannot flood it with near-duplicate material.
MAX_HANDOFF_SNIPPETS = 6


class RAGState(TypedDict, total=False):
    """State carried through the graph and persisted by the checkpointer."""

    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    standalone_question: str
    language: str
    retriever_messages: Annotated[list[AnyMessage], add_messages]
    searches: list[dict]
    snippets: list[dict]
    handoff_note: str
    answer: str


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Process-wide retriever: the embedding model is loaded only once."""
    return HybridRetriever.from_settings()


CAP_REACHED_NUDGE = (
    "You have reached the search limit and cannot search again. "
    "Write your retrieval note now, based only on what you have already found."
)


def _force_handoff_note(scratchpad: list[AnyMessage], backend: str = "cloud") -> str:
    """Ask the retriever for its note when the round cap cut it off mid-search.

    The agent stops with an unanswered tool call, so it never wrote the note the
    Report Generator expects. Re-prompting it once without tools — after
    dropping that dangling call, which the API would otherwise reject — keeps
    the handover honest instead of silently sending an empty note.
    """
    trimmed = list(scratchpad)
    while trimmed and isinstance(trimmed[-1], AIMessage) and trimmed[-1].tool_calls:
        trimmed.pop()
    if not trimmed:
        return ""
    try:
        response = get_llm(backend=backend).invoke(
            [*trimmed, HumanMessage(CAP_REACHED_NUDGE)]
        )
        return str(response.content).strip()
    except Exception:  # a missing note must not fail the whole turn
        return ""


def build_graph(
    *,
    retriever: HybridRetriever | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    llm_backend: str = "cloud",
):
    """Compile the workflow. Pass a checkpointer to enable multi-turn memory.

    ``llm_backend`` selects which configured model family powers every LLM call
    in this compiled graph — the hosted default, or the local GPU server.
    """
    retriever = retriever or get_retriever()
    search_tool = build_search_tool(retriever)
    retriever_llm, retriever_prompt = build_data_retriever(search_tool, llm_backend)
    reporter_llm = build_report_generator(llm_backend)

    # -- nodes --------------------------------------------------------------

    def contextualize(state: RAGState) -> RAGState:
        """Turn the latest message into a self-contained retrieval query."""
        question = str(state["messages"][-1].content)
        history = state["messages"][:-1]

        standalone = question
        if history:
            rewritten = get_llm(temperature=0.0, backend=llm_backend).invoke(
                [
                    SystemMessage(CONTEXTUALIZE_PROMPT),
                    *history[-HISTORY_WINDOW:],
                    HumanMessage(question),
                ]
            )
            standalone = str(rewritten.content).strip() or question

        return {
            "question": question,
            "standalone_question": standalone,
            "language": detect_language(question),
            # Start every turn with an empty retriever scratchpad; only the
            # user-visible `messages` list is meant to survive across turns.
            "retriever_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
            "searches": [],
            "snippets": [],
            "handoff_note": "",
            "answer": "",
        }

    def data_retriever(state: RAGState) -> RAGState:
        """Agent 1: decide what to search for, or declare the search finished."""
        scratchpad = list(state.get("retriever_messages") or [])
        opening: list[AnyMessage] = []
        if not scratchpad:
            opening = [
                SystemMessage(retriever_prompt),
                HumanMessage(state["standalone_question"]),
            ]
            scratchpad = opening

        response = retriever_llm.invoke(scratchpad)
        return {"retriever_messages": [*opening, response]}

    def handoff(state: RAGState) -> RAGState:
        """Collect what the retriever found and pass it to the second agent."""
        snippets: list[dict] = []
        searches: list[dict] = []
        seen: set[str] = set()

        for message in state.get("retriever_messages", []):
            if not isinstance(message, ToolMessage):
                continue
            artifact = message.artifact if isinstance(message.artifact, dict) else {}
            found = artifact.get("snippets", [])
            searches.append({"query": artifact.get("query", ""), "hits": len(found)})
            for snippet in found:
                if snippet["chunk_id"] not in seen:
                    seen.add(snippet["chunk_id"])
                    snippets.append(snippet)

        snippets.sort(key=lambda s: s["fused_score"], reverse=True)
        scratchpad = state.get("retriever_messages", [])
        note = next(
            (
                str(m.content)
                for m in reversed(scratchpad)
                if isinstance(m, AIMessage) and m.content and not m.tool_calls
            ),
            "",
        )
        if not note:
            note = _force_handoff_note(scratchpad, llm_backend)
        return {
            "searches": searches,
            "snippets": snippets[:MAX_HANDOFF_SNIPPETS],
            "handoff_note": note,
        }

    def report_generator(state: RAGState) -> RAGState:
        """Agent 2: synthesise the retrieved sections into the final answer."""
        response = reporter_llm.invoke(
            [
                SystemMessage(render_system_prompt(state.get("language", "en"))),
                *state["messages"][:-1][-HISTORY_WINDOW:],
                HumanMessage(
                    render_handoff(
                        state["question"],
                        state.get("handoff_note", ""),
                        state.get("snippets", []),
                    )
                ),
            ]
        )
        answer = str(response.content)
        return {"answer": answer, "messages": [AIMessage(answer)]}

    # -- wiring -------------------------------------------------------------

    def route_after_retriever(state: RAGState) -> Literal["retrieval_tools", "handoff"]:
        """Loop back to the tool while the agent keeps asking, up to the cap."""
        scratchpad = state.get("retriever_messages") or []
        last = scratchpad[-1] if scratchpad else None
        if isinstance(last, AIMessage) and last.tool_calls:
            rounds = sum(
                1 for m in scratchpad if isinstance(m, AIMessage) and m.tool_calls
            )
            if rounds <= MAX_SEARCH_ROUNDS:
                return "retrieval_tools"
        return "handoff"

    workflow = StateGraph(RAGState)
    workflow.add_node("contextualize", contextualize)
    workflow.add_node("data_retriever", data_retriever)
    workflow.add_node(
        "retrieval_tools", ToolNode([search_tool], messages_key="retriever_messages")
    )
    workflow.add_node("handoff", handoff)
    workflow.add_node("report_generator", report_generator)

    workflow.add_edge(START, "contextualize")
    workflow.add_edge("contextualize", "data_retriever")
    workflow.add_conditional_edges("data_retriever", route_after_retriever)
    workflow.add_edge("retrieval_tools", "data_retriever")
    workflow.add_edge("handoff", "report_generator")
    workflow.add_edge("report_generator", END)

    return workflow.compile(checkpointer=checkpointer)
