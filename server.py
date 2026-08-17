#!/usr/bin/env python3
"""FastAPI server exposing the two-agent workflow to the browser.

The web layer adds no reasoning of its own: it imports the very same compiled
graph as ``main.py`` and streams its events out over Server-Sent Events, so the
page shows what the agents actually did rather than a re-enactment.

    python server.py            # http://localhost:8100
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.config import PROJECT_ROOT, get_settings
from src.graph import build_graph, get_retriever
from src.sessions import SessionStore

WEB_DIR = PROJECT_ROOT / "web"

NODE_LABELS = {
    "contextualize": "Contextualise",
    "data_retriever": "Data Retriever",
    "retrieval_tools": "search_knowledge_base",
    "handoff": "Handoff",
    "report_generator": "Report Generator",
}


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hold one checkpointer, one graph and one warm retriever for the process."""
    settings = get_settings()
    async with AsyncSqliteSaver.from_conn_string(str(settings.chat_db_file)) as saver:
        app.state.retriever = await asyncio.to_thread(get_retriever)
        app.state.graph = build_graph(retriever=app.state.retriever, checkpointer=saver)
        app.state.sessions = SessionStore(settings.chat_db_file)
        try:
            yield
        finally:
            app.state.sessions.close()


app = FastAPI(title="Siam Horizon Policy Assistant", lifespan=lifespan)


# --- metadata ---------------------------------------------------------------


@app.get("/api/meta")
async def meta() -> dict:
    """Everything the page needs to render its header, diagram and KB viewer."""
    settings = get_settings()
    retriever = app.state.retriever
    policies: dict[str, dict] = {}
    for chunk in retriever.chunks:
        entry = policies.setdefault(
            chunk.policy_id, {"policy_id": chunk.policy_id, "editions": {}}
        )
        entry["editions"][chunk.language] = {"title": chunk.title, "body": chunk.body}
    return {
        "llm_model": settings.llm_model,
        "index": retriever.describe(),
        "mermaid": app.state.graph.get_graph().draw_mermaid(),
        "policies": list(policies.values()),
    }


# --- sessions ---------------------------------------------------------------


@app.get("/api/sessions")
async def list_sessions() -> list[dict]:
    return app.state.sessions.list()


@app.post("/api/sessions")
async def create_session() -> dict:
    return app.state.sessions.create(f"web-{uuid.uuid4().hex[:10]}")


@app.get("/api/sessions/{thread_id}/messages")
async def session_messages(thread_id: str) -> list[dict]:
    """Replay a stored conversation from the LangGraph checkpoint."""
    state = await app.state.graph.aget_state(
        {"configurable": {"thread_id": thread_id}}
    )
    messages = (state.values or {}).get("messages", [])
    return [
        {
            "role": "user" if isinstance(message, HumanMessage) else "assistant",
            "content": str(message.content),
        }
        for message in messages
        if isinstance(message, (HumanMessage, AIMessage)) and message.content
    ]


@app.delete("/api/sessions/{thread_id}")
async def delete_session(thread_id: str) -> dict:
    await app.state.graph.checkpointer.adelete_thread(thread_id)
    app.state.sessions.delete(thread_id)
    return {"deleted": thread_id}


# --- chat -------------------------------------------------------------------


def _sse(event: str, **data) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    """Stream one turn of the workflow: node transitions, then answer tokens."""
    thread_id = request.thread_id or f"web-{uuid.uuid4().hex[:10]}"
    app.state.sessions.create(thread_id)
    app.state.sessions.record_turn(thread_id, request.question)

    graph = app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    async def event_stream():
        started = time.perf_counter()
        final: dict = {}
        yield _sse("start", thread_id=thread_id)
        try:
            async for mode, payload in graph.astream(
                {"messages": [HumanMessage(request.question)]},
                config=config,
                stream_mode=["updates", "messages"],
            ):
                if mode == "messages":
                    chunk, metadata = payload
                    if metadata.get("langgraph_node") == "report_generator":
                        text = str(chunk.content or "")
                        if text:
                            yield _sse("token", text=text)
                    continue

                for node, update in (payload or {}).items():
                    yield _sse("node", node=node, label=NODE_LABELS.get(node, node))
                    if not update:
                        continue
                    final.update(update)

                    if node == "contextualize":
                        yield _sse(
                            "rewrite",
                            standalone=update.get("standalone_question", ""),
                            language=update.get("language", "en"),
                        )
                    elif node == "retrieval_tools":
                        # Emitted from the tool node, not from the agent's
                        # request, so a search the round cap rejected never
                        # shows up in the trace as if it had run.
                        for message in update.get("retriever_messages", []):
                            artifact = getattr(message, "artifact", None)
                            if isinstance(artifact, dict):
                                yield _sse(
                                    "search",
                                    query=artifact.get("query", ""),
                                    hits=len(artifact.get("snippets", [])),
                                )
                    elif node == "handoff":
                        yield _sse(
                            "snippets",
                            snippets=update.get("snippets", []),
                            handoff_note=update.get("handoff_note", ""),
                        )

            yield _sse(
                "done",
                answer=final.get("answer", ""),
                searches=final.get("searches", []),
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:  # surfaced in the UI instead of a dead stream
            yield _sse("error", message=f"{type(exc).__name__}: {exc}")

    return EventSourceResponse(event_stream())


# --- static -----------------------------------------------------------------

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="web/index.html is missing")
    return FileResponse(page)


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
