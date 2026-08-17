"""The custom retrieval tool exposed to the Data Retriever agent.

This is the RAG mechanism the agent is allowed to call.  It is built as a
factory closing over an already-warm :class:`HybridRetriever` so that the model
and the BM25 index are loaded exactly once per process rather than per call.

The tool returns ``(content, artifact)``: the string goes back to the LLM as the
tool result, while the artifact carries the structured scoring trail that the
graph stores in state and the web UI renders.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from src.retrieval import HybridRetriever

SNIPPET_TEMPLATE = """[{policy_id} | {language} | {title}]
{body}"""


def build_search_tool(retriever: HybridRetriever) -> BaseTool:
    """Create the ``search_knowledge_base`` tool bound to ``retriever``."""

    @tool("search_knowledge_base", response_format="content_and_artifact")
    def search_knowledge_base(
        query: str,
        state: Annotated[dict, InjectedState],
        top_k: int = 0,
    ) -> tuple[str, dict]:
        """Search the Siam Horizon Group policy handbook and return raw excerpts.

        Use short, keyword-rich queries describing the information you need, in
        the same language the employee used. Call this more than once with
        different wording when a request covers several distinct topics.

        Args:
            query: What to look for, e.g. "international travel approval" or
                "เบี้ยเลี้ยงเดินทางต่างประเทศ".
            top_k: How many policy sections to return. Leave unset for the
                configured default.

        Returns:
            The matching policy sections verbatim, with their policy IDs.
        """
        hits = retriever.search(
            query,
            top_k=top_k or None,
            prefer_language=(state or {}).get("language"),
        )
        if not hits:
            return "NO_RESULTS", {"query": query, "snippets": []}

        rendered = "\n\n".join(
            SNIPPET_TEMPLATE.format(
                policy_id=hit.chunk.policy_id,
                language=hit.chunk.language.upper(),
                title=hit.chunk.title,
                body=hit.chunk.body,
            )
            for hit in hits
        )
        artifact = {"query": query, "snippets": [hit.as_dict() for hit in hits]}
        return rendered, artifact

    return search_knowledge_base
