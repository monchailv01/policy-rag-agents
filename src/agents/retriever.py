"""The Data Retriever agent.

An LLM bound to exactly one tool — the hybrid knowledge-base search — and
instructed never to answer.  It decides *what* to search for and *how many
times*, which is the agentic part: a single fixed vector lookup could not widen
a query after a weak first result or split a two-part question into two searches.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.agents.prompts import DATA_RETRIEVER_PROMPT
from src.llm import get_llm

#: Upper bound on tool calls per turn, so a confused model cannot loop forever.
MAX_SEARCH_ROUNDS = 3


def build_data_retriever(search_tool: BaseTool) -> tuple[BaseChatModel, str]:
    """Return the tool-bound model and the system prompt for the retriever."""
    llm = get_llm().bind_tools([search_tool])
    prompt = DATA_RETRIEVER_PROMPT.format(max_rounds=MAX_SEARCH_ROUNDS)
    return llm, prompt
