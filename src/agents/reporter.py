"""The Report Generator agent.

Deliberately tool-free: it only ever sees the snippets the Data Retriever handed
over, which is what keeps the final answer grounded in the knowledge base and
auditable against it.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.agents.prompts import LANGUAGE_NAMES, REPORT_GENERATOR_PROMPT
from src.llm import get_llm

SNIPPET_BLOCK = """### Section {position}: [{policy_id}] {title}
{body}"""

HANDOFF_TEMPLATE = """\
Employee's question:
{question}

Retrieval note from the Data Retriever:
{handoff_note}

Policy sections retrieved from the handbook:
{sections}

Write the final answer for the employee now."""

NO_SNIPPETS = "(The Data Retriever returned no policy sections.)"


def build_report_generator() -> BaseChatModel:
    """Return the model used to synthesise the final answer (no tools bound)."""
    return get_llm()


def render_system_prompt(language: str) -> str:
    """System prompt, specialised to the language the employee wrote in."""
    return REPORT_GENERATOR_PROMPT.format(
        language_name=LANGUAGE_NAMES.get(language, "English")
    )


def render_handoff(question: str, handoff_note: str, snippets: list[dict]) -> str:
    """Package the retriever's output into the reporter's input message."""
    sections = "\n\n".join(
        SNIPPET_BLOCK.format(
            position=position,
            policy_id=snippet["policy_id"],
            title=snippet["title"],
            body=snippet["body"],
        )
        for position, snippet in enumerate(snippets, start=1)
    )
    return HANDOFF_TEMPLATE.format(
        question=question,
        handoff_note=handoff_note or "(none)",
        sections=sections or NO_SNIPPETS,
    )
