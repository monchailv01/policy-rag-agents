"""Parse ``knowledge_base.txt`` into one chunk per policy section.

The handbook is authored with an explicit section marker::

    ### POL-HR-014 | EN | International Business Travel

Chunking on that marker instead of on a fixed token window means every chunk is
a self-contained policy with its identifier and title intact, which is what
makes citations such as ``[POL-HR-014]`` possible in the final report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SECTION = re.compile(
    r"^###\s*(?P<policy_id>[A-Z0-9\-]+)\s*\|\s*(?P<language>[A-Za-z]{2})\s*\|\s*(?P<title>.+?)\s*$"
)


@dataclass(frozen=True, slots=True)
class PolicyChunk:
    """A single policy section in a single language."""

    policy_id: str
    language: str
    title: str
    body: str

    @property
    def chunk_id(self) -> str:
        return f"{self.policy_id}#{self.language}"

    @property
    def text(self) -> str:
        """Heading + body, i.e. what actually gets indexed and embedded."""
        return f"{self.policy_id} {self.title}\n{self.body}"

    def as_dict(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "policy_id": self.policy_id,
            "language": self.language,
            "title": self.title,
            "body": self.body,
        }


def load_chunks(path: str | Path) -> list[PolicyChunk]:
    """Read the knowledge base file and return its policy sections.

    Lines starting with ``#`` that are not section markers are treated as file
    level comments and ignored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {path}")

    chunks: list[PolicyChunk] = []
    header: re.Match[str] | None = None
    buffer: list[str] = []

    def flush() -> None:
        if header is None:
            return
        body = "\n".join(buffer).strip()
        if body:
            chunks.append(
                PolicyChunk(
                    policy_id=header["policy_id"],
                    language=header["language"].lower(),
                    title=header["title"],
                    body=body,
                )
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        match = _SECTION.match(line)
        if match:
            flush()
            header, buffer = match, []
        elif header is None:
            continue  # file-level comments before the first section
        else:
            buffer.append(line)
    flush()

    if not chunks:
        raise ValueError(f"No '### <ID> | <LANG> | <TITLE>' sections found in {path}")
    return chunks
