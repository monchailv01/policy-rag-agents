"""Small shared helpers."""

from __future__ import annotations

import re

_THAI_CHARS = re.compile(r"[฀-๿]")
_ALPHA_CHARS = re.compile(r"[^\W\d_]", re.UNICODE)


def detect_language(text: str, *, threshold: float = 0.15) -> str:
    """Return ``"th"`` or ``"en"`` for a piece of user input.

    A character-ratio heuristic is deliberate: it is deterministic, instant and
    free, whereas asking the LLM to classify the language would add a round trip
    to every single turn for a decision that Unicode already answers.
    """
    letters = _ALPHA_CHARS.findall(text)
    if not letters:
        return "en"
    thai = len(_THAI_CHARS.findall(text))
    return "th" if thai / len(letters) >= threshold else "en"
