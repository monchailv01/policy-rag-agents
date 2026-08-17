"""A tiny registry of chat sessions.

LangGraph's checkpointer already stores the *content* of every conversation
keyed by ``thread_id``.  What it does not store is the handful of presentation
details the sidebar needs — a title, when the thread was last used, how many
turns it holds — so those live in one extra table in the same SQLite file.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    thread_id  TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT 'New chat',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turns      INTEGER NOT NULL DEFAULT 0
)
"""

TITLE_MAX_CHARS = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SessionStore:
    """Thread-safe CRUD over the ``sessions`` table."""

    def __init__(self, path: str | Path) -> None:
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(_SCHEMA)
            self._connection.commit()

    def create(self, thread_id: str) -> dict:
        timestamp = _now()
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO sessions (thread_id, created_at, updated_at)"
                " VALUES (?, ?, ?)",
                (thread_id, timestamp, timestamp),
            )
            self._connection.commit()
        return self.get(thread_id) or {}

    def record_turn(self, thread_id: str, question: str) -> None:
        """Bump the turn counter, and adopt the first question as the title."""
        title = question.strip().replace("\n", " ")
        if len(title) > TITLE_MAX_CHARS:
            title = title[: TITLE_MAX_CHARS - 1].rstrip() + "…"
        with self._lock:
            self._connection.execute(
                "UPDATE sessions SET turns = turns + 1, updated_at = ?,"
                " title = CASE WHEN turns = 0 THEN ? ELSE title END"
                " WHERE thread_id = ?",
                (_now(), title, thread_id),
            )
            self._connection.commit()

    def get(self, thread_id: str) -> dict | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return dict(row) if row else None

    def list(self) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, thread_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "DELETE FROM sessions WHERE thread_id = ?", (thread_id,)
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()
