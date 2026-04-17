"""Session persistence: history + pending ask_user state."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionState:
    id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_call_id: str | None = None
    pending_question: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def _init(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions(
                    id TEXT PRIMARY KEY,
                    history_json TEXT NOT NULL,
                    pending_tool_call_id TEXT,
                    pending_question TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def load(self, session_id: str) -> SessionState:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        if not row:
            return SessionState(id=session_id)
        return SessionState(
            id=row["id"],
            history=json.loads(row["history_json"]),
            pending_tool_call_id=row["pending_tool_call_id"],
            pending_question=row["pending_question"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save(self, state: SessionState) -> None:
        state.updated_at = time.time()
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO sessions(id, history_json, pending_tool_call_id,
                                     pending_question, metadata_json,
                                     created_at, updated_at)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    history_json=excluded.history_json,
                    pending_tool_call_id=excluded.pending_tool_call_id,
                    pending_question=excluded.pending_question,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    state.id,
                    json.dumps(state.history, ensure_ascii=False),
                    state.pending_tool_call_id,
                    state.pending_question,
                    json.dumps(state.metadata, ensure_ascii=False),
                    state.created_at,
                    state.updated_at,
                ),
            )

    def delete(self, session_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex
