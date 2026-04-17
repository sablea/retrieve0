"""SQLite-backed memory store with embedding recall."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from re0.knowledge.embedder import Embedder


@dataclass
class MemoryRecord:
    id: int
    question: str
    sql: str
    answer: str
    hit_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "question": self.question,
            "sql": self.sql,
            "answer": self.answer,
            "hit_count": self.hit_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
        if self.similarity is not None:
            d["similarity"] = self.similarity
        return d


class MemoryStore:
    def __init__(self, path: str | Path, embedder: Embedder, recall_threshold: float = 0.85):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.recall_threshold = recall_threshold
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def _init(self) -> None:
        with self._conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS memories(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    sql TEXT NOT NULL DEFAULT '',
                    answer TEXT NOT NULL DEFAULT '',
                    embedding BLOB NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row, similarity: float | None = None) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            question=row["question"],
            sql=row["sql"],
            answer=row["answer"],
            hit_count=row["hit_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            similarity=similarity,
        )

    def _load_all(self) -> tuple[list[sqlite3.Row], np.ndarray]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, question, sql, answer, embedding, hit_count, metadata_json, created_at, updated_at FROM memories"
            ).fetchall()
        if not rows:
            return [], np.zeros((0, 1), dtype=np.float32)
        embs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        return rows, embs

    def save(self, question: str, sql: str = "", answer: str = "", metadata: dict[str, Any] | None = None) -> MemoryRecord:
        emb = self.embedder.encode_one(question)
        now = time.time()
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO memories(question, sql, answer, embedding, metadata_json, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    question,
                    sql,
                    answer,
                    emb.tobytes(),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            mid = cur.lastrowid
        return MemoryRecord(
            id=mid, question=question, sql=sql, answer=answer,
            created_at=now, updated_at=now, metadata=metadata or {},
        )

    def recall(self, question: str, threshold: float | None = None, top_k: int = 1) -> list[MemoryRecord]:
        rows, embs = self._load_all()
        if not rows:
            return []
        qv = self.embedder.encode_one(question)
        sims = (embs @ qv).astype(float)
        order = np.argsort(-sims)
        thr = self.recall_threshold if threshold is None else threshold
        out: list[MemoryRecord] = []
        for i in order[:top_k]:
            s = float(sims[i])
            if s >= thr:
                out.append(self._row_to_record(rows[i], similarity=s))
        return out

    def touch_hit(self, memory_id: int) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE memories SET hit_count=hit_count+1, updated_at=? WHERE id=?",
                (time.time(), memory_id),
            )

    def correct(self, memory_id: int, new_sql: str | None = None, new_answer: str | None = None) -> bool:
        fields, params = [], []
        if new_sql is not None:
            fields.append("sql=?")
            params.append(new_sql)
        if new_answer is not None:
            fields.append("answer=?")
            params.append(new_answer)
        if not fields:
            return False
        fields.append("updated_at=?")
        params.append(time.time())
        params.append(memory_id)
        with self._conn() as c:
            cur = c.execute(
                f"UPDATE memories SET {', '.join(fields)} WHERE id=?", params
            )
            return cur.rowcount > 0

    def delete(self, memory_id: int) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            return cur.rowcount > 0

    def list(self, query: str | None = None, limit: int = 50) -> list[MemoryRecord]:
        if query:
            records = self.recall(query, threshold=0.0, top_k=limit)
            return records
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, question, sql, answer, embedding, hit_count, metadata_json, created_at, updated_at "
                "FROM memories ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]
