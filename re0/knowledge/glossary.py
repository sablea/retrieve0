"""Industrial jargon glossary + semantic lookup."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from re0.knowledge.embedder import Embedder


@dataclass
class GlossaryEntry:
    term: str
    aliases: list[str] = field(default_factory=list)
    definition: str = ""
    related_tables: list[str] = field(default_factory=list)
    related_columns: list[str] = field(default_factory=list)

    def render(self) -> str:
        alias = f"（别名：{', '.join(self.aliases)}）" if self.aliases else ""
        rel = []
        if self.related_tables:
            rel.append(f"相关表: {', '.join(self.related_tables)}")
        if self.related_columns:
            rel.append(f"相关列: {', '.join(self.related_columns)}")
        tail = f"\n  {'；'.join(rel)}" if rel else ""
        return f"- **{self.term}**{alias}: {self.definition}{tail}"

    def indexable_text(self) -> str:
        parts = [self.term, *self.aliases]
        if self.definition:
            parts.append(self.definition)
        return " / ".join(p for p in parts if p)


class Glossary:
    def __init__(
        self,
        entries: list[GlossaryEntry],
        embedder: Embedder,
        cache_dir: str | Path = ".re0_cache",
    ):
        self.entries = entries
        self.embedder = embedder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings: np.ndarray | None = None

    @classmethod
    def from_yaml(cls, path: str | Path, embedder: Embedder, cache_dir: str | Path) -> "Glossary":
        p = Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        items = raw.get("entries") or raw.get("terms") or []
        entries = [
            GlossaryEntry(
                term=i["term"],
                aliases=i.get("aliases") or [],
                definition=i.get("definition", ""),
                related_tables=i.get("related_tables") or [],
                related_columns=i.get("related_columns") or [],
            )
            for i in items
        ]
        return cls(entries, embedder, cache_dir)

    def _signature(self) -> str:
        h = hashlib.sha256()
        for e in self.entries:
            h.update(e.indexable_text().encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()[:16]

    def _cache_path(self) -> Path:
        return self.cache_dir / f"glossary-{self._signature()}.npz"

    def build_index(self) -> None:
        if not self.entries:
            self._embeddings = np.zeros((0, 1), dtype=np.float32)
            return
        cache = self._cache_path()
        if cache.exists():
            data = np.load(cache)
            self._embeddings = data["emb"].astype(np.float32)
            return
        texts = [e.indexable_text() for e in self.entries]
        self._embeddings = self.embedder.encode(texts)
        np.savez(cache, emb=self._embeddings)

    def lookup(self, query: str, top_k: int = 5, threshold: float = 0.4) -> list[GlossaryEntry]:
        if not self.entries:
            return []
        if self._embeddings is None:
            self.build_index()
        assert self._embeddings is not None
        qv = self.embedder.encode_one(query)
        sims = (self._embeddings @ qv).astype(float)
        idx = np.argsort(-sims)[:top_k]
        out: list[GlossaryEntry] = []
        for i in idx:
            if float(sims[i]) >= threshold:
                out.append(self.entries[int(i)])
        return out

    def render_matches(self, query: str, top_k: int = 5, threshold: float = 0.4) -> str:
        matches = self.lookup(query, top_k=top_k, threshold=threshold)
        if not matches:
            return ""
        return "## 术语提示\n" + "\n".join(m.render() for m in matches)
