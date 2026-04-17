"""RunContext bundles providers for skill handlers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from re0.core.config import AppConfig
    from re0.db.base import SqlExecutor
    from re0.knowledge.schema import SchemaProvider
    from re0.knowledge.glossary import Glossary
    from re0.memory.store import MemoryStore
    from re0.llm.base import LLMProvider


@dataclass
class RunContext:
    config: "AppConfig"
    llm: "LLMProvider"
    db: "SqlExecutor"
    schema: "SchemaProvider"
    glossary: "Glossary"
    memory: "MemoryStore"
    session_id: str
