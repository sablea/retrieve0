"""Assemble runtime components from config."""
from __future__ import annotations

from dataclasses import dataclass

from re0.core.agent import Agent
from re0.core.config import AppConfig
from re0.core.session import SessionStore
from re0.db.base import SqlExecutor, build_executor
from re0.knowledge.embedder import Embedder, get_embedder
from re0.knowledge.glossary import Glossary
from re0.knowledge.schema import SchemaProvider
from re0.llm.base import LLMProvider
from re0.llm.openai_compat import OpenAICompatProvider
from re0.memory.store import MemoryStore
from re0.skills import default_registry
from re0.core.skill import SkillRegistry


@dataclass
class Runtime:
    config: AppConfig
    llm: LLMProvider
    db: SqlExecutor
    schema: SchemaProvider
    glossary: Glossary
    embedder: Embedder
    memory: MemoryStore
    sessions: SessionStore
    registry: SkillRegistry
    agent: Agent


def build_runtime(cfg: AppConfig) -> Runtime:
    embedder = get_embedder(cfg.embedding)
    schema = SchemaProvider.from_yaml(cfg.knowledge.schema_path)
    glossary = Glossary.from_yaml(cfg.knowledge.glossary_path, embedder, cfg.embedding.cache_dir)
    memory = MemoryStore(cfg.memory.path, embedder, recall_threshold=cfg.memory.recall_threshold)
    sessions = SessionStore(cfg.session.path)
    llm = OpenAICompatProvider(cfg.llm)
    db = build_executor(cfg.db)
    registry = default_registry()
    enabled_names = cfg.skills.get("enabled", ["sql_retrieval"])
    skills = registry.enabled(enabled_names)
    agent = Agent(llm=llm, skills=skills, store=sessions, max_steps=cfg.runtime.max_agent_steps)
    return Runtime(
        config=cfg,
        llm=llm,
        db=db,
        schema=schema,
        glossary=glossary,
        embedder=embedder,
        memory=memory,
        sessions=sessions,
        registry=registry,
        agent=agent,
    )
