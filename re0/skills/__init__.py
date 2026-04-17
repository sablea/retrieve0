"""Built-in skills + dynamic registration."""
from __future__ import annotations

from re0.core.skill import Skill, SkillRegistry
from re0.skills.sql_retrieval.handler import SqlRetrievalSkill


def default_registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.register(SqlRetrievalSkill())
    return reg


__all__ = ["default_registry", "SqlRetrievalSkill", "Skill", "SkillRegistry"]
