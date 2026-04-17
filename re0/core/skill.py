"""Skill base class and registry."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from re0.core.context import RunContext


ToolHandler = Callable[["RunContext", dict[str, Any]], Any]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Skill(Protocol):
    name: str
    description: str

    def system_prompt(self, ctx: "RunContext") -> str: ...
    def tools(self, ctx: "RunContext") -> list[ToolSpec]: ...
    def on_user_message(self, ctx: "RunContext", message: str) -> str | None: ...


@dataclass
class SkillRegistry:
    _skills: dict[str, Skill] = field(default_factory=dict)

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def enabled(self, names: list[str]) -> list[Skill]:
        out = []
        for n in names:
            s = self._skills.get(n)
            if s is not None:
                out.append(s)
        return out

    def all(self) -> list[Skill]:
        return list(self._skills.values())
