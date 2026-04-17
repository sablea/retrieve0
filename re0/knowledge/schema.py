"""Hand-written schema provider (yaml-backed)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ColumnSpec:
    name: str
    type: str = ""
    description: str = ""
    enum: list[str] = field(default_factory=list)
    nullable: bool = True


@dataclass
class TableSpec:
    name: str
    description: str = ""
    columns: list[ColumnSpec] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"## 表 `{self.name}`"]
        if self.description:
            lines.append(self.description)
        if self.primary_keys:
            lines.append(f"主键: {', '.join(self.primary_keys)}")
        lines.append("| 列 | 类型 | 说明 |")
        lines.append("|---|---|---|")
        for c in self.columns:
            desc = c.description
            if c.enum:
                desc = f"{desc}（取值：{', '.join(c.enum)}）" if desc else f"取值：{', '.join(c.enum)}"
            lines.append(f"| `{c.name}` | {c.type} | {desc} |")
        if self.examples:
            lines.append("示例问题：")
            for e in self.examples:
                lines.append(f"- {e}")
        return "\n".join(lines)


class SchemaProvider:
    def __init__(self, tables: list[TableSpec]):
        self.tables = tables
        self._by_name = {t.name: t for t in tables}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SchemaProvider":
        p = Path(path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        tables_raw = raw.get("tables", [])
        tables: list[TableSpec] = []
        for t in tables_raw:
            cols = [
                ColumnSpec(
                    name=c["name"],
                    type=c.get("type", ""),
                    description=c.get("description", ""),
                    enum=c.get("enum", []) or [],
                    nullable=c.get("nullable", True),
                )
                for c in (t.get("columns") or [])
            ]
            tables.append(
                TableSpec(
                    name=t["name"],
                    description=t.get("description", ""),
                    columns=cols,
                    examples=t.get("examples", []) or [],
                    primary_keys=t.get("primary_keys", []) or [],
                )
            )
        return cls(tables)

    def get(self, name: str) -> TableSpec | None:
        return self._by_name.get(name)

    def render_summary(self, focus: list[str] | None = None) -> str:
        tables = [t for t in self.tables if not focus or t.name in focus]
        if not tables:
            return "（没有可用表）"
        # Compact summary: one line per table with column names + descriptions truncated.
        lines: list[str] = ["# 数据表目录"]
        for t in tables:
            col_brief = ", ".join(f"{c.name}({c.type or '?'})" for c in t.columns[:12])
            if len(t.columns) > 12:
                col_brief += f", ... +{len(t.columns) - 12}"
            desc = t.description or ""
            lines.append(f"- **{t.name}** — {desc} | 列: {col_brief}")
        return "\n".join(lines)

    def describe_table(self, name: str) -> str:
        t = self.get(name)
        if not t:
            return f"[错误] 表 {name} 不在 schema 中"
        return t.render()

    def all_names(self) -> list[str]:
        return [t.name for t in self.tables]
