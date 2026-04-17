"""SQL executor abstraction + read-only safety check."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from re0.core.config import DBConfig


_READ_PREFIXES = ("select", "show", "desc", "describe", "explain", "with")
# Only applied to the first token of the first statement — we trust read prefixes
# and detect extra statements separately, so string literals inside a SELECT won't trigger.
_STMT_SPLIT = re.compile(r";\s*\S")


class ReadOnlyViolation(ValueError):
    pass


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"#[^\n]*", " ", sql)
    return sql.strip()


def validate_read_only(sql: str) -> str:
    cleaned = _strip_sql_comments(sql).strip().rstrip(";").strip()
    if not cleaned:
        raise ReadOnlyViolation("empty SQL")
    if _STMT_SPLIT.search(cleaned):
        raise ReadOnlyViolation("multiple statements are not allowed")
    lowered = cleaned.lower()
    if not lowered.startswith(_READ_PREFIXES):
        raise ReadOnlyViolation(
            f"only SELECT/SHOW/DESC/EXPLAIN/WITH allowed, got: {lowered.split()[0]}"
        )
    return cleaned


@dataclass
class SqlResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
        }


class SqlExecutor(Protocol):
    def execute(self, sql: str, params: dict[str, Any] | None = None, limit: int | None = None) -> SqlResult: ...
    def describe(self) -> str: ...
    def healthcheck(self) -> bool: ...


def build_executor(cfg: DBConfig) -> SqlExecutor:
    if cfg.type == "mysql":
        from re0.db.mysql_direct import MysqlDirectExecutor
        assert cfg.mysql is not None
        return MysqlDirectExecutor(cfg.mysql, max_rows=cfg.max_rows, query_timeout_s=cfg.query_timeout_s)
    if cfg.type == "http":
        from re0.db.http_executor import HttpExecutor
        assert cfg.http is not None
        return HttpExecutor(cfg.http, max_rows=cfg.max_rows)
    raise ValueError(f"unknown db.type: {cfg.type}")
