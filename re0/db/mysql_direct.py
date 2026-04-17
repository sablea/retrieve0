"""MySQL direct executor via pymysql."""
from __future__ import annotations

import re
from typing import Any

from re0.core.config import MysqlConfig
from re0.db.base import SqlResult, validate_read_only


_HAS_LIMIT = re.compile(r"\blimit\b", re.IGNORECASE)


class MysqlDirectExecutor:
    def __init__(self, cfg: MysqlConfig, max_rows: int = 500, query_timeout_s: float = 30.0):
        self.cfg = cfg
        self.max_rows = max_rows
        self.query_timeout_s = query_timeout_s

    def _connect(self):
        import pymysql

        return pymysql.connect(
            host=self.cfg.host,
            port=self.cfg.port,
            user=self.cfg.user,
            password=self.cfg.password or "",
            database=self.cfg.database,
            charset=self.cfg.charset,
            connect_timeout=self.cfg.connect_timeout,
            read_timeout=int(self.query_timeout_s),
            autocommit=True,
            cursorclass=__import__("pymysql").cursors.DictCursor,
        )

    def execute(self, sql: str, params: dict[str, Any] | None = None, limit: int | None = None) -> SqlResult:
        clean = validate_read_only(sql)
        eff_limit = min(limit or self.max_rows, self.max_rows)
        if clean.lower().startswith("select") and not _HAS_LIMIT.search(clean):
            clean = f"{clean} LIMIT {eff_limit + 1}"

        with self._connect() as conn:
            if self.cfg.readonly:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SET SESSION TRANSACTION READ ONLY")
                except Exception:  # noqa: BLE001 - not all servers support it
                    pass
            with conn.cursor() as cur:
                cur.execute(clean, params or ())
                rows = cur.fetchall() or []
                columns = [d[0] for d in cur.description] if cur.description else []

        truncated = len(rows) > eff_limit
        if truncated:
            rows = rows[:eff_limit]
        return SqlResult(columns=columns, rows=list(rows), row_count=len(rows), truncated=truncated)

    def describe(self) -> str:
        return f"mysql://{self.cfg.user}@{self.cfg.host}:{self.cfg.port}/{self.cfg.database}"

    def healthcheck(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception:
            return False
