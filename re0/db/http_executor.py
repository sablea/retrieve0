"""HTTP SQL proxy executor."""
from __future__ import annotations

from typing import Any

import httpx

from re0.core.config import HttpDbConfig
from re0.db.base import SqlResult, validate_read_only


class HttpExecutor:
    def __init__(self, cfg: HttpDbConfig, max_rows: int = 500):
        self.cfg = cfg
        self.max_rows = max_rows

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.auth_header:
            h[self.cfg.auth_header_name] = self.cfg.auth_header
        return h

    def execute(self, sql: str, params: dict[str, Any] | None = None, limit: int | None = None) -> SqlResult:
        clean = validate_read_only(sql)
        eff_limit = min(limit or self.max_rows, self.max_rows)
        payload = {"sql": clean, "params": params or {}, "limit": eff_limit}
        with httpx.Client(timeout=self.cfg.timeout_s) as client:
            if self.cfg.method == "POST":
                r = client.post(self.cfg.url, json=payload, headers=self._headers())
            else:
                r = client.get(self.cfg.url, params={"sql": clean, "limit": eff_limit}, headers=self._headers())
            r.raise_for_status()
            data = r.json()

        rows = data.get("rows") or data.get("data") or []
        columns = data.get("columns")
        if not columns and rows:
            columns = list(rows[0].keys())
        truncated = bool(data.get("truncated")) or len(rows) > eff_limit
        if len(rows) > eff_limit:
            rows = rows[:eff_limit]
        return SqlResult(columns=columns or [], rows=rows, row_count=len(rows), truncated=truncated)

    def describe(self) -> str:
        return f"http-sql:{self.cfg.url}"

    def healthcheck(self) -> bool:
        try:
            with httpx.Client(timeout=self.cfg.timeout_s) as client:
                r = client.request("OPTIONS", self.cfg.url, headers=self._headers())
            return r.status_code < 500
        except Exception:
            return False
