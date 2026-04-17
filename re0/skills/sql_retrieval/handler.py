"""Core SQL retrieval skill."""
from __future__ import annotations

import json
from typing import Any

from re0.core.context import RunContext
from re0.core.skill import ToolSpec
from re0.db.base import ReadOnlyViolation


SYSTEM_PROMPT_HEADER = """\
你专精于把制造业自然语言问题翻译成 MySQL 只读查询并解释结果。

工作流程：
1. 读取用户问题，结合下方 schema 摘要与术语提示判断信息是否完整；
2. 必要时先 `recall_memory` 看是否有相似问题的缓存；命中可直接复用；
3. 信息不足 → 调用 `ask_user` 让用户澄清（例如同名产线/时间范围/口径）；
4. 信息充分 → 生成一条 SELECT SQL，调用 `execute_sql` 执行；
5. 根据结果用中文简明回答用户；成功且问题有复用价值时调用 `save_memory`。

硬性约束：
- 只允许 SELECT/SHOW/DESC/EXPLAIN/WITH；禁止 DDL/DML。
- 默认带 LIMIT，行数超过 50 时先汇总再呈现样例。
- 如果用户纠正先前结论（例如"不是 A01 是 A02"），重新推理并重新执行查询，不要沿用错误结果。
"""


class SqlRetrievalSkill:
    name = "sql_retrieval"
    description = "工业制造数据 MySQL 检索"

    def system_prompt(self, ctx: RunContext) -> str:
        parts = [SYSTEM_PROMPT_HEADER, "", "## 可用数据表", ctx.schema.render_summary()]
        return "\n".join(parts)

    def on_user_message(self, ctx: RunContext, message: str) -> str | None:
        hits = ctx.glossary.render_matches(message, top_k=5, threshold=0.4)
        return hits or None

    def tools(self, ctx: RunContext) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="inspect_table",
                description="查看指定数据表的详细 schema（列名、类型、描述、示例问题）。当 schema 摘要不够用时调用。",
                parameters={
                    "type": "object",
                    "properties": {"table": {"type": "string", "description": "表名"}},
                    "required": ["table"],
                },
                handler=self._inspect_table,
            ),
            ToolSpec(
                name="glossary_lookup",
                description="语义检索工业术语表，返回与查询相关的术语定义、相关表/列。",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "要解释的术语或相关问题片段"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
                handler=self._glossary_lookup,
            ),
            ToolSpec(
                name="execute_sql",
                description="执行只读 SQL（SELECT/SHOW/DESC）。返回列名、行数、前若干行数据。默认最多 500 行。",
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "只读 SQL 语句"},
                        "limit": {"type": "integer", "description": "最多返回行数，默认 100"},
                    },
                    "required": ["sql"],
                },
                handler=self._execute_sql,
            ),
            ToolSpec(
                name="recall_memory",
                description="按语义检索历史缓存，命中时返回历史 question/sql/answer。用于避免重复生成 SQL。",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "threshold": {"type": "number", "default": 0.85},
                    },
                    "required": ["query"],
                },
                handler=self._recall_memory,
            ),
            ToolSpec(
                name="save_memory",
                description="保存一条有价值的问题-SQL-结果映射，供以后语义召回。请仅在 SQL 成功并已确认结果正确时调用。",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "sql": {"type": "string"},
                        "answer": {"type": "string", "description": "面向用户的简要结论"},
                    },
                    "required": ["question", "sql", "answer"],
                },
                handler=self._save_memory,
            ),
            ToolSpec(
                name="correct_memory",
                description="用户指出先前结论错误时调用，修正或删除已保存的记忆。action=update 时需要新 sql/answer；action=delete 时只需 id。",
                parameters={
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "action": {"type": "string", "enum": ["update", "delete"]},
                        "new_sql": {"type": "string"},
                        "new_answer": {"type": "string"},
                    },
                    "required": ["id", "action"],
                },
                handler=self._correct_memory,
            ),
        ]

    # --- handlers ---

    def _inspect_table(self, ctx: RunContext, args: dict[str, Any]) -> str:
        return ctx.schema.describe_table(args["table"])

    def _glossary_lookup(self, ctx: RunContext, args: dict[str, Any]) -> str:
        top_k = int(args.get("top_k") or 5)
        matches = ctx.glossary.lookup(args["query"], top_k=top_k, threshold=0.0)
        if not matches:
            return "（未找到相关术语）"
        return "\n".join(m.render() for m in matches)

    def _execute_sql(self, ctx: RunContext, args: dict[str, Any]) -> str:
        sql = args["sql"]
        limit = args.get("limit")
        try:
            limit = int(limit) if limit is not None else None
        except (TypeError, ValueError):
            limit = None
        try:
            res = ctx.db.execute(sql, limit=limit)
        except ReadOnlyViolation as e:
            return f"[只读校验失败] {e}"
        except Exception as e:  # noqa: BLE001
            return f"[SQL 执行错误] {e}"
        preview = res.rows[:20]
        payload = {
            "columns": res.columns,
            "row_count": res.row_count,
            "truncated": res.truncated,
            "rows_preview": preview,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _recall_memory(self, ctx: RunContext, args: dict[str, Any]) -> str:
        thr = float(args.get("threshold", ctx.memory.recall_threshold))
        hits = ctx.memory.recall(args["query"], threshold=thr, top_k=3)
        if not hits:
            return "（无相似历史记忆）"
        return json.dumps([h.to_dict() for h in hits], ensure_ascii=False, default=str)

    def _save_memory(self, ctx: RunContext, args: dict[str, Any]) -> str:
        rec = ctx.memory.save(
            question=args["question"],
            sql=args.get("sql", ""),
            answer=args.get("answer", ""),
        )
        return f"已保存记忆 id={rec.id}"

    def _correct_memory(self, ctx: RunContext, args: dict[str, Any]) -> str:
        mid = int(args["id"])
        action = args["action"]
        if action == "delete":
            ok = ctx.memory.delete(mid)
            return f"删除 id={mid}: {'ok' if ok else 'not found'}"
        if action == "update":
            ok = ctx.memory.correct(mid, new_sql=args.get("new_sql"), new_answer=args.get("new_answer"))
            return f"更新 id={mid}: {'ok' if ok else 'not found'}"
        return f"[错误] 未知 action: {action}"
