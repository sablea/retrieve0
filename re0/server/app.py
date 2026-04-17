"""FastAPI HTTP server."""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from re0.core.agent import AgentResult
from re0.core.config import load_config
from re0.core.context import RunContext
from re0.core.session import SessionStore
from re0.runtime import Runtime, build_runtime


log = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话 id，多轮对话复用")
    message: str | None = Field(None, description="新的用户消息")
    pending_id: str | None = Field(None, description="上一轮 ask_user 返回的 pending_tool_call_id")
    answer: str | None = Field(None, description="对 pending 问题的回答")


class ChatResponse(BaseModel):
    status: str
    session_id: str
    reply: str | None = None
    pending_id: str | None = None
    pending_question: str | None = None
    hit_cache: bool = False
    tool_events: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class MemorySave(BaseModel):
    question: str
    sql: str = ""
    answer: str = ""


class MemoryCorrect(BaseModel):
    new_sql: str | None = None
    new_answer: str | None = None


def _try_cache_hit(rt: Runtime, session_id: str, message: str) -> ChatResponse | None:
    hits = rt.memory.recall(message, top_k=1)
    if not hits:
        return None
    top = hits[0]
    rt.memory.touch_hit(top.id)
    state = rt.sessions.load(session_id)
    state.history.append({"role": "user", "content": message})
    state.history.append(
        {
            "role": "assistant",
            "content": f"[命中缓存 id={top.id}] {top.answer}",
        }
    )
    rt.sessions.save(state)
    return ChatResponse(
        status="final",
        session_id=session_id,
        reply=top.answer,
        hit_cache=True,
        tool_events=[{"type": "cache_hit", "id": top.id, "similarity": top.similarity}],
    )


def create_app(config_path: str | None = None) -> FastAPI:
    cfg_path = config_path or os.getenv("RE0_CONFIG", "config/re0.yaml")
    cfg = load_config(cfg_path)
    rt: Runtime = build_runtime(cfg)
    app = FastAPI(title="re0", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        db_ok = True
        try:
            db_ok = rt.db.healthcheck()
        except Exception:
            db_ok = False
        return {
            "llm_base_url": cfg.llm.base_url,
            "llm_ok": rt.llm.healthcheck(),
            "db": rt.db.describe(),
            "db_ok": db_ok,
            "offline_mode": cfg.runtime.offline_mode,
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        if not req.message and not (req.pending_id and req.answer):
            raise HTTPException(400, "provide either `message` or (`pending_id` + `answer`)")

        state = rt.sessions.load(req.session_id)

        # Cache hit only for fresh user messages and when not resuming.
        if req.message and not req.pending_id and not state.pending_tool_call_id:
            cached = _try_cache_hit(rt, req.session_id, req.message)
            if cached is not None:
                return cached

        ctx = RunContext(
            config=cfg,
            llm=rt.llm,
            db=rt.db,
            schema=rt.schema,
            glossary=rt.glossary,
            memory=rt.memory,
            session_id=req.session_id,
        )

        pending_answer = None
        if req.pending_id:
            if state.pending_tool_call_id != req.pending_id:
                raise HTTPException(
                    409,
                    f"pending_id mismatch; current pending={state.pending_tool_call_id!r}",
                )
            pending_answer = req.answer or ""

        result: AgentResult = rt.agent.run(
            ctx, state, user_message=req.message, pending_answer=pending_answer
        )
        return ChatResponse(
            status=result.status,
            session_id=req.session_id,
            reply=result.reply if result.status == "final" else None,
            pending_id=result.pending_tool_call_id,
            pending_question=result.pending_question,
            tool_events=result.tool_events,
            error=result.error,
        )

    @app.get("/memory")
    def list_memory(query: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return [r.to_dict() for r in rt.memory.list(query=query, limit=limit)]

    @app.post("/memory")
    def save_memory(m: MemorySave) -> dict[str, Any]:
        return rt.memory.save(m.question, m.sql, m.answer).to_dict()

    @app.post("/memory/{mid}/correct")
    def correct_memory(mid: int, m: MemoryCorrect) -> dict[str, Any]:
        ok = rt.memory.correct(mid, new_sql=m.new_sql, new_answer=m.new_answer)
        if not ok:
            raise HTTPException(404, "memory not found")
        return {"ok": True, "id": mid}

    @app.delete("/memory/{mid}")
    def delete_memory(mid: int) -> dict[str, Any]:
        ok = rt.memory.delete(mid)
        if not ok:
            raise HTTPException(404, "memory not found")
        return {"ok": True, "id": mid}

    @app.delete("/session/{sid}")
    def delete_session(sid: str) -> dict[str, Any]:
        rt.sessions.delete(sid)
        return {"ok": True, "id": sid}

    app.state.runtime = rt
    return app


app = None  # lazy


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
