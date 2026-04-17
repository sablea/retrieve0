"""ReAct tool-calling loop with pause/resume for human-in-the-loop."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from re0.core.context import RunContext
from re0.core.session import SessionState, SessionStore
from re0.core.skill import Skill, ToolSpec
from re0.llm.base import LLMProvider, ToolCall

log = logging.getLogger(__name__)


ASK_USER_TOOL_NAME = "ask_user"
FINISH_TOOL_NAME = "finish"


def _ask_user_tool() -> ToolSpec:
    def _handler(ctx: RunContext, args: dict[str, Any]) -> Any:  # pragma: no cover - never runs
        raise RuntimeError("ask_user is handled by Agent, not dispatched")

    return ToolSpec(
        name=ASK_USER_TOOL_NAME,
        description=(
            "向用户提问以获取缺失信息、澄清歧义或让用户做选择。"
            "调用后会话会暂停，等待用户回答再继续。"
            "只在确实缺少关键信息时调用，不要无谓打扰用户。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "要向用户提出的问题"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "可选：给用户的候选答案，帮助其快速选择",
                },
            },
            "required": ["question"],
        },
        handler=_handler,
    )


@dataclass
class AgentResult:
    status: str
    reply: str = ""
    pending_question: str | None = None
    pending_tool_call_id: str | None = None
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


class Agent:
    def __init__(
        self,
        llm: LLMProvider,
        skills: list[Skill],
        store: SessionStore,
        max_steps: int = 8,
    ):
        self.llm = llm
        self.skills = skills
        self.store = store
        self.max_steps = max_steps

    def _collect_tools(self, ctx: RunContext) -> tuple[list[dict[str, Any]], dict[str, ToolSpec]]:
        all_tools: list[ToolSpec] = [_ask_user_tool()]
        for s in self.skills:
            all_tools.extend(s.tools(ctx))
        index = {t.name: t for t in all_tools}
        return [t.to_openai() for t in all_tools], index

    def _build_system(self, ctx: RunContext, user_message: str | None) -> str:
        parts: list[str] = []
        parts.append(
            "你是面向工业制造数据的检索助手。遵循 ReAct 范式：先思考、再调用工具、"
            "观察结果、必要时再调用更多工具，直到可以回答。\n"
            "原则：\n"
            "- 当用户表述模糊或缺关键筛选条件时，调用 `ask_user` 提问而不是瞎猜。\n"
            "- 用户可能在任何时候纠正先前结论或补充信息，收到纠正后要重新规划并可能重新查询。\n"
            "- SQL 只允许 SELECT/SHOW/DESC。要尽量使用 LIMIT。\n"
            "- 所有中间结果都基于真实工具返回，禁止编造数据。\n"
        )
        for s in self.skills:
            parts.append(f"\n### Skill: {s.name}\n{s.system_prompt(ctx)}")
            if user_message is not None:
                extra = s.on_user_message(ctx, user_message)
                if extra:
                    parts.append(f"\n### Context from {s.name}\n{extra}")
        return "\n".join(parts)

    def _assistant_to_dict(self, content: str, tool_calls: list[ToolCall] | None) -> dict[str, Any]:
        d: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in tool_calls
            ]
        return d

    def run(
        self,
        ctx: RunContext,
        state: SessionState,
        user_message: str | None,
        pending_answer: str | None = None,
    ) -> AgentResult:
        """Run ReAct loop.

        - If `pending_answer` is set and state has `pending_tool_call_id`, resume.
        - Otherwise append `user_message` and start a new turn.
        """
        tools_openai, tool_index = self._collect_tools(ctx)

        # Ensure system message exists / is fresh (rebuild each turn to reflect new context).
        sys_msg = {"role": "system", "content": self._build_system(ctx, user_message)}
        if state.history and state.history[0].get("role") == "system":
            state.history[0] = sys_msg
        else:
            state.history.insert(0, sys_msg)

        events: list[dict[str, Any]] = []

        # Resume path.
        if pending_answer is not None and state.pending_tool_call_id:
            state.history.append(
                {
                    "role": "tool",
                    "tool_call_id": state.pending_tool_call_id,
                    "name": ASK_USER_TOOL_NAME,
                    "content": pending_answer,
                }
            )
            events.append({"type": "user_answer", "pending_id": state.pending_tool_call_id, "answer": pending_answer})
            state.pending_tool_call_id = None
            state.pending_question = None

        if user_message:
            state.history.append({"role": "user", "content": user_message})
            events.append({"type": "user_message", "text": user_message})

        for step in range(self.max_steps):
            try:
                msg = self.llm.chat(state.history, tools=tools_openai, tool_choice="auto")
            except Exception as e:  # noqa: BLE001
                log.exception("LLM call failed")
                state.history.append({"role": "assistant", "content": f"[LLM 调用失败] {e}"})
                self.store.save(state)
                return AgentResult(status="error", error=str(e), tool_events=events)

            state.history.append(self._assistant_to_dict(msg.content, msg.tool_calls))

            if not msg.tool_calls:
                self.store.save(state)
                return AgentResult(status="final", reply=msg.content, tool_events=events)

            # Check for ask_user first (pause & return).
            for tc in msg.tool_calls:
                if tc.name == ASK_USER_TOOL_NAME:
                    question = str(tc.arguments.get("question", "")).strip() or "需要更多信息"
                    options = tc.arguments.get("options")
                    if isinstance(options, list) and options:
                        question = question + "\n可选：" + "，".join(map(str, options))
                    state.pending_tool_call_id = tc.id
                    state.pending_question = question
                    events.append({"type": "ask_user", "question": question, "pending_id": tc.id})
                    # Any additional tool_calls in the same turn are ignored for simplicity;
                    # the model will see only the ask_user response next turn.
                    self.store.save(state)
                    return AgentResult(
                        status="need_user_input",
                        pending_question=question,
                        pending_tool_call_id=tc.id,
                        tool_events=events,
                    )

            # Dispatch all tool calls in order.
            for tc in msg.tool_calls:
                spec = tool_index.get(tc.name)
                if spec is None:
                    out = f"[错误] 未知工具: {tc.name}"
                else:
                    try:
                        result = spec.handler(ctx, tc.arguments)
                        out = result if isinstance(result, str) else json.dumps(
                            result, ensure_ascii=False, default=str
                        )
                    except Exception as e:  # noqa: BLE001
                        log.exception("tool %s failed", tc.name)
                        out = f"[工具执行错误] {e}"
                events.append({"type": "tool_call", "name": tc.name, "args": tc.arguments, "output": out[:2000]})
                state.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": out,
                    }
                )
                if tc.name == FINISH_TOOL_NAME:
                    self.store.save(state)
                    return AgentResult(status="final", reply=out, tool_events=events)

        self.store.save(state)
        return AgentResult(
            status="final",
            reply="[达到最大推理步数上限，已停止]",
            tool_events=events,
        )
