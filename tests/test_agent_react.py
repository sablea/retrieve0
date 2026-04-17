"""Tests for the ReAct agent: pause on ask_user, resume on next turn."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tests.conftest import ScriptedLLM, msg_text, msg_tool

from re0.core.agent import Agent
from re0.core.context import RunContext
from re0.core.session import SessionStore
from re0.core.skill import ToolSpec


class RecorderSkill:
    name = "recorder"
    description = "echoes tool calls into a list for assertions"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def system_prompt(self, ctx) -> str:
        return "test skill"

    def on_user_message(self, ctx, msg) -> str | None:
        return None

    def tools(self, ctx) -> list[ToolSpec]:
        def echo(ctx, args):
            self.calls.append(args)
            return f"echo:{args.get('x')}"

        return [
            ToolSpec(
                name="echo",
                description="echo",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
                handler=echo,
            )
        ]


def _ctx(llm) -> RunContext:
    # Only `llm` is used in this test; others are placeholders.
    return RunContext(
        config=None, llm=llm, db=None, schema=None, glossary=None, memory=None, session_id="t1"
    )


def test_agent_finishes_without_tools(tmp_path):
    llm = ScriptedLLM([msg_text("done")])
    store = SessionStore(tmp_path / "s.sqlite")
    skill = RecorderSkill()
    agent = Agent(llm, [skill], store)
    state = store.load("s1")
    r = agent.run(_ctx(llm), state, user_message="hi")
    assert r.status == "final" and r.reply == "done"


def test_agent_dispatches_tool_then_finishes(tmp_path):
    llm = ScriptedLLM(
        [
            msg_tool("c1", "echo", {"x": "hello"}),
            msg_text("finished with echo"),
        ]
    )
    store = SessionStore(tmp_path / "s.sqlite")
    skill = RecorderSkill()
    agent = Agent(llm, [skill], store)
    state = store.load("s1")
    r = agent.run(_ctx(llm), state, user_message="run echo")
    assert r.status == "final"
    assert skill.calls == [{"x": "hello"}]
    tool_events = [e for e in r.tool_events if e["type"] == "tool_call"]
    assert tool_events[0]["name"] == "echo"


def test_agent_pauses_on_ask_user_and_resumes(tmp_path):
    llm = ScriptedLLM(
        [
            msg_tool("c1", "ask_user", {"question": "选哪条线?", "options": ["A01", "A02"]}),
            # After resume, model chooses a tool then finishes.
            msg_tool("c2", "echo", {"x": "A02"}),
            msg_text("done A02"),
        ]
    )
    store = SessionStore(tmp_path / "s.sqlite")
    skill = RecorderSkill()
    agent = Agent(llm, [skill], store)

    state = store.load("s1")
    r1 = agent.run(_ctx(llm), state, user_message="昨天那条线的节拍")
    assert r1.status == "need_user_input"
    assert r1.pending_tool_call_id == "c1"
    assert "A01" in r1.pending_question and "A02" in r1.pending_question

    # Reload state and resume with the answer.
    state2 = store.load("s1")
    assert state2.pending_tool_call_id == "c1"
    r2 = agent.run(_ctx(llm), state2, user_message=None, pending_answer="A02")
    assert r2.status == "final" and "A02" in r2.reply
    assert skill.calls == [{"x": "A02"}]

    state3 = store.load("s1")
    assert state3.pending_tool_call_id is None


def test_agent_max_steps_cap(tmp_path):
    # Model keeps calling tool forever — ensure we stop.
    script = [msg_tool(f"c{i}", "echo", {"x": str(i)}) for i in range(20)]
    llm = ScriptedLLM(script)
    store = SessionStore(tmp_path / "s.sqlite")
    agent = Agent(llm, [RecorderSkill()], store, max_steps=3)
    state = store.load("s1")
    r = agent.run(_ctx(llm), state, user_message="loop")
    assert r.status == "final" and "最大推理步数" in r.reply
