"""Shared fakes for offline testing (no real LLM / no real embedding model)."""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np
import pytest

from re0.llm.base import LLMMessage, ToolCall


class FakeEmbedder:
    """Deterministic bag-of-words hashing -> normalized 64-dim vector."""

    DIM = 64

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        arr = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in (t or "").lower().replace("，", " ").replace(",", " ").split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                arr[i, h % self.DIM] += 1.0
            n = np.linalg.norm(arr[i])
            if n > 0:
                arr[i] /= n
        return arr

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class ScriptedLLM:
    """LLM that returns a pre-scripted sequence of messages."""

    model = "scripted"

    def __init__(self, script: list[LLMMessage]):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.calls.append({"messages": messages, "tools": tools})
        if not self._script:
            return LLMMessage(role="assistant", content="(no more scripted msgs)")
        return self._script.pop(0)

    def healthcheck(self) -> bool:
        return True


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


def msg_text(text: str) -> LLMMessage:
    return LLMMessage(role="assistant", content=text)


def msg_tool(call_id: str, name: str, args: dict[str, Any]) -> LLMMessage:
    return LLMMessage(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
    )
