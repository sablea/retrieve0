"""OpenAI Chat Completions protocol adapter (targets internal vLLM)."""
from __future__ import annotations

import json
from typing import Any

import openai

from re0.core.config import LLMConfig
from re0.llm.base import LLMMessage, ToolCall


class OpenAICompatProvider:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.model = cfg.model
        self._client = openai.OpenAI(
            api_key=cfg.api_key or "EMPTY",
            base_url=cfg.base_url,
            timeout=cfg.timeout_s,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMMessage:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        calls: list[ToolCall] | None = None
        if msg.tool_calls:
            calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return LLMMessage(role="assistant", content=msg.content or "", tool_calls=calls)

    def healthcheck(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
