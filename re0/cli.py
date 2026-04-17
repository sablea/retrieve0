"""Terminal entrypoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid

from dotenv import load_dotenv

from re0.core.config import load_config
from re0.core.context import RunContext
from re0.runtime import build_runtime


def _print_events(events: list[dict], verbose: bool) -> None:
    if not verbose:
        return
    for e in events:
        t = e.get("type")
        if t == "tool_call":
            out = str(e.get("output", ""))[:300]
            print(f"  · {e['name']}({json.dumps(e.get('args', {}), ensure_ascii=False)}) -> {out}")
        elif t == "cache_hit":
            print(f"  · cache_hit id={e['id']} sim={e.get('similarity'):.3f}")
        elif t == "ask_user":
            print(f"  · ask_user: {e['question']}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="re0")
    parser.add_argument("--config", default=os.getenv("RE0_CONFIG", "config/re0.yaml"))
    parser.add_argument("--session", default=None, help="会话 id；不传则生成新的")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印工具调用过程")
    parser.add_argument("question", nargs="?", help="单轮模式：直接提问")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rt = build_runtime(cfg)
    session_id = args.session or f"cli-{uuid.uuid4().hex[:8]}"

    def _run(message: str | None, pending_id: str | None = None, pending_answer: str | None = None):
        state = rt.sessions.load(session_id)
        if message and not pending_id and not state.pending_tool_call_id:
            hits = rt.memory.recall(message, top_k=1)
            if hits:
                top = hits[0]
                rt.memory.touch_hit(top.id)
                print(f"[命中缓存 id={top.id}] {top.answer}")
                return None
        ctx = RunContext(
            config=cfg, llm=rt.llm, db=rt.db, schema=rt.schema,
            glossary=rt.glossary, memory=rt.memory, session_id=session_id,
        )
        result = rt.agent.run(ctx, state, user_message=message, pending_answer=pending_answer)
        _print_events(result.tool_events, args.verbose)
        if result.status == "final":
            print(result.reply)
        elif result.status == "need_user_input":
            pass  # handled by caller
        else:
            print(f"[错误] {result.error}", file=sys.stderr)
        return result

    if args.question:
        r = _run(args.question)
        while r and r.status == "need_user_input":
            print(f"\n需要澄清：{r.pending_question}")
            ans = input("你的回答> ").strip()
            r = _run(None, pending_id=r.pending_tool_call_id, pending_answer=ans)
        return 0

    print(f"re0 session={session_id}（输入 q 退出）")
    while True:
        try:
            q = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q in ("q", "quit", "exit", ""):
            break
        r = _run(q)
        while r and r.status == "need_user_input":
            print(f"\n需要澄清：{r.pending_question}")
            try:
                ans = input("你的回答> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            r = _run(None, pending_id=r.pending_tool_call_id, pending_answer=ans)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
