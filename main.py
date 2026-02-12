#!/usr/bin/env python
import openai
import subprocess
import sys
import os
import json
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 初始化 OpenAI 客户端（从环境变量读取配置）
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")  # 如使用官方 OpenAI 可在 .env 中省略
)

# 定义工具（OpenAI 格式）
TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": """执行 shell 命令。模式：
- 读取: cat/grep/find/ls
- 写入: echo '...' > file
- 子代理: python v0_bash_agent.py 'task description'""",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"}
            },
            "required": ["command"]
        }
    }
}]

SYSTEM_PROMPT = f"CLI agent at {os.getcwd()}. Use bash. Spawn subagent for complex tasks."

def chat(prompt, history=[]):
    # 确保历史记录以系统消息开头
    if not history or history[0].get("role") != "system":
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    
    # 添加用户消息
    history.append({"role": "user", "content": prompt})
    
    while True:
        # 调用 OpenAI API
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",  # 或 "gpt-3.5-turbo"
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8000,
            temperature=0
        )
        
        message = response.choices[0].message
        
        # 保存助手回复到历史（包含可能的 tool_calls）
        msg_dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        history.append(msg_dict)
        
        # 检查是否需要工具调用
        if not message.tool_calls:
            return message.content or ""
        
        # 执行所有工具调用
        for tool_call in message.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
                command = args.get("command", "")
                
                # 执行 shell 命令
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                output = (result.stdout + result.stderr) or "命令执行完成（无输出）"
            except Exception as e:
                output = f"执行错误: {str(e)}"
            
            # 添加工具结果到历史
            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output
            })

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 子代理模式：单次执行
        history = []
        print(chat(sys.argv[1], history))
    else:
        # 交互模式
        history = []
        while (q := input(">> ").strip()) not in ("q", "quit", ""):
            result = chat(q, history)
            print(result)
            print()  # 空行分隔