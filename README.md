# retrieve0 (re0)

[![WIP](https://img.shields.io/badge/status-WIP-yellow?style=flat-square)](https://github.com/sablea/retrieve0)

**re0** 是一个面向制造业工业参数数据检索的智能体框架。自然语言进，只读 SQL 查询 + 结果解读出。

完全内网部署：LLM 走自托管 vLLM（OpenAI 兼容协议）、嵌入模型离线加载、所有外部依赖不触达公网。

## 能力一览

- **手写 schema**：yaml 描述库表/字段含义与取值枚举，注入 prompt 辅助 LLM 选表选列。
- **工业术语表 + 向量召回**：每轮对用户消息做语义检索，把相关术语（节拍/FPY/OEE…）定义与关联表列一起塞进上下文。
- **skill 框架**：`sql_retrieval` 内置；新功能（text2code/图表生成等）= 新增一个 skill 目录，核心框架不改。
- **LLM 可切换**：统一 OpenAI Chat Completions 协议，改一行 `llm.model` / `llm.base_url` 即可接入内网 vLLM 上任意模型。
- **数据库可切换**：`db.type: mysql` 直连或 `db.type: http` 通过外部 SQL 代理，接口一致。
- **记忆机制**：相似历史问题语义召回，命中即复用、跳过 LLM；支持 `save_memory` / `correct_memory`。
- **ReAct + Human-in-the-loop**：信息不足时智能体主动 `ask_user` 暂停等待；用户可中途纠错，Agent 根据历史重新规划。
- **安全护栏**：只允许 `SELECT/SHOW/DESC/EXPLAIN/WITH`，拒绝 DDL/DML、多语句，自动 `LIMIT`。

## 目录结构

```
re0/
├── core/        # agent ReAct 循环、skill 基类、会话持久化、配置加载
├── llm/         # OpenAI 兼容 adapter（内网 vLLM）
├── db/          # MySQL / HTTP 两种 SQL 执行器 + 只读白名单
├── knowledge/   # schema provider、glossary 向量检索、离线 embedder
├── memory/      # SQLite + embedding 召回
├── skills/      # sql_retrieval（内置）
└── server/      # FastAPI 应用
config/          # 示例 schema / glossary / 主配置
main.py          # CLI 入口（= re0.cli:main）
```

## 快速开始（内网）

```bash
pip install -r requirements.txt

# 1. 准备嵌入模型：由运维将权重目录同步到内网（如 bge-small-zh-v1.5）
#    路径填入 config/re0.yaml 的 embedding.model_path

cp config/re0.example.yaml config/re0.yaml
cp config/schema.example.yaml config/schema.yaml
cp config/glossary.example.yaml config/glossary.yaml
cp .env.example .env
# 按实际环境编辑 config/re0.yaml 与 .env（填 vLLM base_url、DB 凭据、embedding 路径等）
```

### CLI

```bash
python main.py --config config/re0.yaml "昨天 A01 线的平均节拍"
# 多轮模式
python main.py --config config/re0.yaml -v
```

### HTTP 服务

```bash
uvicorn --factory re0.server.app:create_app --host 0.0.0.0 --port 8000
# 或：RE0_CONFIG=config/re0.yaml uvicorn --factory re0.server.app:create_app
```

`POST /chat`：

```bash
curl -XPOST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"session_id":"t1","message":"昨天 A02 线的平均节拍"}'
```

三种返回状态：
- `status=final`：`reply` 即最终答复，`hit_cache` 指示是否来自记忆。
- `status=need_user_input`：`pending_id` + `pending_question`。将用户回答再次 POST：
  ```json
  {"session_id":"t1","pending_id":"<id>","answer":"A02 线"}
  ```
- `status=error`：`error` 字段为错误描述。

### 其他端点

- `GET /healthz` — 检查 vLLM / DB 连通性与离线模式。
- `GET /memory?query=...` — 列出 / 语义搜索记忆。
- `POST /memory` — 手工写入；`POST /memory/{id}/correct`、`DELETE /memory/{id}` — 纠正 / 删除。
- `DELETE /session/{id}` — 清空会话。

## 扩展：新增 skill

```
re0/skills/my_feature/
├── __init__.py
├── skill.yaml
└── handler.py   # 实现 Skill 协议：name/description/system_prompt/tools/on_user_message
```
然后在 `re0/skills/__init__.py:default_registry` 注册并在 `skills.enabled` 打开。

## 离线/安全说明

- 代码与示例不包含任何公网 URL；`runtime.offline_mode=true` 会强制 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`。
- SQL 执行前经过白名单 + 多语句检测；MySQL 会话开启 `SET SESSION TRANSACTION READ ONLY`（若支持）。
- 配置中的所有密码/Token 都从环境变量读取，不落盘。

## 开发

```bash
pytest tests/ -q
```

## Inspiration

Inspired by [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code).
