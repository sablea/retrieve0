"""Configuration loading (yaml + env var expansion)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class LLMConfig(BaseModel):
    provider: str = "vllm"
    model: str
    base_url: str
    api_key_env: str | None = None
    api_key: str | None = None
    timeout_s: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 2048

    @model_validator(mode="after")
    def _resolve_api_key(self) -> "LLMConfig":
        if self.api_key_env:
            self.api_key = os.getenv(self.api_key_env, "") or None
        return self


class MysqlConfig(BaseModel):
    host: str
    port: int = 3306
    user: str
    password_env: str | None = None
    password: str | None = None
    database: str
    readonly: bool = True
    charset: str = "utf8mb4"
    connect_timeout: int = 5

    @model_validator(mode="after")
    def _resolve_pwd(self) -> "MysqlConfig":
        if self.password_env:
            self.password = os.getenv(self.password_env, "")
        return self


class HttpDbConfig(BaseModel):
    url: str
    auth_header_env: str | None = None
    auth_header_name: str = "Authorization"
    auth_header: str | None = None
    method: Literal["POST", "GET"] = "POST"
    timeout_s: float = 30.0

    @model_validator(mode="after")
    def _resolve_token(self) -> "HttpDbConfig":
        if self.auth_header_env:
            val = os.getenv(self.auth_header_env, "")
            self.auth_header = val or None
        return self


class DBConfig(BaseModel):
    type: Literal["mysql", "http"] = "mysql"
    mysql: MysqlConfig | None = None
    http: HttpDbConfig | None = None
    max_rows: int = 500
    query_timeout_s: float = 30.0

    @model_validator(mode="after")
    def _check(self) -> "DBConfig":
        if self.type == "mysql" and not self.mysql:
            raise ValueError("db.type=mysql requires db.mysql block")
        if self.type == "http" and not self.http:
            raise ValueError("db.type=http requires db.http block")
        return self


class KnowledgeConfig(BaseModel):
    schema_path: str
    glossary_path: str


class EmbeddingConfig(BaseModel):
    model_path: str
    cache_dir: str = ".re0_cache"
    device: str = "cpu"
    max_seq_length: int = 256


class MemoryConfig(BaseModel):
    path: str = ".re0_cache/memory.sqlite"
    recall_threshold: float = 0.85
    top_k: int = 1


class SessionConfig(BaseModel):
    path: str = ".re0_cache/sessions.sqlite"
    max_turns: int = 30


class RuntimeConfig(BaseModel):
    offline_mode: bool = True
    max_agent_steps: int = 8
    log_level: str = "INFO"


class AppConfig(BaseModel):
    llm: LLMConfig
    db: DBConfig
    knowledge: KnowledgeConfig
    embedding: EmbeddingConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    skills: dict[str, Any] = Field(default_factory=lambda: {"enabled": ["sql_retrieval"]})


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    return obj


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    raw = _expand_env(raw)
    cfg = AppConfig(**raw)

    if cfg.runtime.offline_mode:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return cfg
