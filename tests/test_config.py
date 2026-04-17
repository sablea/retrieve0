"""Config loading smoke tests."""
from __future__ import annotations

import os

from re0.core.config import load_config


def test_load_example_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_TOKEN", "tok")
    monkeypatch.setenv("DB_PASSWORD", "pwd")
    cfg = load_config("config/re0.example.yaml")
    assert cfg.llm.provider == "vllm"
    assert cfg.llm.api_key == "tok"
    assert cfg.db.type == "mysql"
    assert cfg.db.mysql and cfg.db.mysql.password == "pwd"
    assert cfg.runtime.offline_mode is True
    # Offline env vars should be set by loader.
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
