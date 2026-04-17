"""Tests for glossary semantic lookup and memory recall."""
from __future__ import annotations

from pathlib import Path

import yaml

from re0.knowledge.glossary import Glossary, GlossaryEntry
from re0.memory.store import MemoryStore


def _entries():
    return [
        GlossaryEntry(term="节拍", aliases=["cycle time", "CT"], definition="单件产品工序平均时间"),
        GlossaryEntry(term="一次合格率", aliases=["FPY"], definition="一次通过质检比例"),
        GlossaryEntry(term="OEE", aliases=["设备综合效率"], definition="设备综合效率"),
    ]


def test_glossary_recalls_by_alias(tmp_path, fake_embedder):
    g = Glossary(_entries(), fake_embedder, cache_dir=tmp_path)
    hits = g.lookup("cycle time", top_k=3, threshold=0.0)
    assert hits and hits[0].term == "节拍"


def test_glossary_index_is_cached(tmp_path, fake_embedder):
    g1 = Glossary(_entries(), fake_embedder, cache_dir=tmp_path)
    g1.build_index()
    cache_files = list(Path(tmp_path).glob("glossary-*.npz"))
    assert cache_files, "glossary cache file should be created"

    g2 = Glossary(_entries(), fake_embedder, cache_dir=tmp_path)
    g2.build_index()
    assert g2._embeddings is not None


def test_memory_recall_hits_similar(tmp_path, fake_embedder):
    store = MemoryStore(tmp_path / "mem.sqlite", fake_embedder, recall_threshold=0.3)
    store.save("昨天 A01 线的平均节拍", sql="SELECT ...", answer="12.3s")
    hits = store.recall("昨天 A01 线 平均节拍", top_k=1)
    assert hits and "A01" in hits[0].question


def test_memory_recall_miss_unrelated(tmp_path, fake_embedder):
    store = MemoryStore(tmp_path / "mem.sqlite", fake_embedder, recall_threshold=0.6)
    store.save("昨天 A01 线的平均节拍", sql="SELECT ...", answer="12.3s")
    hits = store.recall("完全不相关 的 主题 banana", top_k=1)
    assert hits == []


def test_memory_correct_and_delete(tmp_path, fake_embedder):
    store = MemoryStore(tmp_path / "mem.sqlite", fake_embedder)
    rec = store.save("q1", "SELECT 1", "a1")
    assert store.correct(rec.id, new_answer="a2") is True
    assert store.list()[0].answer == "a2"
    assert store.delete(rec.id) is True
    assert store.list() == []
