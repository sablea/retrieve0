"""Local sentence-transformers embedder (lazy, offline-only)."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

from re0.core.config import EmbeddingConfig


class Embedder:
    """Thin wrapper around sentence-transformers with offline guarantees."""

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            model_path = Path(self.cfg.model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"embedding model not found at {model_path}. "
                    "在内网部署时请将模型权重预先下载到该目录。"
                )
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                str(model_path),
                device=self.cfg.device,
                local_files_only=True,
            )
            self._model.max_seq_length = self.cfg.max_seq_length
            return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        model = self._load()
        arr = model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return arr.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Inputs assumed normalized.
        if a.ndim == 1:
            a = a[None, :]
        if b.ndim == 1:
            b = b[None, :]
        return a @ b.T


_GLOBAL_EMBEDDER: Embedder | None = None
_GLOBAL_LOCK = threading.Lock()


def get_embedder(cfg: EmbeddingConfig) -> Embedder:
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is not None:
        return _GLOBAL_EMBEDDER
    with _GLOBAL_LOCK:
        if _GLOBAL_EMBEDDER is None:
            _GLOBAL_EMBEDDER = Embedder(cfg)
    return _GLOBAL_EMBEDDER
