"""Text embedders behind a small protocol: FastEmbed (ONNX) in production, hashing in tests."""

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import structlog

log = structlog.get_logger(__name__)


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    """Local ONNX embedding model via FastEmbed (no PyTorch)."""

    def __init__(self, model: str, cache_dir: Path, batch_size: int = 32) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model, cache_dir=str(cache_dir))
        self._name = model
        self._batch_size = batch_size
        self._dim = len(next(iter(self._model.embed(["dimension probe"]))))
        tokenizer = getattr(self._model.model, "tokenizer", None)
        truncation = getattr(tokenizer, "truncation", None) or {}
        self.max_tokens: int | None = truncation.get("max_length")

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.embed(list(texts), batch_size=self._batch_size)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in next(iter(self._model.query_embed(text)))]

    def is_truncated(self, text: str) -> bool:
        """True when the model would silently drop the end of ``text``."""
        tokenizer = getattr(self._model.model, "tokenizer", None)
        if tokenizer is None:
            return False
        return bool(tokenizer.encode(text).overflowing)


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder: offline, for tests and CI."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"hash-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
            vector[int.from_bytes(digest[:4], "little") % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_embedder(model: str, cache_dir: Path) -> Embedder:
    if model == "hash":
        return HashEmbedder()
    return FastEmbedEmbedder(model, cache_dir)
