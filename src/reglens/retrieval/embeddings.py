"""Text embedders behind a small protocol: local FastEmbed (ONNX), the OpenAI API over
HTTP, and a hashing embedder for tests."""

import hashlib
import math
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, Protocol

import httpx
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


class OpenAIEmbedder:
    """OpenAI embeddings over plain HTTP (no SDK): batched, with retries and timeouts."""

    KNOWN_DIMS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        model: str,
        api_key: str,
        client: httpx.Client | None = None,
        batch_size: int = 128,
        retries: int = 6,  # backoff 1+2+...+32 s: rides out a short network outage
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OpenAI embeddings need an API key: set REGLENS_EMBEDDING_API_KEY in .env"
            )
        self._model = model
        self._client = client or httpx.Client(
            base_url="https://api.openai.com/v1", timeout=httpx.Timeout(60.0)
        )
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._batch_size = batch_size
        self._retries = retries
        self._sleep = sleep
        self.tokens_used = 0  # billed tokens, for cost reporting
        self._query_cache: dict[str, list[float]] = {}  # guardrails and retrieval share it
        self._dim = self.KNOWN_DIMS.get(model) or len(self._embed(["dimension probe"])[0])

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    @property
    def dim(self) -> int:
        return self._dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self._retries + 1):
            try:
                response = self._client.post(
                    "/embeddings",
                    json={"model": self._model, "input": texts},
                    headers=self._headers,
                )
                if response.status_code not in self.RETRY_STATUSES:
                    response.raise_for_status()
                    payload = response.json()
                    self.tokens_used += int(payload.get("usage", {}).get("total_tokens", 0))
                    rows = sorted(payload["data"], key=lambda row: row["index"])
                    return [[float(x) for x in row["embedding"]] for row in rows]
                error: Exception = httpx.HTTPStatusError(
                    f"status {response.status_code}", request=response.request, response=response
                )
            except httpx.TransportError as exc:
                error = exc
            if attempt == self._retries:
                raise error
            backoff = 2.0**attempt
            log.warning("embedding_retry", attempt=attempt + 1, backoff_s=backoff)
            self._sleep(backoff)
        raise AssertionError("unreachable")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed(list(texts[start : start + self._batch_size])))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a question once: the scope check and retrieval reuse the same vector."""
        if text not in self._query_cache:
            if len(self._query_cache) >= 256:
                self._query_cache.pop(next(iter(self._query_cache)))
            self._query_cache[text] = self._embed([text])[0]
        return self._query_cache[text]


def make_embedder(model: str, cache_dir: Path, api_key: str | None = None) -> Embedder:
    """``hash`` (tests), ``openai/<model>`` (API), or a FastEmbed model name (local)."""
    if model == "hash":
        return HashEmbedder()
    if model.startswith("openai/"):
        return OpenAIEmbedder(model.removeprefix("openai/"), api_key or "")
    return FastEmbedEmbedder(model, cache_dir)


def collection_suffix(model: str) -> str:
    """Index name part for an embedding model, so each model gets its own collection."""
    short = model.rsplit("/", 1)[-1] if not model.startswith("openai/") else model
    return re.sub(r"[^a-z0-9]+", "-", short.lower()).strip("-")
