"""Qdrant vector store: one collection, a named dense vector, chunk metadata as payload.

The vector is named ("dense") so a sparse BM25 vector can be added alongside it for hybrid
search without changing the collection layout.
"""

from collections.abc import Sequence
from pathlib import Path

import structlog
from qdrant_client import QdrantClient, models

from reglens.models import Chunk, ScoredChunk

log = structlog.get_logger(__name__)

DENSE = "dense"


def make_client(url: str | None, path: Path, api_key: str | None = None) -> QdrantClient:
    """A Qdrant server when ``url`` is set, else an embedded on-disk store (":memory:" ok)."""
    if url:
        return QdrantClient(url=url, api_key=api_key)
    if str(path) == ":memory:":
        return QdrantClient(location=":memory:")
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    def recreate(self, dim: int) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            self._collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        )
        log.info("collection_created", collection=self._collection, dim=dim)

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        points = [
            models.PointStruct(
                id=chunk.id, vector={DENSE: list(vector)}, payload=chunk.model_dump(mode="json")
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(self._collection, points=points)

    def search(self, vector: Sequence[float], limit: int) -> list[ScoredChunk]:
        response = self._client.query_points(
            self._collection, query=list(vector), using=DENSE, limit=limit, with_payload=True
        )
        return [
            ScoredChunk(chunk=Chunk.model_validate(point.payload), score=point.score)
            for point in response.points
        ]

    def count(self) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        return self._client.count(self._collection).count
