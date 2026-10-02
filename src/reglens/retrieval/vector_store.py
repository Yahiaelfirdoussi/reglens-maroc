"""Qdrant vector store: one collection, named dense and sparse (BM25) vectors per chunk.

Search modes: "dense" (embeddings), "sparse" (BM25 only, for diagnostics) and "hybrid"
(both, fused server-side with Reciprocal Rank Fusion).
"""

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import structlog
from qdrant_client import QdrantClient, models

from reglens.models import Chunk, ScoredChunk
from reglens.retrieval.sparse import SparseVector

log = structlog.get_logger(__name__)

DENSE = "dense"
SPARSE = "bm25"
SearchMode = Literal["dense", "sparse", "hybrid"]


class MissingSparseIndexError(RuntimeError):
    pass


def make_client(url: str | None, path: Path, api_key: str | None = None) -> QdrantClient:
    """A Qdrant server when ``url`` is set, else an embedded on-disk store (":memory:" ok)."""
    if url:
        return QdrantClient(url=url, api_key=api_key)
    if str(path) == ":memory:":
        return QdrantClient(location=":memory:")
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def _sparse(vector: SparseVector) -> models.SparseVector:
    return models.SparseVector(indices=vector.indices, values=vector.values)


class _SerializedClient:
    """Wraps a Qdrant client so calls from concurrent API requests take turns: the embedded
    (local) Qdrant is not built for concurrent access."""

    def __init__(self, client: QdrantClient) -> None:
        self._client = client
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._client, name)
        if not callable(attribute):
            return attribute

        def call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attribute(*args, **kwargs)

        return call


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client: Any = _SerializedClient(client)
        self._collection = collection

    def recreate(self, dim: int) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            self._collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        log.info("collection_created", collection=self._collection, dim=dim)

    def has_sparse(self) -> bool:
        info = self._client.get_collection(self._collection)
        return SPARSE in (info.config.params.sparse_vectors or {})

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        sparse: Sequence[SparseVector] | None = None,
        extra_payload: dict[str, object] | None = None,
    ) -> None:
        sparse_vectors = sparse if sparse is not None else [None] * len(chunks)
        points = []
        for chunk, vector, bm25 in zip(chunks, vectors, sparse_vectors, strict=True):
            named: dict[str, list[float] | models.SparseVector] = {DENSE: list(vector)}
            if bm25 is not None:
                named[SPARSE] = _sparse(bm25)
            payload = chunk.model_dump(mode="json") | (extra_payload or {})
            points.append(models.PointStruct(id=chunk.id, vector=named, payload=payload))
        self._client.upsert(self._collection, points=points)

    def search(
        self,
        vector: Sequence[float],
        limit: int,
        sparse: SparseVector | None = None,
        mode: SearchMode = "dense",
        candidates: int = 30,
    ) -> list[ScoredChunk]:
        """Top ``limit`` chunks; hybrid fuses the top ``candidates`` of each vector (RRF)."""
        if mode != "dense":
            if sparse is None:
                raise ValueError(f"{mode} search needs a sparse query vector")
            if not self.has_sparse():
                raise MissingSparseIndexError(
                    f"collection {self._collection} has no BM25 vectors: run `reglens ingest` again"
                )
        if mode == "dense":
            response = self._client.query_points(
                self._collection, query=list(vector), using=DENSE, limit=limit, with_payload=True
            )
        elif mode == "sparse":
            assert sparse is not None
            response = self._client.query_points(
                self._collection,
                query=_sparse(sparse),
                using=SPARSE,
                limit=limit,
                with_payload=True,
            )
        else:
            assert sparse is not None
            response = self._client.query_points(
                self._collection,
                prefetch=[
                    models.Prefetch(query=list(vector), using=DENSE, limit=candidates),
                    models.Prefetch(query=_sparse(sparse), using=SPARSE, limit=candidates),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        return [
            ScoredChunk(chunk=Chunk.model_validate(point.payload), score=point.score)
            for point in response.points
        ]

    def count(self) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        return int(self._client.count(self._collection).count)

    def exists(self) -> bool:
        return bool(self._client.collection_exists(self._collection))

    @staticmethod
    def _file_filter(file: str) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key="file", match=models.MatchValue(value=file))]
        )

    def any_payload(self) -> dict[str, object] | None:
        points, _ = self._client.scroll(self._collection, limit=1, with_payload=True)
        return dict(points[0].payload or {}) if points else None

    def file_state(self, file: str) -> dict[str, object] | None:
        """Payload of one indexed chunk of ``file`` (None when the file is not indexed)."""
        points, _ = self._client.scroll(
            self._collection, scroll_filter=self._file_filter(file), limit=1, with_payload=True
        )
        return dict(points[0].payload or {}) if points else None

    def delete_file(self, file: str) -> None:
        """Remove every chunk of one source file."""
        self._client.delete(
            self._collection, points_selector=models.FilterSelector(filter=self._file_filter(file))
        )
