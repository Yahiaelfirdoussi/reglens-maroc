"""Query-time retrieval: dense, sparse (BM25) or hybrid search."""

from reglens.models import ScoredChunk
from reglens.retrieval.embeddings import Embedder
from reglens.retrieval.sparse import BM25Encoder
from reglens.retrieval.vector_store import QdrantStore, SearchMode


class Retriever:
    def __init__(self, embedder: Embedder, store: QdrantStore, mode: SearchMode = "dense") -> None:
        self._embedder = embedder
        self._store = store
        self._mode = mode

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        sparse = BM25Encoder.encode_query(question) if self._mode != "dense" else None
        vector = self._embedder.embed_query(question) if self._mode != "sparse" else []
        return self._store.search(vector, limit=k, sparse=sparse, mode=self._mode)
