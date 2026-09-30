"""Query-time retrieval: embed the question, search the store."""

from reglens.models import ScoredChunk
from reglens.retrieval.embeddings import Embedder
from reglens.retrieval.vector_store import QdrantStore


class Retriever:
    def __init__(self, embedder: Embedder, store: QdrantStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        return self._store.search(self._embedder.embed_query(question), limit=k)
