"""Query-time retrieval: dense, sparse (BM25) or hybrid search, then optional reranking."""

from reglens.models import ScoredChunk
from reglens.retrieval.embeddings import Embedder
from reglens.retrieval.reranker import Reranker, rerank
from reglens.retrieval.sparse import BM25Encoder
from reglens.retrieval.vector_store import QdrantStore, SearchMode
from reglens.text import detect_language


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: QdrantStore,
        mode: SearchMode = "dense",
        reranker: Reranker | None = None,
        candidates: int = 20,
        rerank_languages: frozenset[str] = frozenset(),
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._mode = mode
        self._reranker = reranker
        self._candidates = candidates
        self._rerank_languages = rerank_languages  # empty: rerank every question

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def store(self) -> QdrantStore:
        return self._store

    def reranks(self, question: str) -> bool:
        """Whether this question goes through the reranker (language gate)."""
        if self._reranker is None:
            return False
        return not self._rerank_languages or detect_language(question) in self._rerank_languages

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        sparse = BM25Encoder.encode_query(question) if self._mode != "dense" else None
        vector = self._embedder.embed_query(question) if self._mode != "sparse" else []
        use_reranker = self.reranks(question)
        depth = max(k, self._candidates) if use_reranker else k
        found = self._store.search(vector, limit=depth, sparse=sparse, mode=self._mode)
        if not use_reranker or self._reranker is None:
            return found
        return rerank(self._reranker, question, found, k)
