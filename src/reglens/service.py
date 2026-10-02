"""Builds the answering pipeline from settings; shared by the CLI, the UI and the API."""

from dataclasses import dataclass, field
from pathlib import Path

from reglens.cache import TTLCache
from reglens.config import Settings, get_settings
from reglens.generation.llm import LiteLLMClient
from reglens.guardrails import ScopeClassifier
from reglens.ingestion.pipeline import ingest_document
from reglens.rag import Answer, RagPipeline
from reglens.retrieval.embeddings import make_embedder
from reglens.retrieval.reranker import make_reranker
from reglens.retrieval.retriever import Retriever
from reglens.retrieval.vector_store import QdrantStore, make_client


class EmptyIndexError(RuntimeError):
    pass


def build_retriever(settings: Settings | None = None) -> Retriever:
    settings = settings or get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    store = QdrantStore(
        make_client(settings.qdrant_url, settings.qdrant_path, api_key), settings.collection_name
    )
    if store.count() == 0:
        raise EmptyIndexError("The index is empty. Run `reglens ingest data/raw` first.")
    return Retriever(
        make_embedder(
            settings.embedding_model, settings.model_cache_dir, settings.embedder_api_key()
        ),
        store,
        settings.retrieval_mode,
        make_reranker(settings.reranker, settings.model_cache_dir, settings.rerank_max_chars),
        settings.rerank_candidates,
        settings.rerank_language_set,
    )


def build_pipeline(
    settings: Settings | None = None, k: int | None = None, warm: bool = False
) -> RagPipeline:
    """Guardrails, retrieval, abstention and the LLM (retrieval only when none is set).

    ``warm`` makes one tiny call to the embedding and LLM providers so the first real
    question does not pay for imports and connection setup (for long-running apps).
    """
    settings = settings or get_settings()
    llm = None
    if settings.llm_model:
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        llm = LiteLLMClient(
            settings.llm_model, key, settings.llm_timeout_s, settings.llm_reasoning_effort
        )
        if warm:
            llm.warm_up()
    retriever = build_retriever(settings)
    if warm:
        retriever.embedder.embed_query("warm-up")
    return RagPipeline(
        retriever,
        llm,
        k or settings.top_k,
        ScopeClassifier(retriever.embedder),
        settings.abstain_min_score,
    )


@dataclass
class Service:
    """Everything a long-running app needs: pipeline, index access and answer cache."""

    settings: Settings
    pipeline: RagPipeline
    cache: TTLCache[Answer] = field(init=False)

    def __post_init__(self) -> None:
        self.cache = TTLCache(self.settings.answer_cache_ttl_s, self.settings.answer_cache_size)

    @property
    def retriever(self) -> Retriever:
        return self.pipeline.retriever

    def ingest_file(self, pdf: Path, raw_dir: Path) -> tuple[str, int]:
        """Add or update one document, then drop cached answers (they may be stale)."""
        s = self.settings
        result = ingest_document(
            pdf,
            raw_dir,
            self.retriever.embedder,
            self.retriever.store,
            s.chunk_size,
            s.chunk_overlap,
            s.chunking,
            s.contextual_header,
        )
        self.cache.clear()
        return result


def build_service(settings: Settings | None = None, warm: bool = False) -> Service:
    settings = settings or get_settings()
    return Service(settings, build_pipeline(settings, warm=warm))
