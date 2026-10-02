"""Application settings, loaded from environment variables prefixed with ``REGLENS_``."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REGLENS_", env_file=".env", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    data_dir: Path = Path("data/raw")

    # Qdrant: a server when `qdrant_url` is set, otherwise embedded on-disk at `qdrant_path`.
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: Path = Path("data/qdrant")
    qdrant_collection: str = "reglens"

    # Embeddings (FastEmbed / ONNX). "hash" selects the offline test embedder.
    # "openai/<model>" uses the OpenAI API with `embedding_api_key`.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_api_key: SecretStr | None = None
    model_cache_dir: Path = Path.home() / ".cache" / "reglens" / "models"

    # Chunking: "fixed" windows (baseline) or "legal" (one chunk per article).
    # 450/90 was sized for the local model's 128-token window; API models read far more.
    chunking: Literal["fixed", "legal"] = "fixed"
    chunk_size: int = 450
    chunk_overlap: int = 90
    # Context prepended to each chunk before embedding: "off", "full"
    # (issuer | reference | title | section) or "ref" (issuer | reference | section).
    contextual_header: Literal["off", "full", "ref"] = "off"

    top_k: int = 6
    # "dense" (embeddings), "hybrid" (embeddings + BM25 fused with RRF) or "sparse" (BM25).
    retrieval_mode: Literal["dense", "hybrid", "sparse"] = "dense"
    # Cross-encoder reranking of the first-stage candidates; "" disables it.
    # e.g. "jinaai/jina-reranker-v2-base-multilingual" (local, ONNX).
    reranker: str = ""
    rerank_candidates: int = 20
    # Rerank only questions in these languages (comma-separated, e.g. "ar"); "" = all.
    rerank_languages: str = ""
    # Characters of each candidate the reranker reads (0 = all); shorter is faster.
    rerank_max_chars: int = 0

    @property
    def rerank_language_set(self) -> frozenset[str]:
        return frozenset(x.strip() for x in self.rerank_languages.split(",") if x.strip())

    # LLM via LiteLLM, e.g. "gpt-4o-mini" or "mistral/mistral-small-latest". Empty: no LLM,
    # `ask` returns the retrieved sources only.
    llm_model: str = ""
    llm_api_key: SecretStr | None = None
    llm_timeout_s: float = 60.0
    # Reasoning effort for reasoning models ("minimal" is fastest); "" = provider default.
    llm_reasoning_effort: str = ""
    # LLM-as-judge for evaluation (a different, stronger model than the answering one).
    judge_model: str = ""
    # Prices in USD per million tokens, for cost reporting (0 = unknown: tokens only).
    llm_price_in: float = 0.0
    llm_price_out: float = 0.0
    # --- API ---
    api_host: str = "127.0.0.1"  # 0.0.0.0 inside a container
    api_port: int = 8000
    # Comma-separated keys sent as X-API-Key. Admin keys may also ingest. No key configured
    # leaves the API open (local development only).
    api_keys: SecretStr | None = None
    admin_api_keys: SecretStr | None = None
    rate_limit_per_minute: int = 30  # per key (per client IP when the API is open)
    cors_origins: str = ""  # comma-separated allowed origins; "" = no cross-origin access
    answer_cache_ttl_s: int = 600  # 0 disables the answer cache
    answer_cache_size: int = 512
    upload_max_mb: int = 20
    warm_on_start: bool = True

    def keys(self, admin: bool = False) -> list[str]:
        secret = self.admin_api_keys if admin else self.api_keys
        raw = secret.get_secret_value() if secret else ""
        return [k.strip() for k in raw.split(",") if k.strip()]

    # Abstention floor: below this top retrieval score, answer "not found" without the LLM.
    # Calibrated for text-embedding-3-large (lowest answerable top score on the golden set:
    # 0.407, cross-lingual Arabic), so it only catches clearly unrelated questions.
    abstain_min_score: float = 0.35

    @property
    def collection_name(self) -> str:
        """One collection per embedding model, so indexes for different models coexist."""
        from reglens.retrieval.embeddings import collection_suffix

        name = f"{self.qdrant_collection}__{collection_suffix(self.embedding_model)}"
        if (self.chunking, self.chunk_size, self.chunk_overlap) != ("fixed", 450, 90):
            name += f"__{self.chunking}{self.chunk_size}-{self.chunk_overlap}"
        if self.contextual_header != "off":
            name += "__hdr" if self.contextual_header == "full" else "__hdrref"
        return name

    def embedder_api_key(self) -> str | None:
        return self.embedding_api_key.get_secret_value() if self.embedding_api_key else None

    @field_validator("data_dir", "qdrant_path", "model_cache_dir", mode="after")
    @classmethod
    def _expand_home(cls, value: Path) -> Path:
        return value.expanduser()


@lru_cache
def get_settings() -> Settings:
    return Settings()
