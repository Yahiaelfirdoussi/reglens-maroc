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
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_cache_dir: Path = Path.home() / ".cache" / "reglens" / "models"

    # Fixed-size chunking (baseline). Sized to the embedding model's 128-token window.
    chunk_size: int = 450
    chunk_overlap: int = 90

    top_k: int = 6

    # LLM via LiteLLM, e.g. "gpt-4o-mini" or "mistral/mistral-small-latest". Empty: no LLM,
    # `ask` returns the retrieved sources only.
    llm_model: str = ""
    llm_api_key: SecretStr | None = None
    llm_timeout_s: float = 60.0

    @field_validator("data_dir", "qdrant_path", "model_cache_dir", mode="after")
    @classmethod
    def _expand_home(cls, value: Path) -> Path:
        return value.expanduser()


@lru_cache
def get_settings() -> Settings:
    return Settings()
