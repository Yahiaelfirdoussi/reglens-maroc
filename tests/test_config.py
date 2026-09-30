import pytest

from reglens.config import Settings


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.top_k == 6
    assert settings.qdrant_collection == "reglens"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REGLENS_TOP_K", "10")
    monkeypatch.setenv("REGLENS_LLM_API_KEY", "secret-value")
    settings = Settings(_env_file=None)
    assert settings.top_k == 10
    assert settings.llm_api_key is not None
    assert "secret-value" not in repr(settings)
