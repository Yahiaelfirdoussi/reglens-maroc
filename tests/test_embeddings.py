"""Tests for embedders. The OpenAI API is mocked: no network, no key. Text is FICTIONAL."""

import json

import httpx
import pytest

from reglens.retrieval.embeddings import (
    HashEmbedder,
    OpenAIEmbedder,
    collection_suffix,
    make_embedder,
)


def _openai(handler: httpx.MockTransport, **kwargs: object) -> OpenAIEmbedder:
    client = httpx.Client(base_url="https://api.test/v1", transport=handler)
    return OpenAIEmbedder(
        "text-embedding-3-small", "sk-FICTIONAL", client=client, sleep=lambda _: None, **kwargs
    )


def test_openai_embedder_batches_and_keeps_input_order() -> None:
    calls: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-FICTIONAL"
        body = json.loads(request.content)
        assert body["model"] == "text-embedding-3-small"
        calls.append(body["input"])
        data = [
            {"index": i, "embedding": [float(len(text)), 0.0]}
            for i, text in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": data[::-1], "usage": {"total_tokens": 7}})

    embedder = _openai(httpx.MockTransport(handler), batch_size=2)
    assert embedder.dim == 1536
    vectors = embedder.embed_documents(["a", "bbb", "cc"])
    assert calls == [["a", "bbb"], ["cc"]]
    assert [v[0] for v in vectors] == [1.0, 3.0, 2.0]  # reordered by "index"
    assert embedder.tokens_used == 14
    assert embedder.name == "openai/text-embedding-3-small"


def test_openai_embedder_retries_rate_limits() -> None:
    responses = iter(
        [
            httpx.Response(429),
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}], "usage": {}}),
        ]
    )
    embedder = _openai(httpx.MockTransport(lambda request: next(responses)))
    assert embedder.embed_query("question FICTIONAL") == [1.0]


def test_openai_embedder_raises_on_client_errors() -> None:
    embedder = _openai(httpx.MockTransport(lambda request: httpx.Response(401)))
    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed_query("question FICTIONAL")


def test_openai_embedder_requires_a_key() -> None:
    with pytest.raises(ValueError, match="REGLENS_EMBEDDING_API_KEY"):
        make_embedder("openai/text-embedding-3-small", cache_dir=None, api_key=None)  # type: ignore[arg-type]


def test_make_embedder_and_collection_suffix() -> None:
    assert isinstance(make_embedder("hash", cache_dir=None), HashEmbedder)  # type: ignore[arg-type]
    assert collection_suffix("openai/text-embedding-3-small") == "openai-text-embedding-3-small"
    assert (
        collection_suffix("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        == "paraphrase-multilingual-minilm-l12-v2"
    )
