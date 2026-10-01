"""Tests for BM25 tokenisation, encoding and hybrid retrieval. Text here is FICTIONAL."""

from pathlib import Path

import pytest

from reglens.models import Chunk
from reglens.retrieval.embeddings import HashEmbedder
from reglens.retrieval.retriever import Retriever
from reglens.retrieval.sparse import BM25Encoder, reference_tokens, term_index, tokenize
from reglens.retrieval.vector_store import MissingSparseIndexError, QdrantStore, make_client


@pytest.mark.parametrize(
    "text", ["circulaire n°9/W/2098", "Circulaire 9W2098", "circ. 9/w/98", "la 9 / W / 98"]
)
def test_bam_reference_spellings_share_one_token(text: str) -> None:
    assert reference_tokens(text) == ["ref:9/w/2098"]


def test_ammc_reference_token() -> None:
    assert reference_tokens("Circulaire FICTIONAL n° 07/98") == ["ref:07-98"]


def test_tokenize_folds_accents_drops_stopwords_and_plurals() -> None:
    assert tokenize("Les pondérations des Risques") == ["ponderation", "risque"]
    assert "ref:9/w/2098" in tokenize("selon la circulaire 9/W/2098")


def test_bm25_document_weights_saturate_and_normalise_length() -> None:
    encoder = BM25Encoder(average_length=4)
    one = encoder.encode_document("ratio")
    three = encoder.encode_document("ratio ratio ratio")
    assert one.indices == three.indices == [term_index("ratio")]
    assert one.values[0] < three.values[0] < (encoder.k1 + 1)  # saturates below k1 + 1
    query = BM25Encoder.encode_query("ratio ratio liquidité")
    assert sorted(query.values) == [1.0, 1.0]


def _chunk(n: int, text: str, reference: str) -> Chunk:
    return Chunk(
        id=f"00000000-0000-0000-0000-{n:012d}",
        text=text,
        file=f"bam/{n}.pdf",
        index=0,
        page_start=1,
        page_end=1,
        issuer="BAM",
        reference=reference,
        title="FICTIONAL",
        url="https://example.test",
        section="Article 1",
    )


def test_hybrid_search_finds_exact_reference() -> None:
    from reglens.ingestion.pipeline import sparse_text

    chunks = [
        _chunk(1, "Règles FICTIONAL sur la gouvernance des banques.", "9/W/2098"),
        _chunk(2, "Règles FICTIONAL sur la liquidité des banques.", "8/W/2097"),
        _chunk(3, "Autres règles FICTIONAL sur la gouvernance.", "7/W/2096"),
    ]
    store = QdrantStore(make_client(None, Path(":memory:")), "hybrid")
    embedder = HashEmbedder()
    store.recreate(embedder.dim)
    texts = [sparse_text(c) for c in chunks]
    encoder = BM25Encoder.fit(texts)
    store.upsert(
        chunks,
        embedder.embed_documents([c.text for c in chunks]),
        [encoder.encode_document(t) for t in texts],
    )
    question = "Que prévoit la circulaire 9W2098 ?"  # spelling differs from the metadata
    for mode in ("sparse", "hybrid"):
        top = Retriever(embedder, store, mode).retrieve(question, k=1)[0]  # type: ignore[arg-type]
        assert top.chunk.reference == "9/W/2098"


def test_hybrid_search_requires_bm25_vectors() -> None:
    client = make_client(None, Path(":memory:"))
    from qdrant_client import models

    client.create_collection(
        "old",
        vectors_config={"dense": models.VectorParams(size=256, distance=models.Distance.COSINE)},
    )
    store = QdrantStore(client, "old")
    with pytest.raises(MissingSparseIndexError, match="run `reglens ingest` again"):
        Retriever(HashEmbedder(), store, "hybrid").retrieve("question FICTIONAL", k=1)
