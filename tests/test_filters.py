"""Metadata filters (issuer, language, publication date) and page links.

All documents here are FICTIONAL.
"""

from datetime import date
from pathlib import Path

import pymupdf
import pytest

from reglens.cache import cache_key
from reglens.generation.citations import page_link
from reglens.generation.llm import FakeLLM
from reglens.ingestion.metadata import write_sidecar
from reglens.ingestion.pipeline import ingest
from reglens.models import DocumentMetadata
from reglens.rag import Answer, RagPipeline
from reglens.retrieval.embeddings import HashEmbedder
from reglens.retrieval.retriever import Retriever
from reglens.retrieval.vector_store import QdrantStore, SearchFilter, make_client

TEXT = "Article 1 Le délai FICTIONAL de réponse aux réclamations est de {days} jours."
DOCS = [  # (file, issuer, language, published, days)
    ("bam/fictional-2008.pdf", "BAM", "fr", date(2008, 5, 2), 11),
    ("bam/fictional-2021.pdf", "BAM", "fr", date(2021, 9, 30), 22),
    ("ammc/fictional-2019.pdf", "AMMC", "ar", date(2019, 1, 15), 33),
]
QUESTION = "Quel est le délai FICTIONAL de réponse aux réclamations ?"


@pytest.fixture()
def store(tmp_path: Path) -> QdrantStore:
    for file, issuer, language, published, days in DOCS:
        path = tmp_path / file
        path.parent.mkdir(exist_ok=True)
        doc = pymupdf.open()
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), TEXT.format(days=days))
        doc.save(path)
        write_sidecar(
            path,
            DocumentMetadata(
                title=f"Texte {file} FICTIONAL",
                issuer=issuer,  # type: ignore[arg-type]
                url=f"https://example.test/{file}",
                language=language,  # type: ignore[arg-type]
                published=published,
            ),
        )
    qdrant = QdrantStore(make_client(None, Path(":memory:")), "filters")
    ingest(tmp_path, HashEmbedder(), qdrant, 1200, 200, strategy="legal")
    return qdrant


def _files(store: QdrantStore, where: SearchFilter | None) -> set[str]:
    found = Retriever(HashEmbedder(), store).retrieve(QUESTION, k=10, where=where)
    return {s.chunk.file for s in found}


def test_no_filter_searches_everything(store: QdrantStore) -> None:
    assert len(_files(store, None)) == 3
    assert len(_files(store, SearchFilter())) == 3


def test_issuer_language_and_date_filters(store: QdrantStore) -> None:
    assert _files(store, SearchFilter(issuers=("AMMC",))) == {"ammc/fictional-2019.pdf"}
    assert _files(store, SearchFilter(languages=("fr",))) == {
        "bam/fictional-2008.pdf",
        "bam/fictional-2021.pdf",
    }
    assert _files(store, SearchFilter(date_from=date(2019, 1, 15))) == {
        "ammc/fictional-2019.pdf",  # bounds are inclusive
        "bam/fictional-2021.pdf",
    }
    assert _files(store, SearchFilter(date_to=date(2010, 12, 31))) == {"bam/fictional-2008.pdf"}
    both = SearchFilter(issuers=("BAM",), date_from=date(2015, 1, 1), date_to=date(2024, 12, 31))
    assert _files(store, both) == {"bam/fictional-2021.pdf"}


def test_hybrid_search_applies_the_filter_to_both_vectors(store: QdrantStore) -> None:
    retriever = Retriever(HashEmbedder(), store, "hybrid")
    found = retriever.retrieve(QUESTION, k=10, where=SearchFilter(issuers=("AMMC",)))
    assert {s.chunk.file for s in found} == {"ammc/fictional-2019.pdf"}


def test_nothing_matching_the_filter_means_not_found(store: QdrantStore) -> None:
    rag = RagPipeline(Retriever(HashEmbedder(), store), FakeLLM("Réponse FICTIONAL [1]."), k=3)
    answer = rag.answer(QUESTION, SearchFilter(issuers=("ACAPS",)))
    assert answer.abstention == "low_score" and answer.sources == []
    streamed = list(rag.stream(QUESTION, SearchFilter(date_to=date(2000, 1, 1))))
    assert isinstance(streamed[-1], Answer) and streamed[-1].abstention == "low_score"


def test_summary_counts_documents_issuers_and_years(store: QdrantStore) -> None:
    summary = store.summary()
    assert summary.documents == 3
    assert summary.issuers == ["AMMC", "BAM"]
    assert (summary.first_year, summary.last_year) == (2008, 2021)


def test_filter_key_is_stable_and_separates_cache_entries() -> None:
    a = SearchFilter(issuers=("BAM", "AMMC"), date_from=date(2020, 1, 1))
    b = SearchFilter(issuers=("AMMC", "BAM"), date_from=date(2020, 1, 1))
    assert a.key() == b.key() != ""
    assert SearchFilter().key() == ""
    assert cache_key("Q ?", 6) == cache_key("q  ?", 6, "")  # unfiltered keys are unchanged
    assert cache_key("Q ?", 6) != cache_key("Q ?", 6, a.key())


def test_page_link_encodes_the_url_and_targets_the_page() -> None:
    raw = "https://bank.test/download/Circulaire n° 9Z2099 FICTIONAL.pdf"
    assert page_link(raw, 4) == (
        "https://bank.test/download/Circulaire%20n%C2%B0%209Z2099%20FICTIONAL.pdf#page=4"
    )
    encoded = "https://bank.test/files/Note%20FICTIONAL.pdf#page=1"
    assert page_link(encoded, 7) == "https://bank.test/files/Note%20FICTIONAL.pdf#page=7"


def test_api_filters_and_page_links(tmp_path: Path) -> None:
    from tests.test_api import USER, _client

    with _client(tmp_path) as client:
        headers = {"X-API-Key": USER}
        question = "Quel est le ratio de liquidité ?"
        filtered = client.post(
            "/v1/query", json={"question": question, "issuers": ["AMMC"]}, headers=headers
        ).json()
        assert filtered["status"] == "not_found"  # the only document is a BAM text
        answered = client.post("/v1/query", json={"question": question}, headers=headers).json()
        assert answered["status"] == "answered" and answered["cached"] is False
        source = answered["sources"][0]
        assert source["link"] == f"{source['url']}#page={source['page_start']}"
        assert answered["citations"][0]["link"].endswith("#page=1")
        bad = client.post(
            "/v1/query",
            json={"question": question, "date_from": "2024-01-01", "date_to": "2020-01-01"},
            headers=headers,
        )
        assert bad.status_code == 422
