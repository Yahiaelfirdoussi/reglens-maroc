"""API tests: fake LLM, hash embedder, in-memory Qdrant. Documents here are FICTIONAL."""

import json
from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from reglens.api.main import create_app
from reglens.config import Settings
from reglens.generation.llm import FakeLLM
from reglens.guardrails import ScopeClassifier
from reglens.ingestion.metadata import write_sidecar
from reglens.ingestion.pipeline import ingest
from reglens.models import DocumentMetadata
from reglens.rag import RagPipeline
from reglens.retrieval.embeddings import HashEmbedder
from reglens.retrieval.retriever import Retriever
from reglens.retrieval.vector_store import QdrantStore, make_client
from reglens.service import Service

USER, ADMIN = "user-key-FICTIONAL", "admin-key-FICTIONAL"
TEXT = "Article 1 Le ratio FICTIONAL de liquidité des banques est au moins égal à 140 %."


def _pdf(path: Path | None, text: str) -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text)
    data: bytes = doc.tobytes()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def _client(tmp_path: Path, **overrides: object) -> TestClient:
    raw = tmp_path / "raw"
    pdf = raw / "bam" / "fictional.pdf"
    _pdf(pdf, TEXT)
    write_sidecar(
        pdf,
        DocumentMetadata(
            title="Circulaire FICTIONAL", issuer="BAM", reference="91/Z/2099", url="https://x.test"
        ),
    )
    options: dict[str, object] = {
        "api_keys": USER,
        "admin_api_keys": ADMIN,
        "rate_limit_per_minute": 100,
        "data_dir": raw,
        "chunking": "legal",
        "chunk_size": 1200,
        "chunk_overlap": 200,
        "warm_on_start": False,
    }
    options.update(overrides)
    settings = Settings(_env_file=None, **options)  # type: ignore[arg-type]
    store = QdrantStore(make_client(None, Path(":memory:")), "api")
    embedder = HashEmbedder()
    ingest(raw, embedder, store, 1200, 200, strategy="legal")
    pipeline = RagPipeline(
        Retriever(embedder, store),
        FakeLLM("Le ratio est de 140 % [1]."),
        k=3,
        scope=ScopeClassifier(embedder),
    )
    return TestClient(create_app(settings, Service(settings, pipeline)))


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with _client(tmp_path) as test_client:
        yield test_client


def _ask(client: TestClient, question: str, key: str | None = USER) -> dict[str, object]:
    headers = {"X-API-Key": key} if key else {}
    response = client.post("/v1/query", json={"question": question}, headers=headers)
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_health_ready_and_metrics(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["chunks"] > 0
    _ask(client, "Quel est le ratio de liquidité ?")
    body = client.get("/metrics").text
    assert "reglens_requests_total" in body and "reglens_stage_seconds" in body


def test_query_requires_a_valid_key(client: TestClient) -> None:
    assert client.post("/v1/query", json={"question": "Ratio ?"}).status_code == 401
    bad = client.post("/v1/query", json={"question": "Ratio ?"}, headers={"X-API-Key": "wrong"})
    assert bad.status_code == 401


def test_query_answers_with_citations_and_caches(client: TestClient) -> None:
    first = _ask(client, "Quel est le ratio de liquidité des banques ?")
    assert first["status"] == "answered" and first["cached"] is False
    assert first["citations"][0]["reference"] == "91/Z/2099"  # type: ignore[index]
    assert any(s["cited"] for s in first["sources"])  # type: ignore[union-attr]
    again = _ask(client, "  quel est le RATIO de liquidité des banques ?  ")
    assert again["cached"] is True and again["answer"] == first["answer"]


def test_request_id_is_echoed(client: TestClient) -> None:
    response = client.post(
        "/v1/query",
        json={"question": "Ratio de liquidité ?"},
        headers={"X-API-Key": USER, "X-Request-ID": "req-FICTIONAL-1"},
    )
    assert response.headers["X-Request-ID"] == "req-FICTIONAL-1"
    assert response.json()["request_id"] == "req-FICTIONAL-1"


def test_guardrails_through_the_api(client: TestClient) -> None:
    meta = _ask(client, "Quel modèle d'IA utilises-tu ?")
    assert meta["status"] == "out_of_scope" and meta["guardrail"] == "meta"
    assert meta["sources"] == []


def test_stream_sends_tokens_then_the_answer(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/query/stream",
        json={"question": "Ratio de liquidité ?"},
        headers={"X-API-Key": USER},
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    events = [block for block in body.split("\n\n") if block]
    assert events[0].startswith("event: token")
    final = events[-1]
    assert final.startswith("event: answer")
    payload = json.loads(final.split("data: ", 1)[1])
    assert payload["status"] == "answered"


def test_rate_limit_returns_429_with_retry_after(tmp_path: Path) -> None:
    with _client(tmp_path, rate_limit_per_minute=2) as limited:
        _ask(limited, "Ratio ?")
        _ask(limited, "Ratio ?")
        third = limited.post("/v1/query", json={"question": "Ratio ?"}, headers={"X-API-Key": USER})
        assert third.status_code == 429 and int(third.headers["Retry-After"]) >= 1


def test_ingest_is_admin_only_and_validates_uploads(client: TestClient, tmp_path: Path) -> None:
    form = {"title": "Directive FICTIONAL", "issuer": "BAM", "url": "https://x.test/d.pdf"}
    pdf = _pdf(None, "Article 1 Le délai FICTIONAL est de onze jours.")
    files = {"file": ("directive.pdf", pdf, "application/pdf")}
    assert (
        client.post("/v1/ingest", data=form, files=files, headers={"X-API-Key": USER}).status_code
        == 403
    )
    not_pdf = {"file": ("notes.txt", b"hello", "text/plain")}
    assert (
        client.post(
            "/v1/ingest", data=form, files=not_pdf, headers={"X-API-Key": ADMIN}
        ).status_code
        == 415
    )
    fake_pdf = {"file": ("fake.pdf", b"not really a pdf", "application/pdf")}
    assert (
        client.post(
            "/v1/ingest", data=form, files=fake_pdf, headers={"X-API-Key": ADMIN}
        ).status_code
        == 415
    )


def test_ingest_adds_then_skips_unchanged_and_clears_the_cache(client: TestClient) -> None:
    _ask(client, "Quel est le délai FICTIONAL ?")  # fills the cache
    form = {"title": "Directive FICTIONAL", "issuer": "BAM", "url": "https://x.test/d.pdf"}
    pdf = _pdf(None, "Article 1 Le délai FICTIONAL de traitement est de onze jours.")
    headers = {"X-API-Key": ADMIN}
    added = client.post(
        "/v1/ingest",
        data=form,
        files={"file": ("directive.pdf", pdf, "application/pdf")},
        headers=headers,
    )
    assert added.status_code == 200, added.text
    assert added.json()["status"] == "added" and added.json()["chunks"] >= 1
    again = client.post(
        "/v1/ingest",
        data=form,
        files={"file": ("directive.pdf", pdf, "application/pdf")},
        headers=headers,
    )
    assert again.json()["status"] == "unchanged"
    assert _ask(client, "Quel est le délai FICTIONAL ?")["cached"] is False  # cache was cleared


def test_upload_size_limit(tmp_path: Path) -> None:
    with _client(tmp_path, upload_max_mb=0) as small:
        form = {"title": "Directive FICTIONAL", "issuer": "BAM", "url": "https://x.test/d.pdf"}
        files = {"file": ("big.pdf", b"%PDF-1.4 FICTIONAL", "application/pdf")}
        response = small.post("/v1/ingest", data=form, files=files, headers={"X-API-Key": ADMIN})
        assert response.status_code == 413


def test_open_mode_without_keys(tmp_path: Path) -> None:
    with _client(tmp_path, api_keys=None, admin_api_keys=None) as open_client:
        assert _ask(open_client, "Ratio de liquidité ?", key=None)["status"] == "answered"


def test_api_starts_even_when_the_index_is_missing(tmp_path: Path) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        embedding_model="hash",
        qdrant_path=tmp_path / "empty-qdrant",
        api_keys=USER,
        warm_on_start=False,
    )
    with TestClient(create_app(settings)) as down:
        assert down.get("/health").status_code == 200
        ready = down.get("/ready")
        assert ready.status_code == 503 and "EmptyIndexError" in ready.json()["error"]
        query = down.post("/v1/query", json={"question": "Ratio ?"}, headers={"X-API-Key": USER})
        assert query.status_code == 503
