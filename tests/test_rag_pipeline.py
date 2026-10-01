"""End-to-end tests: FICTIONAL corpus -> ingest -> retrieve -> answer -> evaluate.

Uses the hash embedder, an in-memory Qdrant and a fake LLM: no network, no API key.
"""

from collections.abc import Sequence
from pathlib import Path

import pymupdf
import pytest

from reglens.evaluation.golden import Evidence, GoldenItem
from reglens.evaluation.metrics import first_hit_rank, hit_rate, mrr, quote_coverage
from reglens.evaluation.runner import evaluate_retrieval, render_markdown
from reglens.generation.citations import build_citations, strip_invalid_citations
from reglens.generation.llm import FakeLLM
from reglens.generation.prompts import build_messages
from reglens.ingestion.metadata import write_sidecar
from reglens.ingestion.pipeline import chunk_id, ingest
from reglens.models import DocumentMetadata
from reglens.rag import RagPipeline
from reglens.retrieval.embeddings import HashEmbedder
from reglens.retrieval.retriever import Retriever
from reglens.retrieval.vector_store import QdrantStore, make_client

DOCS = {
    "bam/fictional-liquidite.pdf": [
        "Article 1 Le ratio de liquidité FICTIONAL doit être au moins égal à 999 % "
        "sur un horizon de 77 jours calendaires.",
        "Article 2 Les actifs FICTIONAL de niveau Z sont retenus dans la limite de 12 %.",
    ],
    "ammc/fictional-bourse.pdf": [
        "Article 5 Le délai de règlement-livraison FICTIONAL est de onze jours de bourse.",
    ],
}


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    for file, pages in DOCS.items():
        path = tmp_path / file
        path.parent.mkdir(exist_ok=True)
        doc = pymupdf.open()
        for text in pages:
            doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text)
        doc.save(path)
        issuer = "BAM" if file.startswith("bam") else "AMMC"
        write_sidecar(
            path,
            DocumentMetadata(
                title=f"Texte {file} FICTIONAL",
                issuer=issuer,
                reference="99/Z/2099",
                url=f"https://example.test/{file}",
                language="fr",
            ),
        )
    return tmp_path


@pytest.fixture()
def retriever(corpus: Path) -> Retriever:
    store = QdrantStore(make_client(None, Path(":memory:")), "test")
    embedder = HashEmbedder()
    stats = ingest(corpus, embedder, store, size=120, overlap=20)
    assert stats.documents == 2
    assert stats.chunks == store.count() > 2
    return Retriever(embedder, store)


def test_retrieval_finds_the_right_article(retriever: Retriever) -> None:
    results = retriever.retrieve("ratio de liquidité FICTIONAL 999 %", k=3)
    assert results[0].chunk.file == "bam/fictional-liquidite.pdf"
    assert "999 %" in results[0].chunk.text
    assert results[0].chunk.page_start == 1


def test_failed_ingest_keeps_the_existing_index(corpus: Path, retriever: Retriever) -> None:
    store = retriever._store
    before = store.count()

    class BrokenEmbedder(HashEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            raise ConnectionError("network down (FICTIONAL)")

    with pytest.raises(ConnectionError):
        ingest(corpus, BrokenEmbedder(), store, size=120, overlap=20)
    assert store.count() == before


def test_contextual_header_is_embedded_but_not_stored_as_text(corpus: Path) -> None:
    store = QdrantStore(make_client(None, Path(":memory:")), "hdr")
    seen: list[str] = []

    class RecordingEmbedder(HashEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            seen.extend(texts)
            return super().embed_documents(texts)

    ingest(
        corpus, RecordingEmbedder(), store, size=300, overlap=50, strategy="legal", header="full"
    )
    hit = Retriever(HashEmbedder(), store).retrieve("ratio de liquidité 999 %", k=1)[0].chunk
    assert hit.header == (
        "BAM | 99/Z/2099 | Texte bam/fictional-liquidite.pdf FICTIONAL | Articles 1-2"
    )  # two short fictional articles are merged
    assert hit.text.startswith("Article 1")  # stored text is the official wording only
    assert any(t.startswith(hit.header + "\n" + "Article 1") for t in seen)


def test_chunk_ids_are_stable() -> None:
    assert chunk_id("bam/x.pdf", 3) == chunk_id("bam/x.pdf", 3) != chunk_id("bam/x.pdf", 4)


def test_answer_with_fake_llm_keeps_only_valid_citations(retriever: Retriever) -> None:
    llm = FakeLLM("Le ratio est de 999 % [1], voir aussi [9].")
    answer = RagPipeline(retriever, llm, k=2).answer("Quel ratio de liquidité FICTIONAL ?")
    assert answer.text == "Le ratio est de 999 % [1], voir aussi."
    assert [c.number for c in answer.citations] == [1]
    system, user = llm.calls[0]
    assert "ONLY the sources" in system["content"]
    assert '<source id="1"' in user["content"]


def test_retrieval_only_mode(retriever: Retriever) -> None:
    answer = RagPipeline(retriever, None, k=2).answer("délai de règlement-livraison")
    assert answer.text is None
    assert len(answer.sources) == 2


def test_prompt_escapes_source_attributes(retriever: Retriever) -> None:
    sources = retriever.retrieve("ratio", k=1)
    sources[0].chunk.title = 'Titre "piégé" <FICTIONAL>'
    content = build_messages("Q ?", sources)[1]["content"]
    assert 'title="Titre &quot;piégé&quot; &lt;FICTIONAL&gt;"' in content


def test_citation_helpers() -> None:
    assert strip_invalid_citations("A [1][3] et B [0].", 2) == "A [1] et B."
    assert build_citations("Rien.", []) == []


def test_metrics() -> None:
    assert first_hit_rank([False, True, True]) == 2
    assert first_hit_rank([False]) is None
    ranks = [1, 2, None, 7]
    assert hit_rate(ranks, 6) == 0.5
    assert mrr(ranks, 6) == pytest.approx((1 + 0.5) / 4)
    assert quote_coverage("ratio de liquidité", "Le RATIO de liquidité") == 1.0


def test_evaluation_runner(retriever: Retriever) -> None:
    items = [
        GoldenItem(
            id="q1",
            question="Quel est le délai de règlement-livraison FICTIONAL ?",
            language="fr",
            answerable=True,
            theme="t",
            expected_facts=["onze jours"],
            evidence=Evidence(
                file="ammc/fictional-bourse.pdf",
                page=1,
                quote="délai de règlement-livraison FICTIONAL est de onze jours de bourse",
            ),
        ),
        GoldenItem(id="u1", question="Hors corpus ?", language="en", answerable=False, theme="u"),
    ]
    report = evaluate_retrieval(items, retriever, k=3, label="test", config={"x": 1})
    assert len(report.items) == 1  # unanswerable questions are not retrieval-scored
    summary = report.summary()
    assert summary["hit@3"] == 1.0
    assert summary["doc_hit@3"] == 1.0
    assert "| **All** | 1 |" in render_markdown(report)
