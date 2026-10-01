"""Tests for reranking with the offline stand-in. Text here is FICTIONAL."""

from collections.abc import Sequence
from pathlib import Path

from reglens.models import Chunk, ScoredChunk
from reglens.retrieval.embeddings import HashEmbedder
from reglens.retrieval.reranker import OverlapReranker, make_reranker, rerank
from reglens.retrieval.retriever import Retriever
from reglens.retrieval.vector_store import QdrantStore, make_client


def _scored(n: int, text: str, score: float) -> ScoredChunk:
    chunk = Chunk(
        id=f"00000000-0000-0000-0000-{n:012d}",
        text=text,
        file=f"bam/{n}.pdf",
        index=0,
        page_start=1,
        page_end=1,
        issuer="BAM",
        title="FICTIONAL",
        url="https://example.test",
    )
    return ScoredChunk(chunk=chunk, score=score)


def test_rerank_reorders_and_keeps_first_stage_scores() -> None:
    candidates = [
        _scored(1, "Texte FICTIONAL sans rapport", 0.9),
        _scored(2, "Le délai de réclamation FICTIONAL est de quarante jours", 0.5),
        _scored(3, "Autre texte FICTIONAL", 0.4),
    ]
    top = rerank(OverlapReranker(), "délai de réclamation", candidates, k=2)
    assert [s.chunk.file for s in top] == ["bam/2.pdf", "bam/1.pdf"]
    assert top[0].first_stage_score == 0.5
    assert top[0].score > top[1].score
    assert rerank(OverlapReranker(), "q", [], k=3) == []


def test_make_reranker() -> None:
    assert make_reranker("", Path(".")) is None
    assert isinstance(make_reranker("overlap", Path(".")), OverlapReranker)


class _CountingReranker(OverlapReranker):
    def __init__(self) -> None:
        self.seen = 0

    def scores(self, question: str, texts: Sequence[str]) -> list[float]:
        self.seen = len(texts)
        return super().scores(question, texts)


def test_reranking_can_be_limited_to_some_languages() -> None:
    store = QdrantStore(make_client(None, Path(":memory:")), "gate")
    embedder = HashEmbedder()
    chunks = [_scored(n, f"Texte FICTIONAL numéro {n}", 0).chunk for n in range(25)]
    store.recreate(embedder.dim)
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    reranker = _CountingReranker()
    retriever = Retriever(embedder, store, reranker=reranker, rerank_languages=frozenset({"ar"}))
    assert not retriever.reranks("Quel est le délai FICTIONAL ?")
    retriever.retrieve("Quel est le délai FICTIONAL ?", k=3)
    assert reranker.seen == 0  # French question: dense only
    assert retriever.reranks("ما هو الأجل؟")
    retriever.retrieve("ما هو الأجل؟", k=3)
    assert reranker.seen == 20  # Arabic question: reranked


def test_fastembed_reranker_loads_lazily() -> None:
    from reglens.retrieval.reranker import FastEmbedReranker

    reranker = FastEmbedReranker("any/model", Path("."), max_chars=10)
    assert reranker._model is None  # nothing loaded until a question needs it


def test_retriever_fetches_candidates_then_keeps_k() -> None:
    store = QdrantStore(make_client(None, Path(":memory:")), "rr")
    embedder = HashEmbedder()
    chunks = [
        _scored(n, f"Article FICTIONAL numéro {n} sur le thème {n % 5}", 0).chunk for n in range(30)
    ]
    store.recreate(embedder.dim)
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    reranker = _CountingReranker()
    results = Retriever(embedder, store, reranker=reranker, candidates=20).retrieve("thème 3", k=4)
    assert reranker.seen == 20
    assert len(results) == 4
    assert all(r.first_stage_score is not None for r in results)
