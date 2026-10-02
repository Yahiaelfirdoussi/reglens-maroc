"""Ingestion: PDFs + sidecars -> cleaned pages -> chunks -> embeddings -> vector store."""

import hashlib
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog

from reglens.ingestion.chunking import TextSpan, fixed_size_chunks, legal_chunks
from reglens.ingestion.cleaning import clean_page
from reglens.ingestion.metadata import read_sidecar
from reglens.ingestion.ocr import page_texts
from reglens.models import Chunk, DocumentMetadata
from reglens.retrieval.embeddings import Embedder
from reglens.retrieval.sparse import BM25Encoder
from reglens.retrieval.vector_store import QdrantStore

log = structlog.get_logger(__name__)


def chunk_id(file: str, index: int) -> str:
    """Stable id: re-ingesting the same document overwrites its points."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"reglens:{file}#{index}"))


Chunking = Literal["fixed", "legal"]


def split(pages: list[str], strategy: Chunking, size: int, overlap: int) -> list[TextSpan]:
    if strategy == "legal":
        return legal_chunks(pages, size, overlap)
    return fixed_size_chunks(pages, size, overlap)


HeaderStyle = Literal["off", "full", "ref"]


def contextual_header(
    metadata: DocumentMetadata, section: str | None, style: HeaderStyle = "full"
) -> str | None:
    """Deterministic context prepended before embedding (missing parts skipped).

    "full": issuer | reference | title | section; "ref": issuer | reference | section.
    """
    if style == "off":
        return None
    title = metadata.title if style == "full" else None
    parts = [metadata.issuer, metadata.reference, title, section]
    return " | ".join(part for part in parts if part)


def sparse_text(chunk: Chunk) -> str:
    """Text indexed by BM25: the chunk plus its reference and article label.

    Metadata goes to the exact-match channel only; prepending it to the dense text lowered
    retrieval quality (see docs/results.md, contextual header).
    """
    parts = [chunk.reference or "", chunk.section or "", chunk.text]
    return " ".join(part for part in parts if part)


def file_sha256(pdf: Path) -> str:
    return hashlib.sha256(pdf.read_bytes()).hexdigest()


def document_chunks(
    pdf: Path,
    raw_dir: Path,
    size: int,
    overlap: int,
    strategy: Chunking = "fixed",
    header: HeaderStyle = "off",
) -> list[Chunk]:
    metadata = read_sidecar(pdf)
    file = pdf.relative_to(raw_dir).as_posix()
    pages = [clean_page(text) for text in page_texts(pdf)]
    fingerprint = file_sha256(pdf)
    return [
        Chunk(
            id=chunk_id(file, index),
            text=span.text,
            file=file,
            index=index,
            page_start=span.page_start,
            page_end=span.page_end,
            issuer=metadata.issuer,
            reference=metadata.reference,
            title=metadata.title,
            url=metadata.url,
            language=metadata.language,
            published=metadata.published,
            section=span.section,
            header=contextual_header(metadata, span.section, header),
            content_sha256=fingerprint,
        )
        for index, span in enumerate(split(pages, strategy, size, overlap))
    ]


def iter_chunks(
    raw_dir: Path,
    size: int,
    overlap: int,
    strategy: Chunking = "fixed",
    header: HeaderStyle = "off",
) -> Iterator[Chunk]:
    for pdf in sorted(raw_dir.rglob("*.pdf")):
        yield from document_chunks(pdf, raw_dir, size, overlap, strategy, header)


@dataclass(frozen=True)
class IngestStats:
    documents: int
    chunks: int
    truncated: int  # chunks longer than the embedding model's window


def ingest(
    raw_dir: Path,
    embedder: Embedder,
    store: QdrantStore,
    size: int,
    overlap: int,
    batch_size: int = 64,
    strategy: Chunking = "fixed",
    header: HeaderStyle = "off",
) -> IngestStats:
    """Rebuild the collection from every document under ``raw_dir``.

    Everything is embedded before the old collection is dropped, so a failure while
    embedding (network, quota) leaves the existing index untouched.
    """
    chunks = list(iter_chunks(raw_dir, size, overlap, strategy, header))
    is_truncated = getattr(embedder, "is_truncated", None)
    truncated = sum(is_truncated(c.embedding_text) for c in chunks) if is_truncated else 0
    if truncated:
        log.warning("chunks_truncated", count=truncated, total=len(chunks))
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(embedder.embed_documents([c.embedding_text for c in batch]))
    keyword_texts = [sparse_text(c) for c in chunks]
    encoder = BM25Encoder.fit(keyword_texts)
    sparse = [encoder.encode_document(text) for text in keyword_texts]
    store.recreate(embedder.dim)
    corpus_stats: dict[str, object] = {BM25_AVERAGE_LENGTH: encoder.average_length}
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        store.upsert(chunks[start:end], vectors[start:end], sparse[start:end], corpus_stats)
    documents = len({c.file for c in chunks})
    log.info("ingest_done", documents=documents, chunks=len(chunks), embedder=embedder.name)
    return IngestStats(documents, len(chunks), truncated)


BM25_AVERAGE_LENGTH = "bm25_average_length"


def ingest_document(
    pdf: Path,
    raw_dir: Path,
    embedder: Embedder,
    store: QdrantStore,
    size: int,
    overlap: int,
    strategy: Chunking = "fixed",
    header: HeaderStyle = "off",
) -> tuple[str, int]:
    """Add or update one document without rebuilding the index (idempotent).

    Returns ("unchanged" | "updated" | "added", chunk count). An unchanged file (same
    content hash) is skipped; a changed file has its old chunks replaced. New chunks are
    embedded before anything is deleted, so a failure leaves the index as it was.
    """
    chunks = document_chunks(pdf, raw_dir, size, overlap, strategy, header)
    if not chunks:
        return "unchanged", 0
    if not store.exists():
        store.recreate(embedder.dim)
    file = chunks[0].file
    previous = store.file_state(file)
    if previous and previous.get("content_sha256") == chunks[0].content_sha256:
        log.info("document_unchanged", file=file)
        return "unchanged", len(chunks)

    vectors = embedder.embed_documents([c.embedding_text for c in chunks])
    reference = store.any_payload() or {}
    average = reference.get(BM25_AVERAGE_LENGTH)
    keyword_texts = [sparse_text(c) for c in chunks]
    encoder = (
        BM25Encoder(float(average))
        if isinstance(average, int | float)
        else BM25Encoder.fit(keyword_texts)
    )
    sparse = [encoder.encode_document(text) for text in keyword_texts]
    store.delete_file(file)
    store.upsert(chunks, vectors, sparse, {BM25_AVERAGE_LENGTH: encoder.average_length})
    status = "updated" if previous else "added"
    log.info("document_ingested", file=file, status=status, chunks=len(chunks))
    return status, len(chunks)
