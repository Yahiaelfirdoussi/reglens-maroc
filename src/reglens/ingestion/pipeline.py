"""Ingestion: PDFs + sidecars -> cleaned pages -> chunks -> embeddings -> vector store."""

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
    store.recreate(embedder.dim)
    for start in range(0, len(chunks), batch_size):
        store.upsert(chunks[start : start + batch_size], vectors[start : start + batch_size])
    documents = len({c.file for c in chunks})
    log.info("ingest_done", documents=documents, chunks=len(chunks), embedder=embedder.name)
    return IngestStats(documents, len(chunks), truncated)
