"""Ingestion: PDFs + sidecars -> cleaned pages -> chunks -> embeddings -> vector store."""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import structlog

from reglens.ingestion.chunking import fixed_size_chunks
from reglens.ingestion.cleaning import clean_page
from reglens.ingestion.metadata import read_sidecar
from reglens.ingestion.ocr import page_texts
from reglens.models import Chunk
from reglens.retrieval.embeddings import Embedder
from reglens.retrieval.vector_store import QdrantStore

log = structlog.get_logger(__name__)


def chunk_id(file: str, index: int) -> str:
    """Stable id: re-ingesting the same document overwrites its points."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"reglens:{file}#{index}"))


def document_chunks(pdf: Path, raw_dir: Path, size: int, overlap: int) -> list[Chunk]:
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
        )
        for index, span in enumerate(fixed_size_chunks(pages, size, overlap))
    ]


def iter_chunks(raw_dir: Path, size: int, overlap: int) -> Iterator[Chunk]:
    for pdf in sorted(raw_dir.rglob("*.pdf")):
        yield from document_chunks(pdf, raw_dir, size, overlap)


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
) -> IngestStats:
    """Rebuild the collection from every document under ``raw_dir``."""
    chunks = list(iter_chunks(raw_dir, size, overlap))
    is_truncated = getattr(embedder, "is_truncated", None)
    truncated = sum(is_truncated(c.text) for c in chunks) if is_truncated else 0
    if truncated:
        log.warning("chunks_truncated", count=truncated, total=len(chunks))
    store.recreate(embedder.dim)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        store.upsert(batch, embedder.embed_documents([c.text for c in batch]))
    documents = len({c.file for c in chunks})
    log.info("ingest_done", documents=documents, chunks=len(chunks), embedder=embedder.name)
    return IngestStats(documents, len(chunks), truncated)
