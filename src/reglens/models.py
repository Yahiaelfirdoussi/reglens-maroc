"""Shared domain models."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

Issuer = Literal["BAM", "AMMC", "ACAPS"]
Language = Literal["fr", "ar", "en"]


class DocumentMetadata(BaseModel):
    """Metadata stored in the ``<file>.yaml`` sidecar next to each source PDF."""

    title: str
    issuer: Issuer
    reference: str | None = None
    published: date | None = None
    url: str
    language: Language | None = None
    source_page: str | None = None
    sha256: str | None = None
    fetched_at: datetime | None = None
    # "native" = PDF text layer; "ocr" = text comes from the ``<file>.ocr.json`` sidecar.
    text_source: Literal["native", "ocr"] | None = None
    # True when reference or date could not be parsed and must be filled by hand.
    needs_review: bool = False
    # Provenance of manually curated values (e.g. which act a date comes from).
    notes: str | None = None


class Chunk(BaseModel):
    """A retrievable passage with the metadata needed to cite it."""

    id: str
    text: str
    file: str  # path relative to the raw data directory
    index: int  # position of the chunk within its document
    page_start: int  # 1-based
    page_end: int
    issuer: Issuer
    reference: str | None = None
    title: str
    url: str
    language: Language | None = None
    published: date | None = None
    section: str | None = None
    # Context prepended for embedding only ("BAM | 14/G/2013 | <title> | Article 4"); the
    # stored text stays the official wording used for citations.
    header: str | None = None

    @property
    def embedding_text(self) -> str:
        return f"{self.header}\n{self.text}" if self.header else self.text


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
