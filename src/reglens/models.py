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
