"""Golden evaluation set: schema, loading and evidence validation.

Every answerable question carries its evidence: the source file, page and an exact quote.
``validate`` checks that the quote really is on that page and that each expected fact appears
in the quote, so expected answers can never drift away from the documents.
"""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from reglens.evaluation.text_metrics import normalize
from reglens.ingestion.ocr import page_texts
from reglens.models import Issuer, Language

Status = Literal["draft", "approved", "rejected"]


class Evidence(BaseModel):
    file: str  # path relative to the raw data directory
    page: int  # 1-based
    quote: str


class GoldenItem(BaseModel):
    id: str
    question: str
    language: Language
    answerable: bool
    theme: str
    expected_issuer: Issuer | None = None
    expected_reference: str | None = None
    expected_section: str | None = None
    expected_facts: list[str] = []
    evidence: Evidence | None = None
    status: Status = "draft"
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "GoldenItem":
        if self.answerable and self.evidence is None:
            raise ValueError(f"{self.id}: answerable questions need evidence")
        if not self.answerable and (self.evidence or self.expected_facts):
            raise ValueError(f"{self.id}: unanswerable questions carry no evidence or facts")
        return self


def load_golden(path: Path) -> list[GoldenItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GoldenItem.model_validate_json(line) for line in lines if line.strip()]


def write_golden(path: Path, items: list[GoldenItem]) -> None:
    path.write_text(
        "".join(
            json.dumps(item.model_dump(exclude_none=True), ensure_ascii=False) + "\n"
            for item in items
        ),
        encoding="utf-8",
    )


def compact(text: str) -> str:
    """Comparison form: normalised, lower-case, no whitespace ("100 %" == "100%")."""
    return re.sub(r"\s+", "", normalize(text)).lower()


def validate(item: GoldenItem, raw_dir: Path) -> list[str]:
    """Return the problems with one item (empty list when it is consistent)."""
    if item.evidence is None:
        return []
    errors = []
    pdf = raw_dir / item.evidence.file
    if not pdf.exists():
        return [f"{item.id}: file not found: {item.evidence.file}"]
    pages = page_texts(pdf)
    if not 1 <= item.evidence.page <= len(pages):
        return [f"{item.id}: page {item.evidence.page} out of range (1-{len(pages)})"]
    quote = compact(item.evidence.quote)
    if quote not in compact(pages[item.evidence.page - 1]):
        errors.append(f"{item.id}: quote not found on page {item.evidence.page}")
    for fact in item.expected_facts:
        if compact(fact) not in quote:
            errors.append(f"{item.id}: fact {fact!r} not in quote")
    return errors


def find_page(pdf: Path, quote: str) -> int | None:
    """First page whose text contains the quote (used when drafting items)."""
    target = compact(quote)
    for number, text in enumerate(page_texts(pdf), start=1):
        if target in compact(text):
            return number
    return None
