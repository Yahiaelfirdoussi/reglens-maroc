"""Tests for scanned-PDF detection and OCR. All documents here are FICTIONAL."""

import json
from pathlib import Path

import pymupdf

from reglens.ingestion.metadata import read_sidecar, write_sidecar
from reglens.ingestion.ocr import (
    column_split,
    needs_ocr,
    ocr_directory,
    ocr_document,
    ocr_path,
    page_texts,
)
from reglens.models import DocumentMetadata

NATIVE_TEXT = "Article 1 — Texte FICTIONAL avec une couche texte native. " * 5


def _make_pdf(path: Path, pages: int, native: bool) -> Path:
    doc = pymupdf.open()
    for _ in range(pages):
        page = doc.new_page()
        if native:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), NATIVE_TEXT)
        else:  # image-like page: drawings only, no text layer
            page.draw_rect(pymupdf.Rect(50, 50, 300, 300), fill=(0.5, 0.5, 0.5))
    doc.save(path)
    doc.close()
    write_sidecar(
        path,
        DocumentMetadata(title="Texte FICTIONAL", issuer="BAM", url="https://bam.test/x.pdf"),
    )
    return path


def fake_engine(page: pymupdf.Page, language: str, dpi: int) -> str:
    return f"FICTIONAL OCR page {page.number + 1} ({language}, {dpi} dpi)"


COLUMN_TEXT = "Texte FICTIONAL sur plusieurs lignes pour remplir la colonne. " * 40


def _page(doc: pymupdf.Document, boxes: list[pymupdf.Rect], text: str) -> pymupdf.Page:
    page = doc.new_page(width=595, height=842)
    for box in boxes:
        page.insert_textbox(box, text, fontsize=9)
    return page


def test_column_split_detects_two_columns() -> None:
    doc = pymupdf.open()
    page = _page(
        doc, [pymupdf.Rect(40, 100, 285, 800), pymupdf.Rect(310, 100, 555, 800)], COLUMN_TEXT
    )
    split = column_split(page)
    assert split is not None
    assert 285 <= split <= 310


def test_column_split_ignores_single_column_and_short_pages() -> None:
    doc = pymupdf.open()
    single = _page(doc, [pymupdf.Rect(40, 100, 555, 800)], COLUMN_TEXT)
    assert column_split(single) is None
    # Signature-like page: a few lines on the left, a small block on the right.
    short = _page(doc, [pymupdf.Rect(40, 100, 250, 140)], "Article FICTIONAL.\nFin du texte.")
    short.insert_textbox(pymupdf.Rect(380, 250, 555, 280), "Signé : FICTIONAL", fontsize=9)
    assert column_split(short) is None
    assert column_split(doc.new_page()) is None  # blank


def test_needs_ocr(tmp_path: Path) -> None:
    with pymupdf.open(_make_pdf(tmp_path / "native.pdf", 2, native=True)) as doc:
        assert not needs_ocr(doc)
    with pymupdf.open(_make_pdf(tmp_path / "scan.pdf", 2, native=False)) as doc:
        assert needs_ocr(doc)


def test_scanned_pdf_is_ocrd_and_sidecar_updated(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "scan.pdf", 2, native=False)
    outcome = ocr_document(pdf, language="fra", dpi=150, engine=fake_engine)
    assert outcome.status == "ocr"
    assert outcome.pages == 2

    payload = json.loads(ocr_path(pdf).read_text("utf-8"))
    assert payload["language"] == "fra"
    assert [p["page"] for p in payload["pages"]] == [1, 2]
    assert page_texts(pdf) == [
        "FICTIONAL OCR page 1 (fra, 150 dpi)",
        "FICTIONAL OCR page 2 (fra, 150 dpi)",
    ]
    assert read_sidecar(pdf).text_source == "ocr"


def test_cached_ocr_is_reused_unless_forced(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "scan.pdf", 1, native=False)
    ocr_document(pdf, engine=fake_engine)

    def failing_engine(page: pymupdf.Page, language: str, dpi: int) -> str:
        raise AssertionError("should not re-run OCR")

    assert ocr_document(pdf, engine=failing_engine).status == "cached"
    forced = ocr_document(pdf, force=True, engine=lambda p, lang, dpi: "FICTIONAL again")
    assert forced.status == "ocr"
    assert page_texts(pdf) == ["FICTIONAL again"]


def test_native_pdf_is_left_alone(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "native.pdf", 1, native=True)
    outcomes = ocr_directory(tmp_path)
    assert [o.status for o in outcomes] == ["native"]
    assert not ocr_path(pdf).exists()
    assert "FICTIONAL" in page_texts(pdf)[0]
    assert read_sidecar(pdf).text_source == "native"
