"""Detect scanned PDFs and OCR them.

A PDF counts as scanned when its text layer averages fewer than ``MIN_CHARS_PER_PAGE``
characters per page. OCR text is written per page to ``<file>.ocr.json`` next to the PDF (the
original is never modified), and the metadata sidecar records ``text_source``.

The OCR engine is the Tesseract CLI on pages rendered by PyMuPDF (no extra Python
dependency). It sits behind the ``PageOcr`` callable so tests can inject a fake.
"""

import json
import os
import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import pymupdf
import structlog

from reglens.ingestion.metadata import read_sidecar, sidecar_path, write_sidecar

log = structlog.get_logger(__name__)

MIN_CHARS_PER_PAGE = 100
DEFAULT_LANGUAGE = "fra"
DEFAULT_DPI = 300
# Metadata language code -> Tesseract model.
TESSERACT_LANGUAGES = {"fr": "fra", "ar": "ara", "en": "eng"}
TESSERACT_CMD = os.environ.get("REGLENS_TESSERACT_CMD", "tesseract")
OCR_TIMEOUT_S = 300

PageOcr = Callable[[pymupdf.Page, str, int], str]
Status = Literal["native", "ocr", "cached"]


def tesseract_page(page: pymupdf.Page, language: str, dpi: int) -> str:
    """OCR a page; two-column pages are OCR'd column by column to keep the reading order.

    Tesseract merges the columns of Bulletin Officiel pages (narrow gutter) into full-width
    lines, interleaving unrelated sentences, so we split at the detected gutter first.
    """
    split = column_split(page)
    if split is None:
        return _ocr_clip(page, page.rect, language, dpi)
    r = page.rect
    left = pymupdf.Rect(r.x0, r.y0, split, r.y1)
    right = pymupdf.Rect(split, r.y0, r.x1, r.y1)
    return _ocr_clip(page, left, language, dpi) + "\n" + _ocr_clip(page, right, language, dpi)


def _ocr_clip(page: pymupdf.Page, clip: pymupdf.Rect, language: str, dpi: int) -> str:
    """Render a page region and OCR it with the Tesseract CLI (plain-text output).

    We deliberately avoid PyMuPDF's OCR-to-PDF path: re-extracting text from Tesseract's PDF
    output silently dropped whole table rows (e.g. risk-weight rows "0 % | 20 % | 50 %"),
    while Tesseract's own text output keeps them (measured in docs/corpus.md).
    """
    png = page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")
    result = subprocess.run(
        [TESSERACT_CMD, "stdin", "stdout", "-l", language, "--dpi", str(dpi)],
        input=png,
        capture_output=True,
        check=True,
        timeout=OCR_TIMEOUT_S,
    )
    output: bytes = result.stdout
    return output.decode("utf-8")


def column_split(
    page: pymupdf.Page,
    dpi: int = 50,
    band: tuple[float, float] = (0.42, 0.56),
    max_gutter_ratio: float = 0.10,
    min_side_rows: float = 0.30,
) -> float | None:
    """Return the x coordinate (PDF points) of a two-column gutter, or None.

    Works on a low-resolution grayscale render. A gutter is a narrow vertical strip near the
    page centre whose ink is below ``max_gutter_ratio`` of the median ink of the text area
    (one dense pixel column is tolerated: Bulletin Officiel pages draw a rule there). Both
    sides must carry text over at least ``min_side_rows`` of the body height, which rules out
    short pages such as a signature block. Thresholds were calibrated on the corpus.
    """
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    width, height = pixmap.width, pixmap.height
    samples, stride = pixmap.samples, pixmap.stride
    y0, y1 = int(height * 0.12), int(height * 0.95)  # skip the running header and footer

    def dark(x: int, y: int) -> bool:
        return bool(samples[y * stride + x] < 128)

    ink = [sum(dark(x, y) for y in range(y0, y1)) for x in range(width)]
    body = sorted(ink[int(width * 0.1) : int(width * 0.9)])
    typical = body[len(body) // 2]
    if typical == 0:
        return None

    window = 3
    scores = {
        x: sum(sorted(ink[x - window : x + window + 1])[:-1]) / (2 * window)
        for x in range(int(width * band[0]), int(width * band[1]) + 1)
    }
    gutter = min(scores.values())
    if gutter / typical > max_gutter_ratio:
        return None
    emptiest = [x for x, score in scores.items() if score == gutter]
    x_split = emptiest[len(emptiest) // 2]  # middle of the gap, away from both columns

    def rows_with_text(x_from: int, x_to: int) -> int:
        return sum(any(dark(x, y) for x in range(x_from, x_to)) for y in range(y0, y1))

    min_rows = min_side_rows * (y1 - y0)
    if rows_with_text(int(width * 0.08), x_split - window) < min_rows:
        return None
    if rows_with_text(x_split + window, int(width * 0.92)) < min_rows:
        return None
    return float(x_split / width * page.rect.width)


def ocr_path(pdf: Path) -> Path:
    return pdf.with_suffix(".ocr.json")


def mean_chars_per_page(doc: pymupdf.Document) -> float:
    if doc.page_count == 0:
        return 0.0
    return float(sum(len(page.get_text().strip()) for page in doc) / doc.page_count)


def needs_ocr(doc: pymupdf.Document, threshold: int = MIN_CHARS_PER_PAGE) -> bool:
    return mean_chars_per_page(doc) < threshold


@dataclass(frozen=True)
class OcrOutcome:
    pdf: Path
    status: Status
    pages: int


def document_language(pdf: Path) -> str:
    """Tesseract language for a PDF, from its sidecar (French when unknown)."""
    if sidecar_path(pdf).exists():
        language = read_sidecar(pdf).language
        if language is not None:
            return TESSERACT_LANGUAGES[language]
    return DEFAULT_LANGUAGE


def ocr_document(
    pdf: Path,
    language: str | None = None,
    dpi: int = DEFAULT_DPI,
    force: bool = False,
    engine: PageOcr = tesseract_page,
) -> OcrOutcome:
    """OCR one PDF if it has no usable text layer; update its sidecar's ``text_source``.

    ``language`` defaults to the document's own language (see ``document_language``).
    """
    language = language or document_language(pdf)
    with pymupdf.open(pdf) as doc:
        pages = doc.page_count
        if not needs_ocr(doc):
            status: Status = "native"
        elif ocr_path(pdf).exists() and not force:
            status = "cached"
        else:
            texts = [engine(page, language, dpi) for page in doc]
            payload = {
                "engine": "tesseract",
                "language": language,
                "dpi": dpi,
                "pages": [{"page": n, "text": t} for n, t in enumerate(texts, start=1)],
            }
            ocr_path(pdf).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            status = "ocr"
            log.info("document_ocr_done", file=str(pdf), pages=pages)

    if sidecar_path(pdf).exists():
        metadata = read_sidecar(pdf)
        text_source = "native" if status == "native" else "ocr"
        if metadata.text_source != text_source:
            write_sidecar(pdf, metadata.model_copy(update={"text_source": text_source}))
    return OcrOutcome(pdf, status, pages)


def ocr_directory(
    raw_dir: Path,
    language: str | None = None,
    dpi: int = DEFAULT_DPI,
    force: bool = False,
    workers: int = 1,
) -> list[OcrOutcome]:
    """OCR every scanned PDF under ``raw_dir``, largest first so workers finish together."""
    pdfs = sorted(raw_dir.rglob("*.pdf"), key=lambda p: p.stat().st_size, reverse=True)
    run = partial(ocr_document, language=language, dpi=dpi, force=force)
    if workers <= 1:
        return [run(pdf) for pdf in pdfs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, pdfs))


def page_texts(pdf: Path) -> list[str]:
    """Text of each page: from the OCR sidecar when present, else the PDF text layer."""
    if ocr_path(pdf).exists():
        payload = json.loads(ocr_path(pdf).read_text(encoding="utf-8"))
        return [str(page["text"]) for page in payload["pages"]]
    with pymupdf.open(pdf) as doc:
        return [page.get_text() for page in doc]
