"""Discover and download official regulatory PDFs from issuer listing pages.

Each listing page (see ``data/source_pages.yaml``) is parsed for links to PDF files. The title
comes from the enclosing listing row when there is one (AMMC layout), otherwise from the link
text (BAM layout). Reference numbers and dates are parsed from the title only; anything that
cannot be parsed is left empty and the document is flagged ``needs_review`` — metadata is never
guessed.
"""

import hashlib
import re
import time
import unicodedata
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import structlog
import yaml
from pydantic import BaseModel

from reglens.ingestion.metadata import read_sidecar, sidecar_path, write_sidecar, write_yaml
from reglens.models import DocumentMetadata, Issuer, Language

log = structlog.get_logger(__name__)

USER_AGENT = "RegLensCorpusBot/0.1 (non-commercial research; polite crawler)"
MAX_PDF_BYTES = 50 * 1024 * 1024
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class SourcePage(BaseModel):
    url: str
    issuer: Issuer
    language: Language = "fr"
    max_pages: int = 10


@dataclass(frozen=True)
class Candidate:
    url: str
    title: str
    issuer: Issuer
    language: Language
    source_page: str
    reference: str | None
    published: date | None


# --- Parsing (pure functions) -------------------------------------------------------------


@dataclass(frozen=True)
class _Link:
    href: str
    text: str
    row_title: str | None


class _ListingParser(HTMLParser):
    """Collects ``<a href>`` links, the listing-row title they belong to, and the next page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[_Link] = []
        self.next_href: str | None = None
        self._href: str | None = None
        self._text: list[str] = []
        self._row_title: str | None = None
        self._title_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if "views-field-title" in classes:
            self._title_parts = []
        if tag == "a" and attributes.get("href"):
            self._href = attributes["href"]
            self._text = []
            if attributes.get("rel") == "next":
                self.next_href = attributes["href"]

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._title_parts is not None:
            self._row_title = _squash(" ".join(self._title_parts)) or None
            self._title_parts = None
        elif tag == "a" and self._href is not None:
            self.links.append(_Link(self._href, _squash(" ".join(self._text)), self._row_title))
            self._href = None
        elif tag == "li":
            self._row_title = None

    def handle_data(self, data: str) -> None:
        if self._title_parts is not None:
            self._title_parts.append(data)
        if self._href is not None:
            self._text.append(data)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_pdf(href: str) -> bool:
    return urlsplit(href).path.lower().endswith(".pdf")


def parse_listing(html: str, page: SourcePage, page_url: str) -> tuple[list[Candidate], str | None]:
    """Return the PDF candidates on a listing page and the absolute URL of the next page."""
    parser = _ListingParser()
    parser.feed(html)
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for link in parser.links:
        if not _is_pdf(link.href):
            continue
        url = urljoin(page_url, link.href)
        if url in seen:
            continue
        seen.add(url)
        title = link.row_title or link.text or _filename(url)
        candidates.append(
            Candidate(
                url=url,
                title=title,
                issuer=page.issuer,
                language=page.language,
                source_page=page.url,
                reference=parse_reference(title, page.issuer),
                published=parse_french_date(title),
            )
        )
    next_url = urljoin(page_url, parser.next_href) if parser.next_href else None
    return candidates, next_url


def _filename(url: str) -> str:
    return PurePosixPath(unquote(urlsplit(url).path)).stem


_BAM_REF = re.compile(r"(?<!\d)(\d{1,3})\s*[/.]?\s*([GW])\s*[/.]?\s*(\d{4}|\d{2})(?!\d)", re.I)
_AMMC_REF = re.compile(r"n\s*°\s*(\d{1,2})\s*[-_/]\s*(\d{4}|\d{2})(?!\d)", re.I)


# Fallback for other numbered texts (e.g. ministerial orders "n° 174-97", "n°57/DCEC/2001").
_GENERIC_REF = re.compile(r"n\s*°\s*(\d+(?:\s*[-/]\s*[A-Z0-9]+)+)(?![\w])", re.I)


def parse_reference(title: str, issuer: Issuer) -> str | None:
    """Extract a normalised reference (BAM ``14/G/2013``, AMMC ``01-23``) from a title.

    The earliest reference in the title wins, so "Lettre circulaire n°01/DSB/2007 ... de la
    circulaire n°25/G/2006" yields the letter's own number, not the one it refers to. On a
    tie the issuer-specific (normalised) form is preferred over the generic one.
    """
    found: list[tuple[int, int, str]] = []  # (position, priority, reference)
    if issuer == "BAM" and (match := _BAM_REF.search(title)):
        number, series, year = match.groups()
        full_year = year if len(year) == 4 else f"20{year}"
        found.append((match.start(1), 0, f"{int(number)}/{series.upper()}/{full_year}"))
    elif issuer == "AMMC" and (match := _AMMC_REF.search(title)):
        number, year = match.groups()
        found.append((match.start(1), 0, f"{int(number):02d}-{year}"))
    if generic := _GENERIC_REF.search(title):
        found.append((generic.start(1), 1, re.sub(r"\s+", "", generic.group(1))))
    return min(found)[2] if found else None


_MONTHS = {
    "janvier": 1,
    "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
}
_FR_DATE = re.compile(r"\b(1er|\d{1,2})\s+([a-zéèêûô]+)\s+(\d{4})\b", re.I)


def _fold(text: str) -> str:
    """Lowercase and strip accents."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def parse_french_date(text: str) -> date | None:
    """Parse the first French date such as ``13 Août 2013`` or ``1er avril 2005``."""
    for day, month, year in _FR_DATE.findall(text):
        month_number = _MONTHS.get(_fold(month))
        if month_number is None:
            continue
        try:
            return date(int(year), month_number, 1 if day.lower() == "1er" else int(day))
        except ValueError:
            continue
    return None


def slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _fold(text).encode("ascii", "ignore").decode()).strip("-")
    return slug[:max_length].rstrip("-") or "document"


def document_stem(candidate: Candidate) -> str:
    """Stable file stem: issuer + readable slug + short URL hash (avoids collisions)."""
    url_hash = hashlib.sha1(candidate.url.encode()).hexdigest()[:8]
    return f"{candidate.issuer.lower()}-{slugify(candidate.title)}-{url_hash}"


def select(
    candidates: Iterable[Candidate],
    issuer: str | None = None,
    match: str | None = None,
    limit: int | None = None,
    urls: Collection[str] | None = None,
) -> list[Candidate]:
    """Filter candidates by URL allow-list, issuer and title regex, then cap the count."""
    pattern = re.compile(match, re.I) if match else None
    selected = [
        c
        for c in candidates
        if (urls is None or c.url in urls)
        and (issuer is None or c.issuer == issuer.upper())
        and (pattern is None or pattern.search(c.title))
    ]
    return selected if limit is None else selected[:limit]


class SelectionEntry(BaseModel):
    """One curated document. Optional fields override values parsed from the website, after
    manual verification against the document itself (record the source in ``notes``)."""

    url: str
    reference: str | None = None
    published: date | None = None
    language: Language | None = None
    notes: str | None = None


def load_selection(path: Path) -> list[SelectionEntry]:
    """Read the curated corpus (``documents: [{url: ..., reference: ...}, ...]``)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [SelectionEntry.model_validate(item) for item in data.get("documents", [])]


def apply_overrides(pdf: Path, entry: SelectionEntry) -> bool:
    """Apply curated values to a document's sidecar. Returns True if the sidecar changed."""
    metadata = read_sidecar(pdf)
    updates = entry.model_dump(include=entry.model_fields_set - {"url"})
    updated = metadata.model_copy(update=updates)
    updated.needs_review = updated.reference is None or updated.published is None
    if updated == metadata:
        return False
    write_sidecar(pdf, updated)
    return True


# --- HTTP ---------------------------------------------------------------------------------


class RobotsDisallowedError(RuntimeError):
    pass


class Fetcher:
    """Polite HTTP client: honours robots.txt, waits between requests, retries with backoff."""

    def __init__(
        self,
        client: httpx.Client,
        delay: float = 1.0,
        retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._delay = delay
        self._retries = retries
        self._sleep = sleep
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request = 0.0

    def get(self, url: str) -> httpx.Response:
        if not self._allowed(url):
            raise RobotsDisallowedError(url)
        for attempt in range(self._retries + 1):
            self._wait()
            try:
                response = self._client.get(url)
                if response.status_code not in RETRY_STATUSES:
                    response.raise_for_status()
                    return response
                error: Exception = httpx.HTTPStatusError(
                    f"status {response.status_code}", request=response.request, response=response
                )
            except httpx.TransportError as exc:
                error = exc
            if attempt == self._retries:
                raise error
            backoff = 2.0**attempt
            log.warning("fetch_retry", url=url, attempt=attempt + 1, backoff_s=backoff)
            self._sleep(backoff)
        raise AssertionError("unreachable")

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._delay:
            self._sleep(self._delay - elapsed)
        self._last_request = time.monotonic()

    def _allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            robots = RobotFileParser()
            try:
                response = self._client.get(f"{origin}/robots.txt")
                robots.parse(response.text.splitlines() if response.status_code == 200 else [])
            except httpx.HTTPError:
                robots.parse([])
            self._robots[origin] = robots
        return self._robots[origin].can_fetch(USER_AGENT, url)


def make_client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True)


# --- Pipeline -----------------------------------------------------------------------------


def load_source_pages(path: Path) -> list[SourcePage]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [SourcePage.model_validate(item) for item in data.get("pages", [])]


def discover(pages: Iterable[SourcePage], fetcher: Fetcher) -> list[Candidate]:
    """Crawl every listing page (following pagination) and return unique PDF candidates."""
    found: dict[str, Candidate] = {}
    for page in pages:
        url: str | None = page.url
        for _ in range(page.max_pages):
            if url is None:
                break
            response = fetcher.get(url)
            candidates, url = parse_listing(response.text, page, str(response.url))
            for candidate in candidates:
                found.setdefault(candidate.url, candidate)
            log.info("listing_parsed", page=page.url, candidates=len(candidates))
    return list(found.values())


def download(
    candidate: Candidate, out_dir: Path, fetcher: Fetcher, max_bytes: int = MAX_PDF_BYTES
) -> Path | None:
    """Download one PDF and write its YAML sidecar. Skips files already on disk.

    Returns the PDF path, or ``None`` when the response is not a usable PDF.
    """
    target_dir = out_dir / candidate.issuer.lower()
    stem = document_stem(candidate)
    pdf_path = target_dir / f"{stem}.pdf"
    if pdf_path.exists() and sidecar_path(pdf_path).exists():
        log.info("document_skipped", file=str(pdf_path), reason="exists")
        return pdf_path

    content = fetcher.get(candidate.url).content
    if not content.startswith(b"%PDF"):
        log.warning("document_rejected", url=candidate.url, reason="not_pdf")
        return None
    if len(content) > max_bytes:
        log.warning("document_rejected", url=candidate.url, reason="too_large", size=len(content))
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(content)
    metadata = DocumentMetadata(
        title=candidate.title,
        issuer=candidate.issuer,
        reference=candidate.reference,
        published=candidate.published,
        url=candidate.url,
        language=candidate.language,
        source_page=candidate.source_page,
        sha256=hashlib.sha256(content).hexdigest(),
        fetched_at=datetime.now(UTC),
        needs_review=candidate.reference is None or candidate.published is None,
    )
    write_sidecar(pdf_path, metadata)
    log.info("document_downloaded", file=str(pdf_path), bytes=len(content))
    return pdf_path


def write_manifest(raw_dir: Path, manifest_path: Path) -> int:
    """Rebuild the corpus manifest from every sidecar under ``raw_dir``."""
    documents = []
    for sidecar in sorted(raw_dir.rglob("*.yaml")):
        pdf = sidecar.with_suffix(".pdf")
        entry: dict[str, object] = {"file": pdf.relative_to(raw_dir).as_posix()}
        entry.update(read_sidecar(pdf).model_dump(mode="json", exclude={"fetched_at"}))
        documents.append(entry)
    write_yaml(manifest_path, {"documents": documents})
    return len(documents)
