"""Tests for the corpus fetcher. All HTML, references and titles here are FICTIONAL."""

from datetime import date
from pathlib import Path

import httpx
import pytest
import yaml

from reglens.ingestion.fetch import (
    Candidate,
    Fetcher,
    RobotsDisallowedError,
    SourcePage,
    apply_overrides,
    discover,
    document_stem,
    download,
    load_selection,
    parse_french_date,
    parse_listing,
    parse_reference,
    select,
    write_manifest,
)
from reglens.ingestion.metadata import read_sidecar

# FICTIONAL listing in the BAM layout: the link text is the title.
BAM_HTML = """
<div><a class="link-pdf pdf" href="/content/download/1/1/fictional-a.pdf">
  Circulaire FICTIONAL n°97/G/99 du 1er avril 2099 relative aux tests</a></div>
<div><a class="link-pdf pdf" href="/content/download/2/2/fictional-b.pdf">
  Note FICTIONAL sans référence</a></div>
<div><a href="/autre-page">Not a PDF</a></div>
"""

# FICTIONAL listing in the AMMC layout: the title lives in a separate row field.
AMMC_HTML = """
<ul>
  <li class="actualites-row">
    <div class="views-field views-field-title"><span class="field-content">
      Circulaire FICTIONAL n°98-99 relative aux essais</span></div>
    <div class="views-field"><a href="/files/arrete-fictional.pdf">arrete-fictional.pdf</a></div>
  </li>
</ul>
<a href="?page=1" rel="next">Suivant</a>
"""

BAM_PAGE = SourcePage(url="https://bam.test/list", issuer="BAM")
AMMC_PAGE = SourcePage(url="https://ammc.test/list", issuer="AMMC", max_pages=3)


def test_parse_bam_listing() -> None:
    candidates, next_url = parse_listing(BAM_HTML, BAM_PAGE, BAM_PAGE.url)
    assert next_url is None
    assert [c.url for c in candidates] == [
        "https://bam.test/content/download/1/1/fictional-a.pdf",
        "https://bam.test/content/download/2/2/fictional-b.pdf",
    ]
    first, second = candidates
    assert first.title.startswith("Circulaire FICTIONAL n°97/G/99")
    assert first.reference == "97/G/2099"
    assert first.published == date(2099, 4, 1)
    assert second.reference is None
    assert second.published is None


def test_parse_ammc_listing_uses_row_title_and_pagination() -> None:
    candidates, next_url = parse_listing(AMMC_HTML, AMMC_PAGE, AMMC_PAGE.url)
    assert next_url == "https://ammc.test/list?page=1"
    assert len(candidates) == 1
    assert candidates[0].title == "Circulaire FICTIONAL n°98-99 relative aux essais"
    assert candidates[0].reference == "98-99"


@pytest.mark.parametrize(
    ("title", "issuer", "expected"),
    [
        ("Circulaire n°97/G/99 FICTIONAL", "BAM", "97/G/2099"),
        ("Circulaire 9W2098 FICTIONAL", "BAM", "9/W/2098"),
        ("Directive n° 8 W 97 FICTIONAL", "BAM", "8/W/2097"),
        ("Directive du 1er avril 2099 FICTIONAL", "BAM", None),
        ("Arrêté FICTIONAL n° 9999-98 du 2 mai 2098", "BAM", "9999-98"),
        ("Circulaire FICTIONAL n°99/XYZ/2098", "BAM", "99/XYZ/2098"),
        ("Lettre FICTIONAL n°09/XYZ/2098 appliquant la circulaire n°97/G/99", "BAM", "09/XYZ/2098"),
        ("Circulaire n° 07_98 FICTIONAL", "AMMC", "07-98"),
        ("Circulaire sans numéro FICTIONAL", "AMMC", None),
    ],
)
def test_parse_reference(title: str, issuer: str, expected: str | None) -> None:
    assert parse_reference(title, issuer) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("du 13 Août 2099", date(2099, 8, 13)),
        ("du 20 decembre 2098", date(2098, 12, 20)),
        ("du 1er février 2097", date(2097, 2, 1)),
        ("du 31 février 2097", None),
        ("sans date", None),
    ],
)
def test_parse_french_date(text: str, expected: date | None) -> None:
    assert parse_french_date(text) == expected


def _candidate(title: str, issuer: str = "BAM", url: str = "https://bam.test/a.pdf") -> Candidate:
    return Candidate(url, title, issuer, "fr", "https://bam.test/list", None, None)  # type: ignore[arg-type]


def test_select_filters_by_issuer_regex_and_limit() -> None:
    candidates = [
        _candidate("Liquidité FICTIONAL"),
        _candidate("Autre FICTIONAL", issuer="AMMC"),
        _candidate("Ratio de liquidité FICTIONAL"),
    ]
    assert len(select(candidates, issuer="bam")) == 2
    assert len(select(candidates, match="liquidit")) == 2
    assert len(select(candidates, match="liquidit", limit=1)) == 1


def test_document_stem_is_stable_and_ascii() -> None:
    stem = document_stem(_candidate("Circulaire été FICTIONAL"))
    assert stem == document_stem(_candidate("Circulaire été FICTIONAL"))
    assert stem.startswith("bam-circulaire-ete-fictional-")
    assert stem.isascii()


def _fetcher(routes: dict[str, httpx.Response]) -> Fetcher:
    def handler(request: httpx.Request) -> httpx.Response:
        return routes.get(str(request.url), httpx.Response(404))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return Fetcher(client, delay=0, sleep=lambda _: None)


def test_discover_follows_pagination_and_dedupes() -> None:
    fetcher = _fetcher(
        {
            "https://ammc.test/list": httpx.Response(200, text=AMMC_HTML),
            "https://ammc.test/list?page=1": httpx.Response(200, text=AMMC_HTML),
        }
    )
    assert len(discover([AMMC_PAGE], fetcher)) == 1


def test_robots_disallow_is_honoured() -> None:
    fetcher = _fetcher(
        {"https://bam.test/robots.txt": httpx.Response(200, text="User-agent: *\nDisallow: /")}
    )
    with pytest.raises(RobotsDisallowedError):
        fetcher.get("https://bam.test/list")


def test_retries_then_succeeds() -> None:
    responses = iter([httpx.Response(503), httpx.Response(200, text="ok")])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return next(responses)

    sleeps: list[float] = []
    fetcher = Fetcher(httpx.Client(transport=httpx.MockTransport(handler)), 0, 2, sleeps.append)
    assert fetcher.get("https://bam.test/x").text == "ok"
    assert 1.0 in sleeps


def test_download_writes_pdf_sidecar_and_manifest(tmp_path: Path) -> None:
    candidates, _ = parse_listing(BAM_HTML, BAM_PAGE, BAM_PAGE.url)
    good, not_pdf = candidates
    fetcher = _fetcher(
        {
            good.url: httpx.Response(200, content=b"%PDF-1.4 FICTIONAL"),
            not_pdf.url: httpx.Response(200, content=b"<html>error</html>"),
        }
    )

    pdf_path = download(good, tmp_path, fetcher)
    assert pdf_path is not None and pdf_path.read_bytes().startswith(b"%PDF")
    sidecar = yaml.safe_load(pdf_path.with_suffix(".yaml").read_text("utf-8"))
    assert sidecar["reference"] == "97/G/2099"
    assert sidecar["published"] == "2099-04-01"
    assert sidecar["needs_review"] is False
    assert download(not_pdf, tmp_path, fetcher) is None

    # Second call is a no-op: the file already exists.
    assert download(good, tmp_path, _fetcher({})) == pdf_path

    manifest = tmp_path / "sources.yaml"
    assert write_manifest(tmp_path, manifest) == 1
    entry = yaml.safe_load(manifest.read_text("utf-8"))["documents"][0]
    assert entry["file"] == pdf_path.relative_to(tmp_path).as_posix()


def test_selection_overrides_fill_missing_metadata(tmp_path: Path) -> None:
    selection = tmp_path / "selection.yaml"
    selection.write_text(
        "documents:\n"
        "  - url: https://bam.test/content/download/2/2/fictional-b.pdf\n"
        "    reference: 96/W/2099\n"
        "    published: 2099-03-04\n"
        "    notes: FICTIONAL - date read from the signature line\n",
        encoding="utf-8",
    )
    [entry] = load_selection(selection)
    _, no_metadata = parse_listing(BAM_HTML, BAM_PAGE, BAM_PAGE.url)[0]
    assert entry.url == no_metadata.url
    fetcher = _fetcher({no_metadata.url: httpx.Response(200, content=b"%PDF-1.4 FICTIONAL")})
    pdf_path = download(no_metadata, tmp_path, fetcher)
    assert pdf_path is not None and read_sidecar(pdf_path).needs_review

    assert apply_overrides(pdf_path, entry) is True
    metadata = read_sidecar(pdf_path)
    assert metadata.reference == "96/W/2099"
    assert metadata.published == date(2099, 3, 4)
    assert metadata.needs_review is False
    assert metadata.title.startswith("Note FICTIONAL")  # untouched fields are kept
    assert apply_overrides(pdf_path, entry) is False  # idempotent
