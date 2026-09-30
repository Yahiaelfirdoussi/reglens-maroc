"""Tests for text cleaning and fixed-size chunking. All text here is FICTIONAL."""

import pytest

from reglens.ingestion.chunking import fixed_size_chunks
from reglens.ingestion.cleaning import clean_page, strip_page_number_lines


def test_clean_page_rejoins_ocr_lines_and_fixes_articles() -> None:
    ocr = "Article12\nLes\nétablissements\nFICTIONAL\nci-\ndessus\nArticie 13\nTexte.\n7\n"
    assert clean_page(ocr) == (
        "Article 12 Les établissements FICTIONAL ci-dessus Article 13 Texte."
    )


def test_clean_page_removes_letterhead_and_bulletin_headers() -> None:
    page = (
        "N° 9999 bis — 3 chaoual 2099 (7-6-2099) BULLETIN OFFICIEL 1234\n"
        "Article 1 Règle FICTIONAL publiée au Bulletin officiel.\n"
        "277, Boulevard Mohammed V - B.P 445 Rabat - Maroc - Tél. : (212) 537 81 81 81 "
        "www.bkam.ma\n"
        "Article 2 Suite FICTIONAL."
    )
    assert clean_page(page) == (
        "Article 1 Règle FICTIONAL publiée au Bulletin officiel. Article 2 Suite FICTIONAL."
    )


def test_strip_page_number_lines_only_at_edges() -> None:
    assert strip_page_number_lines("3\nTexte FICTIONAL\n20\nFin\n4") == "Texte FICTIONAL\n20\nFin"


def test_fixed_size_chunks_cover_text_with_overlap_and_pages() -> None:
    pages = [" ".join(f"mot{i}" for i in range(60)), " ".join(f"page2mot{i}" for i in range(60))]
    spans = fixed_size_chunks(pages, size=200, overlap=40)
    assert all(len(s.text) <= 200 for s in spans)
    assert spans[0].page_start == 1
    assert spans[-1].page_end == 2
    assert any(s.page_start == 1 and s.page_end == 2 for s in spans)  # crosses the break
    words = {w for s in spans for w in s.text.split()}
    assert words == set(pages[0].split()) | set(pages[1].split())  # nothing lost
    assert not any(w.startswith("ot") or w.startswith("ge2") for w in words)  # no cut words
    first, second = spans[0].text.split(), spans[1].text.split()
    assert set(first) & set(second)  # consecutive windows overlap


def test_fixed_size_chunks_validates_arguments() -> None:
    with pytest.raises(ValueError):
        fixed_size_chunks(["texte"], size=100, overlap=100)
    assert fixed_size_chunks(["", ""], size=100, overlap=10) == []
