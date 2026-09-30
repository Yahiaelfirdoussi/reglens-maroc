"""Text cleaning for extracted and OCR'd pages (pure functions).

Targets the artefacts measured on this corpus (see docs/corpus.md): justified text split into
one word per line, OCR'd article headings such as "Article12" or "Articie", the Bank
Al-Maghrib letterhead address repeated on every page, and Bulletin officiel running headers.
"""

import re
import unicodedata

_PAGE_NUMBER_LINE = re.compile(r"\s*[-–]?\s*\d{1,3}\s*[-–]?\s*")

_BOILERPLATE = [
    # Bank Al-Maghrib letterhead: "277, Boulevard Mohammed V - B.P 445 Rabat - ... www.bkam.ma"
    re.compile(
        r"277,?\s*Boulevard\s+Mohammed\s+V.{0,160}?(?:w{2,3}\.?\s*bkam\.ma|waw\.bkam\.ma)",
        re.IGNORECASE,
    ),
    # Bulletin officiel headers, both layouts:
    #   "N° 6784 bis — 3 chaoual 1440 (7-6-2019) BULLETIN OFFICIEL 1317"
    #   "850 BULLETIN OFFICIEL N° 7178 — 23 chaabane 1444 (16-3-2023)"
    re.compile(
        r"N[°º]\s*\d{4}(?:\s*bis)?\s*[—–-]\s*[^()]{3,40}\([^)]{5,15}\)\s*BULLETIN\s+OFFICIEL"
        r"(?:\s*\d{1,4}\b)?"
    ),
    re.compile(
        r"\b\d{1,4}\s*BULLETIN\s+OFFICIEL\s*N[°º]\s*\d{4}(?:\s*bis)?\s*[—–-]\s*[^()]{3,40}"
        r"\([^)]{5,15}\)"
    ),
    # Leftovers: an isolated upper-case running title (the body says "Bulletin officiel").
    re.compile(r"\bBULLETIN\s+OFFICIEL\b"),
]


def strip_page_number_lines(text: str) -> str:
    """Drop lines that are only a page number, among the first or last two lines."""
    lines = [line for line in text.splitlines() if line.strip()]
    edges = {0, 1, len(lines) - 2, len(lines) - 1}
    kept = [
        line
        for i, line in enumerate(lines)
        if not (i in edges and _PAGE_NUMBER_LINE.fullmatch(line))
    ]
    return "\n".join(kept)


def clean_page(text: str) -> str:
    """Return one page as a single normalised paragraph of running text."""
    text = unicodedata.normalize("NFC", text)
    text = strip_page_number_lines(text)
    text = re.sub(r"\s+", " ", text)  # re-joins words that OCR put on separate lines
    for pattern in _BOILERPLATE:
        text = pattern.sub(" ", text)
    text = re.sub(r"\bArticie\b", "Article", text)
    text = re.sub(r"\b(Article|ARTICLE)(\d)", r"\1 \2", text)
    text = re.sub(r"(\w)- (?=[a-zà-ÿ])", r"\1-", text)  # "ci- dessus" -> "ci-dessus"
    return re.sub(r"\s+", " ", text).strip()
