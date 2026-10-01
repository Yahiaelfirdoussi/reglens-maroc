"""Chunking strategies (pure functions).

- ``fixed_size_chunks``: fixed character windows with overlap (Phase 1 baseline).
- ``legal_chunks``: one chunk per article. Long articles are split on sentence boundaries
  with overlap, short neighbouring articles are merged, tables and lists stay attached to the
  sentence that introduces them, and documents without articles fall back to sentence
  windows.

Every chunk records the pages it covers, and legal chunks record their article label.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSpan:
    text: str
    page_start: int  # 1-based
    page_end: int
    section: str | None = None  # e.g. "Article 4", "Articles 3-4", None for a preamble


class _Document:
    """Pages joined into one string, with offset -> page lookup."""

    def __init__(self, pages: list[str]) -> None:
        self.starts: list[int] = []
        offset = 0
        for page in pages:
            self.starts.append(offset)
            offset += len(page) + 1  # joined with one space
        self.text = " ".join(pages)

    def span(self, start: int, end: int, section: str | None = None) -> TextSpan | None:
        piece = self.text[start:end]
        stripped = piece.strip()
        if not stripped:
            return None
        first = start + (len(piece) - len(piece.lstrip()))
        last = start + len(piece.rstrip()) - 1
        return TextSpan(
            stripped, bisect_right(self.starts, first), bisect_right(self.starts, last), section
        )


def _windows(text: str, start: int, end: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Character windows over ``text[start:end]``, cut on spaces."""
    bounds: list[tuple[int, int]] = []
    while start < end:
        stop = min(start + size, end)
        if stop < end:
            cut = text.rfind(" ", start + size // 2, stop)
            if cut != -1:
                stop = cut
        bounds.append((start, stop))
        if stop >= end:
            break
        next_start = max(stop - overlap, start + 1)
        space = text.find(" ", next_start, stop)
        start = space + 1 if space != -1 else next_start
    return bounds


def fixed_size_chunks(pages: list[str], size: int, overlap: int) -> list[TextSpan]:
    """Split a document into windows of at most ``size`` characters.

    Consecutive windows share about ``overlap`` characters. Windows end on a space when one
    exists in their second half, so words are not cut.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    doc = _Document(pages)
    spans = [doc.span(a, b) for a, b in _windows(doc.text, 0, len(doc.text), size, overlap)]
    return [s for s in spans if s is not None]


# --- Legal structure ------------------------------------------------------------------------

# Headings: "Article 4", "Article premier", "ARTICLE PREMIER. —", "ART. 2.", "Article 1.20",
# "Article 11 bis", Arabic "المادة 3". In-text references use a lower-case "article"
# ("l'article 5 ci-dessus") and are excluded by case and by what follows the number.
_HEADING = re.compile(
    r"(?<![’'\w])(?:Article|ARTICLE|Art\.|ART\.)\s*"
    r"(?P<num>premier|PREMIER|Premier|1er|\d+(?:\s*\.\s*\d+)*(?:\s+(?:bis|ter|quater))?)\b"
    r"(?!\s*(?:de la |de l[’']|du |des |ci-|précit|susvis|,|et |à |ou |alinéa))"
    r"|(?<!\w)المادة\s+(?P<ar>\d+|الأولى)"
)
_SENTENCE_BREAK = re.compile(r"(?<=[;:!?])\s+|(?<=[\w)»%]\.)\s+(?=[A-ZÀ-ÖØ-Þ«(\-–•\d])")
_NUMBERISH = re.compile(r"\d")


def _number_key(raw: str) -> tuple[int, ...]:
    value = raw.replace(" ", "").lower()
    if value in ("premier", "1er", "الأولى"):
        return (1,)
    value = re.sub(r"(bis|ter|quater)$", "", value)
    return tuple(int(part) for part in value.split(".") if part.isdigit())


def _label(raw: str) -> str:
    value = re.sub(r"\s+", "", raw.lower())
    value = re.sub(r"(bis|ter|quater)$", r" \1", value)
    return "Article premier" if value in ("premier", "1er", "الأولى") else f"Article {value}"


def find_articles(text: str) -> list[tuple[int, str]]:
    """(position, label) of every article heading, with isolated OCR misreads repaired.

    A number that breaks an otherwise regular sequence (11, 42, 13) is relabelled from its
    neighbours (11, 12, 13); the split position is unaffected.
    """
    found = [(m.start(), m.group("num") or m.group("ar")) for m in _HEADING.finditer(text)]
    keys = [_number_key(raw) for _, raw in found]
    labels = [_label(raw) for _, raw in found]
    for i in range(1, len(found) - 1):
        before, here, after = keys[i - 1], keys[i], keys[i + 1]
        simple = all(len(k) == 1 for k in (before, here, after))
        if simple and after[0] - before[0] == 2 and here[0] != before[0] + 1:
            labels[i] = f"Article {before[0] + 1}"
    return [(position, label) for (position, _), label in zip(found, labels, strict=True)]


def _is_table_like(text: str) -> bool:
    tokens = text.split()
    return (
        len(tokens) >= 12 and sum(bool(_NUMBERISH.search(t)) for t in tokens) / len(tokens) >= 0.25
    )


def _units(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Sentence-like units; a unit ending with ':' stays with what it introduces."""
    units: list[tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE_BREAK.finditer(text, start, end):
        units.append((cursor, match.start()))
        cursor = match.end()
    units.append((cursor, end))
    merged: list[tuple[int, int]] = []
    for unit in units:
        if merged and text[merged[-1][0] : merged[-1][1]].rstrip().endswith(":"):
            merged[-1] = (merged[-1][0], unit[1])
        else:
            merged.append(unit)
    return merged


def _pack(
    text: str, start: int, end: int, size: int, overlap: int, table_size: int
) -> list[tuple[int, int]]:
    """Pack sentence units into chunks of at most ``size`` characters, with overlap."""
    if end - start <= size:
        return [(start, end)]
    chunks: list[tuple[int, int]] = []
    current: list[tuple[int, int]] = []

    def flush() -> None:
        if current:
            chunks.append((current[0][0], current[-1][1]))

    for unit in _units(text, start, end):
        length = unit[1] - unit[0]
        if length > size:  # an oversized unit: keep a table whole, window anything else
            flush()
            current = []
            if _is_table_like(text[unit[0] : unit[1]]) and length <= table_size:
                chunks.append(unit)
            else:
                chunks.extend(_windows(text, unit[0], unit[1], size, overlap))
            continue
        if current and unit[1] - current[0][0] > size:
            flush()
            carried: list[tuple[int, int]] = []  # overlap: trailing units of the last chunk
            for previous in reversed(current):
                if unit[1] - previous[0] > size or current[-1][1] - previous[0] > overlap:
                    break
                carried.insert(0, previous)
            current = carried
        current.append(unit)
    flush()
    return chunks


def _merge_label(first: str | None, last: str | None) -> str | None:
    if first is None or last is None or first == last:
        return first or last
    return f"Articles {first.removeprefix('Article ')}-{last.removeprefix('Article ')}"


def legal_chunks(
    pages: list[str],
    size: int,
    overlap: int,
    min_size: int = 300,
    table_size: int | None = None,
) -> list[TextSpan]:
    """One chunk per article (split when long, merged with neighbours when short)."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    table_size = table_size or 2 * size
    doc = _Document(pages)
    text = doc.text
    articles = find_articles(text)

    # Sections: preamble (before the first heading) then one per article.
    sections: list[tuple[int, int, str | None]] = []
    boundaries = [position for position, _ in articles]
    if not articles or boundaries[0] > 0:
        sections.append((0, boundaries[0] if articles else len(text), None))
    for i, (position, heading) in enumerate(articles):
        stop = boundaries[i + 1] if i + 1 < len(articles) else len(text)
        sections.append((position, stop, heading))

    # Merge short neighbouring articles (never the preamble with an article).
    merged: list[tuple[int, int, str | None, str | None]] = []  # start, end, first, last
    for start, stop, label in sections:
        if (
            merged
            and label is not None
            and merged[-1][2] is not None
            and merged[-1][1] - merged[-1][0] < min_size
            and stop - merged[-1][0] <= size
        ):
            first_start, _, first_label, _ = merged[-1]
            merged[-1] = (first_start, stop, first_label, label)
        else:
            merged.append((start, stop, label, label))

    spans: list[TextSpan] = []
    for start, stop, first, last in merged:
        section = _merge_label(first, last)
        for a, b in _pack(text, start, stop, size, overlap, table_size):
            span = doc.span(a, b, section)
            if span is not None:
                spans.append(span)
    return spans
