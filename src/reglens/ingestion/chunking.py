"""Chunking strategies (pure functions).

Phase 1 baseline: fixed-size character windows with overlap, cut on word boundaries and
spanning page breaks. Each window records the pages it covers so answers can cite pages.
"""

from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSpan:
    text: str
    page_start: int  # 1-based
    page_end: int


def fixed_size_chunks(pages: list[str], size: int, overlap: int) -> list[TextSpan]:
    """Split a document into windows of at most ``size`` characters.

    Consecutive windows share about ``overlap`` characters. Windows end on a space when one
    exists in their second half, so words are not cut.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")

    starts, parts, offset = [], [], 0
    for page in pages:
        starts.append(offset)
        parts.append(page)
        offset += len(page) + 1  # joined with one space
    text = " ".join(parts)

    def page_of(position: int) -> int:
        return bisect_right(starts, position)  # 1-based page number

    spans: list[TextSpan] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end]
        stripped = piece.strip()
        if stripped:
            first = start + (len(piece) - len(piece.lstrip()))
            last = start + len(piece.rstrip()) - 1
            spans.append(TextSpan(stripped, page_of(first), page_of(last)))
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)  # start the next window on a word
        start = space + 1 if space != -1 else next_start
    return spans
