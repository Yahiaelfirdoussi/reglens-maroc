"""Post-generation citation handling: keep only citations that point to real sources."""

import re

from pydantic import BaseModel

from reglens.models import ScoredChunk

_CITATION = re.compile(r"\[(\d+)\]")


class Citation(BaseModel):
    number: int
    issuer: str
    reference: str | None
    title: str
    page_start: int
    page_end: int
    url: str
    snippet: str


def strip_invalid_citations(answer: str, source_count: int) -> str:
    """Remove ``[n]`` markers whose number does not match a provided source."""

    def keep(match: re.Match[str]) -> str:
        return match.group(0) if 1 <= int(match.group(1)) <= source_count else ""

    return re.sub(r"[ \t]+([.,;:])", r"\1", _CITATION.sub(keep, answer))


def build_citations(
    answer: str, sources: list[ScoredChunk], snippet_chars: int = 240
) -> list[Citation]:
    """Citation objects for the sources actually cited, in order of first appearance."""
    cited: list[int] = []
    for match in _CITATION.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= len(sources) and number not in cited:
            cited.append(number)
    return [
        Citation(
            number=n,
            issuer=sources[n - 1].chunk.issuer,
            reference=sources[n - 1].chunk.reference,
            title=sources[n - 1].chunk.title,
            page_start=sources[n - 1].chunk.page_start,
            page_end=sources[n - 1].chunk.page_end,
            url=sources[n - 1].chunk.url,
            snippet=sources[n - 1].chunk.text[:snippet_chars],
        )
        for n in cited
    ]
