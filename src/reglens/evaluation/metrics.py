"""Retrieval metrics (pure functions)."""

import re
from collections.abc import Sequence

from reglens.evaluation.text_metrics import normalize
from reglens.models import Chunk

PASSAGE_COVERAGE = 0.5


def tokens(text: str) -> list[str]:
    return re.findall(r"\w{2,}", normalize(text).lower())


def quote_coverage(quote: str, text: str) -> float:
    """Share of the quote's words that appear in ``text``."""
    quote_tokens = tokens(quote)
    if not quote_tokens:
        return 0.0
    present = set(tokens(text))
    return sum(token in present for token in quote_tokens) / len(quote_tokens)


def is_passage_hit(chunk: Chunk, file: str, quote: str) -> bool:
    """Right document AND the chunk contains most of the evidence quote."""
    return chunk.file == file and quote_coverage(quote, chunk.text) >= PASSAGE_COVERAGE


def first_hit_rank(hits: Sequence[bool]) -> int | None:
    """1-based rank of the first relevant result, or None."""
    return next((rank for rank, hit in enumerate(hits, start=1) if hit), None)


def hit_rate(ranks: Sequence[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks) if ranks else 0.0


def mrr(ranks: Sequence[int | None], k: int) -> float:
    """Mean reciprocal rank, counting only hits within the top ``k``."""
    if not ranks:
        return 0.0
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[index]
