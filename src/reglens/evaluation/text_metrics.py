"""Text-similarity metrics used to measure OCR quality against a reference text."""

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

_QUOTES = str.maketrans(
    {"’": "'", "‘": "'", "«": '"', "»": '"', "“": '"', "”": '"', "–": "-", "—": "-"}
)


def normalize(text: str) -> str:
    """NFC, unify quotes/dashes and collapse whitespace, so layout differences do not count."""
    text = unicodedata.normalize("NFC", text).translate(_QUOTES)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: Sequence[object], b: Sequence[object]) -> int:
    """Edit distance between two sequences (characters or words)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (item_a != item_b))
            )
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[object], hypothesis: Sequence[object]) -> tuple[int, int]:
    """Return (edits, reference length) so rates can be micro-averaged across pages."""
    return levenshtein(reference, hypothesis), len(reference)


def cer_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    return error_rate(normalize(reference), normalize(hypothesis))


def wer_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    return error_rate(normalize(reference).split(), normalize(hypothesis).split())


def bag_of_words_f1(reference: str, hypothesis: str) -> float:
    """Order-insensitive word overlap: separates recognition errors from reading-order ones."""
    ref, hyp = Counter(normalize(reference).split()), Counter(normalize(hypothesis).split())
    overlap = sum((ref & hyp).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / sum(hyp.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*(?:\s*%)?")


def number_tokens(text: str) -> Counter[str]:
    """Numbers and percentages ("100 %" -> "100%"): the facts a regulatory answer quotes."""
    return Counter(re.sub(r"\s+", "", m) for m in _NUMBER.findall(normalize(text)))


def strip_page_number(text: str, page_number: int) -> str:
    """Drop the page's own number when it stands alone in the first or last two lines.

    Matching the actual page number (not "any short number") keeps table cells that happen
    to end a page, e.g. a lone "20" in a risk-weight grid.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    edges = {0, 1, len(lines) - 2, len(lines) - 1}
    pattern = re.compile(rf"\s*[-–]?\s*{page_number}\s*[-–]?\s*")
    for i in sorted(edges, reverse=True):  # header and footer may both carry it
        if 0 <= i < len(lines) and pattern.fullmatch(lines[i]):
            del lines[i]
    return "\n".join(lines)


def number_counts(
    reference: str, hypothesis: str, page_number: int | None = None
) -> tuple[int, int]:
    """Return (numbers recovered, numbers in reference), ignoring the page's own number."""
    if page_number is not None:
        reference = strip_page_number(reference, page_number)
        hypothesis = strip_page_number(hypothesis, page_number)
    ref, hyp = number_tokens(reference), number_tokens(hypothesis)
    return sum((ref & hyp).values()), sum(ref.values())
