"""BM25 sparse vectors for exact-term matching (references, legal terms, figures).

Qdrant applies the IDF part server-side (sparse vector with ``Modifier.IDF``); documents carry
the BM25 term-frequency part, queries a weight of 1 per distinct term.

The tokenizer is tuned to this corpus: accents folded, French/English stop words removed, a
light plural strip, and regulatory references normalised to one token whatever the spelling
("n°5/W/2017", "5W2017", "5/W/17" -> "ref:5/w/2017"; AMMC "n° 01/20", "01-20" -> "ref:01-20").
"""

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

_STOPWORDS_TEXT = """
a au aux avec ce ces cet cette ci comme dans de des du elle elles en est et etre il ils
la le les leur leurs lui mais me meme mes moi mon ne ni nos notre nous on ou par pas pour
qu que quel quelle quelles quels qui sa sans se ses si son sont sur ta te tes toi ton tu un
une vos votre vous y d l n s c j m t ainsi apres avant cela celle celui ceux dont lors
selon sous tout tous toute toutes vers
the an and are as at be by can do does for from has have how in is it its of on or
that their this to was what when where which who why will with my your i you
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())

_BAM_REF = re.compile(r"(?<!\d)(\d{1,3})\s*/?\s*([gw])\s*/?\s*(\d{4}|\d{2})(?!\d)", re.IGNORECASE)
_AMMC_REF = re.compile(r"n\s*°\s*(\d{1,2})\s*[-_/]\s*(\d{4}|\d{2})(?!\d)", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")


def fold(text: str) -> str:
    """Lower-case and strip accents."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def reference_tokens(text: str) -> list[str]:
    tokens = []
    for number, series, year in _BAM_REF.findall(text):
        full_year = year if len(year) == 4 else f"20{year}"
        tokens.append(f"ref:{int(number)}/{series.lower()}/{full_year}")
    for number, year in _AMMC_REF.findall(text):
        tokens.append(f"ref:{int(number):02d}-{year}")
    return tokens


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 4 and word[-1] in "sx" else word


def tokenize(text: str) -> list[str]:
    words = [_stem(w) for w in _WORD.findall(fold(text)) if w not in STOPWORDS]
    return words + reference_tokens(text)


def term_index(term: str) -> int:
    """Stable 31-bit index for a term (collisions are negligible at this vocabulary size)."""
    return (
        int.from_bytes(hashlib.md5(term.encode(), usedforsecurity=False).digest()[:4], "little")
        >> 1
    )


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class BM25Encoder:
    """Document side of BM25 (k1, b); the collection-level IDF is applied by Qdrant."""

    def __init__(self, average_length: float, k1: float = 1.2, b: float = 0.75) -> None:
        self.average_length = max(average_length, 1.0)
        self.k1 = k1
        self.b = b

    @classmethod
    def fit(cls, texts: Iterable[str]) -> "BM25Encoder":
        lengths = [len(tokenize(text)) for text in texts]
        return cls(sum(lengths) / len(lengths) if lengths else 1.0)

    def encode_document(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        counts = Counter(term_index(t) for t in tokens)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / self.average_length)
        items = sorted(counts.items())
        return SparseVector(
            [i for i, _ in items], [tf * (self.k1 + 1) / (tf + norm) for _, tf in items]
        )

    @staticmethod
    def encode_query(text: str) -> SparseVector:
        indices = sorted({term_index(t) for t in tokenize(text)})
        return SparseVector(indices, [1.0] * len(indices))
