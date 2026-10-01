"""Cross-encoder reranking: re-score (question, chunk) pairs read together.

The first-stage search returns ``candidates`` chunks; the reranker keeps the best ``k``.
Local ONNX model via FastEmbed in production, a lexical-overlap stand-in for tests.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from reglens.models import ScoredChunk


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def scores(self, question: str, texts: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    """Local multilingual cross-encoder (ONNX, no PyTorch)."""

    def __init__(
        self,
        model: str,
        cache_dir: Path,
        batch_size: int = 32,
        max_chars: int = 0,
        threads: int | None = None,
    ) -> None:
        self._name = model
        self._cache_dir = cache_dir
        self._threads = threads
        self._batch_size = batch_size
        self._max_chars = max_chars
        self._model: Any = None  # loaded on first use: questions that skip reranking never pay

    @property
    def name(self) -> str:
        return self._name

    def _loaded(self) -> Any:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(
                self._name, cache_dir=str(self._cache_dir), threads=self._threads
            )
        return self._model

    def scores(self, question: str, texts: Sequence[str]) -> list[float]:
        if self._max_chars:  # an article's opening usually carries its rule
            texts = [text[: self._max_chars] for text in texts]
        model = self._loaded()
        return [float(s) for s in model.rerank(question, list(texts), batch_size=self._batch_size)]


class OverlapReranker:
    """Deterministic stand-in for tests: share of question words found in the text."""

    @property
    def name(self) -> str:
        return "overlap"

    def scores(self, question: str, texts: Sequence[str]) -> list[float]:
        words = set(re.findall(r"\w{3,}", question.lower()))
        return [
            len(words & set(re.findall(r"\w{3,}", text.lower()))) / (len(words) or 1)
            for text in texts
        ]


def rerank(
    reranker: Reranker, question: str, candidates: list[ScoredChunk], k: int
) -> list[ScoredChunk]:
    """Best ``k`` candidates by reranker score (ties keep first-stage order)."""
    if not candidates:
        return []
    scores = reranker.scores(question, [c.chunk.text for c in candidates])
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return [
        ScoredChunk(
            chunk=candidates[i].chunk, score=scores[i], first_stage_score=candidates[i].score
        )
        for i in order[:k]
    ]


def make_reranker(
    model: str, cache_dir: Path, max_chars: int = 0, threads: int | None = None
) -> Reranker | None:
    """``""`` disables reranking; ``overlap`` selects the test stand-in."""
    if not model:
        return None
    if model == "overlap":
        return OverlapReranker()
    return FastEmbedReranker(model, cache_dir, max_chars=max_chars, threads=threads)
