"""RAG pipeline orchestration: retrieve, then (optionally) generate a cited answer."""

import time

from pydantic import BaseModel

from reglens.generation.citations import Citation, build_citations, strip_invalid_citations
from reglens.generation.llm import LLM
from reglens.generation.prompts import build_messages
from reglens.models import ScoredChunk
from reglens.retrieval.retriever import Retriever


class Answer(BaseModel):
    question: str
    text: str | None  # None when no LLM is configured (retrieval-only mode)
    citations: list[Citation]
    sources: list[ScoredChunk]
    retrieval_ms: float
    generation_ms: float | None = None


class RagPipeline:
    def __init__(self, retriever: Retriever, llm: LLM | None, k: int) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        sources = self._retriever.retrieve(question, self._k)
        retrieval_ms = (time.perf_counter() - start) * 1000
        if self._llm is None:
            return Answer(
                question=question,
                text=None,
                citations=[],
                sources=sources,
                retrieval_ms=retrieval_ms,
            )
        start = time.perf_counter()
        raw = self._llm.complete(build_messages(question, sources))
        text = strip_invalid_citations(raw, len(sources))
        return Answer(
            question=question,
            text=text,
            citations=build_citations(text, sources),
            sources=sources,
            retrieval_ms=retrieval_ms,
            generation_ms=(time.perf_counter() - start) * 1000,
        )
