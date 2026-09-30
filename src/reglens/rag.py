"""RAG pipeline orchestration: retrieve, then (optionally) generate a cited answer."""

import time

from pydantic import BaseModel

from reglens.generation.citations import Citation, build_citations, strip_invalid_citations
from reglens.generation.llm import LLM
from reglens.generation.prompts import build_messages
from reglens.guardrails import ScopeClassifier, check, fixed_answer
from reglens.models import ScoredChunk
from reglens.retrieval.retriever import Retriever


class Answer(BaseModel):
    question: str
    text: str | None  # None when no LLM is configured (retrieval-only mode)
    citations: list[Citation]
    sources: list[ScoredChunk]
    retrieval_ms: float
    generation_ms: float | None = None
    # "ok" when the question went through; otherwise the guardrail that answered it
    # ("meta", "off_topic", "greeting", "too_long", "empty") without retrieval or LLM.
    guardrail: str = "ok"


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM | None,
        k: int,
        scope: ScopeClassifier | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k
        self._scope = scope

    def answer(self, question: str) -> Answer:
        guard = check(question, self._scope)
        if not guard.allowed:
            return Answer(
                question=guard.question,
                text=fixed_answer(guard),
                citations=[],
                sources=[],
                retrieval_ms=0.0,
                guardrail=guard.verdict,
            )
        question = guard.question
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
