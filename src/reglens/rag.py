"""RAG pipeline orchestration: guardrails, retrieval, abstention, cited generation.

Abstention happens in two layers:
1. Score floor: when even the best retrieved chunk is far from the question, a fixed
   "not found" answer is returned without calling the LLM.
2. The LLM: told to reply exactly ``NOT_FOUND`` when the sources do not contain the answer;
   that signal becomes the same fixed answer in the user's language.
"""

import time
from typing import Literal

from pydantic import BaseModel

from reglens.generation.citations import Citation, build_citations, strip_invalid_citations
from reglens.generation.llm import LLM
from reglens.generation.prompts import NOT_FOUND_SIGNAL, build_messages, not_found_answer
from reglens.guardrails import ScopeClassifier, check, fixed_answer
from reglens.models import ScoredChunk
from reglens.retrieval.retriever import Retriever
from reglens.text import detect_language

Abstention = Literal["none", "low_score", "llm"]


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
    # Why the assistant said the texts do not contain the answer, if it did.
    abstention: Abstention = "none"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM | None,
        k: int,
        scope: ScopeClassifier | None = None,
        min_score: float = 0.0,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k
        self._scope = scope
        self._min_score = min_score

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
        language = detect_language(question)
        start = time.perf_counter()
        sources = self._retriever.retrieve(question, self._k)
        retrieval_ms = (time.perf_counter() - start) * 1000

        if not sources or sources[0].score < self._min_score:
            return Answer(
                question=question,
                text=not_found_answer(language),
                citations=[],
                sources=sources,
                retrieval_ms=retrieval_ms,
                abstention="low_score",
            )
        if self._llm is None:
            return Answer(
                question=question,
                text=None,
                citations=[],
                sources=sources,
                retrieval_ms=retrieval_ms,
            )

        start = time.perf_counter()
        raw = self._llm.complete(build_messages(question, sources, language))
        generation_ms = (time.perf_counter() - start) * 1000
        usage = getattr(self._llm, "last_usage", None) or {}
        tokens = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }
        if NOT_FOUND_SIGNAL in raw.upper():  # also when the model explains first
            return Answer(
                question=question,
                text=not_found_answer(language),
                citations=[],
                sources=sources,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                abstention="llm",
                **tokens,
            )
        text = strip_invalid_citations(raw, len(sources))
        return Answer(
            question=question,
            text=text,
            citations=build_citations(text, sources),
            sources=sources,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            **tokens,
        )
