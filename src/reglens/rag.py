"""RAG pipeline orchestration: guardrails, retrieval, abstention, cited generation.

Abstention happens in two layers:
1. Score floor: when even the best retrieved chunk is far from the question, a fixed
   "not found" answer is returned without calling the LLM.
2. The LLM: told to reply exactly ``NOT_FOUND`` when the sources do not contain the answer;
   that signal becomes the same fixed answer in the user's language.
"""

import time
from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel

from reglens.generation.citations import Citation, build_citations, strip_invalid_citations
from reglens.generation.llm import LLM
from reglens.generation.prompts import NOT_FOUND_SIGNAL, build_messages, not_found_answer
from reglens.guardrails import ScopeClassifier, check, fixed_answer
from reglens.models import Language, ScoredChunk
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

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def top_k(self) -> int:
        return self._k

    def _prepare(self, question: str) -> Answer | tuple[str, Language, list[ScoredChunk], float]:
        """Everything before the LLM. Returns a final Answer (guardrail, abstention, no LLM)
        or what the LLM needs: question, language, sources and retrieval time."""
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
        return question, language, sources, retrieval_ms

    def _finish(
        self,
        raw: str,
        question: str,
        language: Language,
        sources: list[ScoredChunk],
        retrieval_ms: float,
        generation_ms: float,
    ) -> Answer:
        """Turn the model's text into the final answer (abstention, citations, tokens)."""
        usage = getattr(self._llm, "last_usage", None) or {}
        common = {
            "question": question,
            "sources": sources,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        }
        if NOT_FOUND_SIGNAL in raw.upper():  # also when the model explains first
            return Answer(text=not_found_answer(language), citations=[], abstention="llm", **common)
        text = strip_invalid_citations(raw.strip(), len(sources))
        return Answer(text=text, citations=build_citations(text, sources), **common)

    def answer(self, question: str) -> Answer:
        prepared = self._prepare(question)
        if isinstance(prepared, Answer):
            return prepared
        question, language, sources, retrieval_ms = prepared
        assert self._llm is not None
        start = time.perf_counter()
        raw = self._llm.complete(build_messages(question, sources, language))
        generation_ms = (time.perf_counter() - start) * 1000
        return self._finish(raw, question, language, sources, retrieval_ms, generation_ms)

    def stream(self, question: str) -> Iterator[str | Answer]:
        """Yield the answer text as it is generated, then the final Answer.

        The final Answer is authoritative: if the model ends with NOT_FOUND, the streamed
        text is replaced by the fixed "not found" answer. Text that starts like the signal is
        held back so a user never sees "NOT_FOUND" appear.
        """
        prepared = self._prepare(question)
        if isinstance(prepared, Answer):
            yield prepared
            return
        question, language, sources, retrieval_ms = prepared
        assert self._llm is not None
        stream = getattr(self._llm, "stream", None)
        messages = build_messages(question, sources, language)
        start = time.perf_counter()
        if stream is None:
            raw = self._llm.complete(messages)
        else:
            parts: list[str] = []
            held = ""  # text that could still be the start of the signal
            released = False
            for delta in stream(messages):
                parts.append(delta)
                if released:
                    yield delta
                    continue
                held += delta
                head = held.lstrip().upper()
                if NOT_FOUND_SIGNAL.startswith(head) or head.startswith(NOT_FOUND_SIGNAL):
                    continue  # might be (or is) the signal: do not show it
                released = True
                yield held
            raw = "".join(parts)
        generation_ms = (time.perf_counter() - start) * 1000
        yield self._finish(raw, question, language, sources, retrieval_ms, generation_ms)
