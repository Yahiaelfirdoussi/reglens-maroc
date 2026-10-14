"""RAG pipeline orchestration: guardrails, retrieval, abstention, cited generation.

Abstention happens in two layers:
1. Score floor: when even the best retrieved chunk is far from the question, a fixed
   "not found" answer is returned without calling the LLM.
2. The LLM: told to reply exactly ``NOT_FOUND`` when the sources do not contain the answer;
   that signal becomes the same fixed answer in the user's language.

Follow-up questions ("et pour l'AMMC ?") are first rewritten into a standalone question
using the last few turns; that question is what gets checked, searched and answered. Earlier
answers only resolve references: they are never given to the model as sources.
"""

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import structlog
from pydantic import BaseModel

from reglens.generation.citations import Citation, build_citations, strip_invalid_citations
from reglens.generation.llm import LLM
from reglens.generation.prompts import (
    NOT_FOUND_SIGNAL,
    build_condense_messages,
    build_messages,
    not_found_answer,
)
from reglens.guardrails import (
    MAX_QUESTION_CHARS,
    GuardrailResult,
    ScopeClassifier,
    check,
    fixed_answer,
)
from reglens.models import Language, ScoredChunk
from reglens.retrieval.retriever import Retriever
from reglens.retrieval.vector_store import SearchFilter
from reglens.text import detect_language

log = structlog.get_logger(__name__)

Abstention = Literal["none", "low_score", "llm"]
History = Sequence[tuple[str, str]]  # earlier (question, answer text) turns, oldest first
MAX_HISTORY_TURNS = 3


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
    # A follow-up rewritten from the conversation: the question actually answered.
    standalone_question: str | None = None
    rewrite_ms: float = 0.0


@dataclass
class _Context:
    """What the LLM needs once guardrails and retrieval have passed."""

    question: str  # as asked (sanitised)
    standalone: str  # what is searched and answered
    language: Language
    sources: list[ScoredChunk]
    retrieval_ms: float
    rewrite_ms: float
    rewrite_tokens: tuple[int, int]

    @property
    def rewritten(self) -> str | None:
        return self.standalone if self.standalone != self.question else None


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

    def condense(self, history: History, question: str) -> tuple[str, tuple[int, int]]:
        """Standalone form of a follow-up question (the question itself on any failure)."""
        assert self._llm is not None
        try:
            raw = self._llm.complete(build_condense_messages(history, question))
        except Exception as error:  # a failed rewrite must not fail the answer
            log.warning("rewrite_failed", error=type(error).__name__)
            return question, (0, 0)
        usage = getattr(self._llm, "last_usage", None) or {}
        tokens = (int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        rewritten = lines[0].strip("\"'«»“” ") if lines else ""
        if not rewritten or len(rewritten) > MAX_QUESTION_CHARS:
            return question, tokens
        return rewritten, tokens

    def _blocked(self, guard: GuardrailResult, asked: str, rewrite_ms: float = 0.0) -> Answer:
        return Answer(
            question=asked,
            text=fixed_answer(guard),
            citations=[],
            sources=[],
            retrieval_ms=0.0,
            guardrail=guard.verdict,
            rewrite_ms=rewrite_ms,
        )

    def _prepare(
        self, question: str, where: SearchFilter | None = None, history: History = ()
    ) -> Answer | _Context:
        """Everything before the LLM. Returns a final Answer (guardrail, abstention, no LLM)
        or what the LLM needs. ``where`` restricts the search; nothing matching it means
        "not found". ``history`` turns a follow-up into a standalone question."""
        turns = list(history)[-MAX_HISTORY_TURNS:]
        follow_up = bool(turns) and self._llm is not None
        # A follow-up is only judged off-topic once its context is resolved.
        guard = check(question, None if follow_up else self._scope)
        if not guard.allowed:
            return self._blocked(guard, guard.question)
        asked = standalone = guard.question
        rewrite_ms, rewrite_tokens = 0.0, (0, 0)
        if follow_up:
            start = time.perf_counter()
            standalone, rewrite_tokens = self.condense(turns, asked)
            rewrite_ms = (time.perf_counter() - start) * 1000
            guard = check(standalone, self._scope)
            if not guard.allowed:
                return self._blocked(guard, asked, rewrite_ms)
            standalone = guard.question
            log.info("question_rewritten", changed=standalone != asked, ms=round(rewrite_ms))
        language = detect_language(standalone)
        start = time.perf_counter()
        sources = self._retriever.retrieve(standalone, self._k, where)
        context = _Context(
            asked,
            standalone,
            language,
            sources,
            (time.perf_counter() - start) * 1000,
            rewrite_ms,
            rewrite_tokens,
        )
        if not sources or sources[0].score < self._min_score:
            return Answer(
                question=asked,
                text=not_found_answer(language),
                citations=[],
                sources=sources,
                retrieval_ms=context.retrieval_ms,
                abstention="low_score",
                standalone_question=context.rewritten,
                rewrite_ms=rewrite_ms,
            )
        if self._llm is None:
            return Answer(
                question=asked,
                text=None,
                citations=[],
                sources=sources,
                retrieval_ms=context.retrieval_ms,
                standalone_question=context.rewritten,
                rewrite_ms=rewrite_ms,
            )
        return context

    def _finish(self, raw: str, context: _Context, generation_ms: float) -> Answer:
        """Turn the model's text into the final answer (abstention, citations, tokens)."""
        usage = getattr(self._llm, "last_usage", None) or {}
        sources = context.sources
        abstained = NOT_FOUND_SIGNAL in raw.upper()  # also when the model explains first
        text = (
            not_found_answer(context.language)
            if abstained
            else strip_invalid_citations(raw.strip(), len(sources))
        )
        return Answer(
            question=context.question,
            text=text,
            citations=[] if abstained else build_citations(text, sources),
            sources=sources,
            retrieval_ms=context.retrieval_ms,
            generation_ms=generation_ms,
            abstention="llm" if abstained else "none",
            prompt_tokens=int(usage.get("prompt_tokens", 0)) + context.rewrite_tokens[0],
            completion_tokens=int(usage.get("completion_tokens", 0)) + context.rewrite_tokens[1],
            standalone_question=context.rewritten,
            rewrite_ms=context.rewrite_ms,
        )

    def answer(
        self, question: str, where: SearchFilter | None = None, history: History = ()
    ) -> Answer:
        prepared = self._prepare(question, where, history)
        if isinstance(prepared, Answer):
            return prepared
        assert self._llm is not None
        start = time.perf_counter()
        messages = build_messages(prepared.standalone, prepared.sources, prepared.language)
        raw = self._llm.complete(messages)
        return self._finish(raw, prepared, (time.perf_counter() - start) * 1000)

    def stream(
        self, question: str, where: SearchFilter | None = None, history: History = ()
    ) -> Iterator[str | Answer]:
        """Yield the answer text as it is generated, then the final Answer.

        The final Answer is authoritative: if the model ends with NOT_FOUND, the streamed
        text is replaced by the fixed "not found" answer. Text that starts like the signal is
        held back so a user never sees "NOT_FOUND" appear.
        """
        prepared = self._prepare(question, where, history)
        if isinstance(prepared, Answer):
            yield prepared
            return
        assert self._llm is not None
        stream = getattr(self._llm, "stream", None)
        messages = build_messages(prepared.standalone, prepared.sources, prepared.language)
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
        yield self._finish(raw, prepared, (time.perf_counter() - start) * 1000)
