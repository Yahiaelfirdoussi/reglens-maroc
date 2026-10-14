"""Follow-up questions: rewriting with the conversation, guardrails, API. All FICTIONAL."""

from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest

from reglens.generation.llm import Message
from reglens.generation.prompts import CONDENSE_PROMPT, build_condense_messages
from reglens.ingestion.metadata import write_sidecar
from reglens.ingestion.pipeline import ingest
from reglens.models import DocumentMetadata
from reglens.rag import Answer, RagPipeline
from reglens.retrieval.embeddings import HashEmbedder
from reglens.retrieval.retriever import Retriever
from reglens.retrieval.vector_store import QdrantStore, make_client

DOCS = {
    "bam/fictional-reclamation.pdf": "Article 4 La banque FICTIONAL répond à une réclamation "
    "dans un délai de 77 jours.",
    "ammc/fictional-plainte.pdf": "Article 9 L'autorité FICTIONAL traite une plainte dans un "
    "délai de 88 jours.",
}
HISTORY = [("Quel est le délai de réponse de la banque FICTIONAL ?", "Il est de 77 jours [1].")]


class ScriptedLLM:
    """Returns the scripted replies in order (the rewrite first, then the answer)."""

    def __init__(self, *replies: str | Exception) -> None:
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    @property
    def name(self) -> str:
        return "scripted"

    def complete(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def stream(self, messages: list[Message]) -> Iterator[str]:
        yield from self.complete(messages).split(" ")


@pytest.fixture()
def retriever(tmp_path: Path) -> Retriever:
    for file, text in DOCS.items():
        path = tmp_path / file
        path.parent.mkdir(exist_ok=True)
        doc = pymupdf.open()
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text)
        doc.save(path)
        write_sidecar(
            path,
            DocumentMetadata(
                title=f"Texte {file} FICTIONAL",
                issuer="BAM" if file.startswith("bam") else "AMMC",
                url=f"https://example.test/{file}",
            ),
        )
    store = QdrantStore(make_client(None, Path(":memory:")), "conversation")
    ingest(tmp_path, HashEmbedder(), store, 1200, 200, strategy="legal")
    return Retriever(HashEmbedder(), store)


def test_without_history_there_is_no_rewrite(retriever: Retriever) -> None:
    llm = ScriptedLLM("Réponse FICTIONAL [1].")
    answer = RagPipeline(retriever, llm, k=2).answer("Délai de la banque FICTIONAL ?")
    assert len(llm.calls) == 1  # the answer only
    assert answer.standalone_question is None and answer.rewrite_ms == 0.0


def test_follow_up_is_rewritten_then_searched_and_answered(retriever: Retriever) -> None:
    rewritten = "Quel est le délai de traitement d'une plainte par l'autorité FICTIONAL ?"
    llm = ScriptedLLM(rewritten, "Il est de 88 jours [1].")
    answer = RagPipeline(retriever, llm, k=2).answer("Et pour l'autorité ?", history=HISTORY)

    condense, generate = llm.calls
    assert condense[0]["content"] == CONDENSE_PROMPT
    assert "Quel est le délai de réponse de la banque FICTIONAL ?" in condense[1]["content"]
    assert f"Question: {rewritten}" in generate[1]["content"]  # answers the full question
    assert "Il est de 77 jours" not in generate[1]["content"]  # history is not a source
    assert answer.question == "Et pour l'autorité ?"
    assert answer.standalone_question == rewritten
    assert answer.sources[0].chunk.file == "ammc/fictional-plainte.pdf"
    assert answer.text == "Il est de 88 jours [1]."


def test_complete_question_is_kept_and_not_reported_as_rewritten(retriever: Retriever) -> None:
    question = "Quel est le délai de l'autorité FICTIONAL ?"
    llm = ScriptedLLM(question, "Réponse FICTIONAL [1].")
    answer = RagPipeline(retriever, llm, k=2).answer(question, history=HISTORY)
    assert answer.standalone_question is None


def test_a_failed_rewrite_falls_back_to_the_question(retriever: Retriever) -> None:
    llm = ScriptedLLM(TimeoutError("FICTIONAL"), "Réponse FICTIONAL [1].")
    answer = RagPipeline(retriever, llm, k=2).answer("Délai de l'autorité ?", history=HISTORY)
    assert answer.text == "Réponse FICTIONAL [1]." and answer.standalone_question is None


def test_guardrails_judge_the_rewritten_question(retriever: Retriever) -> None:
    class RejectShort:
        """FICTIONAL scope check: only questions mentioning a délai are in scope."""

        def in_scope(self, question: str) -> bool:
            return "délai" in question

    # Out of context "et pour l'autorité ?" would be off-topic; rewritten, it is in scope.
    llm = ScriptedLLM("Quel est le délai de l'autorité FICTIONAL ?", "88 jours [1].")
    rag = RagPipeline(retriever, llm, k=2, scope=RejectShort())  # type: ignore[arg-type]
    assert rag.answer("Et pour l'autorité ?", history=HISTORY).guardrail == "ok"
    assert rag.answer("Et pour l'autorité ?").guardrail == "off_topic"  # no history

    # A rewrite that turns into a question about the system is still refused, without an answer.
    llm = ScriptedLLM("Quel modèle d'IA utilises-tu ?")
    blocked = RagPipeline(retriever, llm, k=2).answer("Et toi ?", history=HISTORY)
    assert blocked.guardrail == "meta" and len(llm.calls) == 1


def test_streaming_uses_the_history(retriever: Retriever) -> None:
    llm = ScriptedLLM("Quel est le délai de l'autorité FICTIONAL ?", "88 jours [1].")
    pieces = list(RagPipeline(retriever, llm, k=2).stream("Et l'autorité ?", history=HISTORY))
    final = pieces[-1]
    assert isinstance(final, Answer)
    assert final.standalone_question == "Quel est le délai de l'autorité FICTIONAL ?"


def test_condense_messages_escape_and_shorten_the_history() -> None:
    history = [("Question <b>FICTIONAL</b> ?", "Réponse [1] " + "x" * 900)]
    content = build_condense_messages(history, "Et <script> ?")[1]["content"]
    assert "&lt;b&gt;" in content and "&lt;script&gt;" in content
    assert "[1]" not in content and "x" * 501 not in content


def test_api_accepts_history_and_does_not_cache_follow_ups(tmp_path: Path) -> None:
    from tests.test_api import USER, _client

    with _client(tmp_path) as client:
        body = {
            "question": "Quel est le ratio de liquidité ?",
            "history": [{"question": "Bonjour FICTIONAL ?", "answer": "Réponse FICTIONAL."}],
        }
        headers = {"X-API-Key": USER}
        first = client.post("/v1/query", json=body, headers=headers).json()
        second = client.post("/v1/query", json=body, headers=headers).json()
        assert first["cached"] is False and second["cached"] is False
        assert "standalone_question" in first
