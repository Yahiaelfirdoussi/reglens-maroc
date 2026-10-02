"""Tests for abstention and answer evaluation. No network; text here is FICTIONAL."""

from reglens.evaluation.answers import citation_coverage, evaluate_answers, fact_found, summarize
from reglens.evaluation.golden import Evidence, GoldenItem
from reglens.generation.llm import FakeLLM
from reglens.generation.prompts import NOT_FOUND_SIGNAL, SYSTEM_PROMPT
from reglens.models import Chunk, ScoredChunk
from reglens.rag import RagPipeline


class _StubRetriever:
    def __init__(self, score: float) -> None:
        self.score = score

    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        chunk = Chunk(
            id="00000000-0000-0000-0000-000000000001",
            text="Article 3 Le ratio FICTIONAL est au moins égal à 5,5 % des risques.",
            file="bam/fictional.pdf",
            index=0,
            page_start=1,
            page_end=1,
            issuer="BAM",
            title="FICTIONAL",
            url="https://example.test",
            section="Article 3",
        )
        return [ScoredChunk(chunk=chunk, score=self.score)]


def _pipeline(score: float, llm: FakeLLM, min_score: float = 0.35) -> RagPipeline:
    return RagPipeline(_StubRetriever(score), llm, k=1, min_score=min_score)  # type: ignore[arg-type]


def test_facts_match_across_number_formats() -> None:
    assert fact_found("5,5 %", "The ratio is 5.5% of risks")
    assert fact_found("5,5 %", "النسبة ٥٫٥٪ من المخاطر")
    assert fact_found("quarante jours ouvrés", "dans un délai de quarante  jours ouvrés")
    assert not fact_found("55 %", "5,5 %")
    assert not fact_found("5 %", "un taux de 25 %")  # part of a longer number
    assert not fact_found("30", "300 jours")
    assert fact_found("30 jours", "dans les 30 jours")
    assert fact_found("30 jours ouvrables", "dans un délai de trente jours ouvrables")
    assert fact_found("quatre (4) mois", "dans les 4 mois")
    assert fact_found("deux cent mille (200.000) dirhams", "limitée à 200 000 dirhams")
    assert fact_found("1 250 %", "une pondération de 1250 %")


def test_numbers_found_ignores_the_answer_language() -> None:
    from reglens.evaluation.answers import numbers_found

    assert numbers_found("30 jours ouvrables", "within 30 business days")
    assert numbers_found("5,5 %", "النسبة ٥٫٥٪")
    assert numbers_found("quarante jours ouvrés", "within forty working days")
    assert not numbers_found("5,5 %", "un ratio de 5 %")
    assert not numbers_found("gratuitement", "free of charge")  # not a numeric fact


def test_citation_coverage() -> None:
    assert citation_coverage("Le ratio est de 100%.[1] Il porte sur les 30 jours.[1]") == 1.0
    assert citation_coverage("Le ratio est de 5,5 % [1]. Il est fixé par la circulaire.") == 0.5
    assert citation_coverage("Oui [1].") == 1.0


def test_low_score_abstains_without_calling_the_llm() -> None:
    llm = FakeLLM()
    answer = _pipeline(0.2, llm).answer("Quel est le ratio FICTIONAL ?")
    assert answer.abstention == "low_score"
    assert answer.text is not None and answer.text.startswith("Je n'ai pas trouvé")
    assert llm.calls == []


def test_llm_not_found_signal_becomes_a_fixed_answer_in_the_question_language() -> None:
    answer = _pipeline(0.6, FakeLLM(NOT_FOUND_SIGNAL)).answer("What is the FICTIONAL ratio?")
    assert answer.abstention == "llm"
    assert answer.text is not None and answer.text.startswith("I could not find")
    assert answer.citations == []


def test_not_found_after_an_explanation_still_abstains() -> None:
    llm = FakeLLM(f"Les sources ne précisent pas ce taux FICTIONAL. {NOT_FOUND_SIGNAL}")
    assert _pipeline(0.6, llm).answer("Quel est le taux FICTIONAL ?").abstention == "llm"


def test_question_language_is_stated_to_the_model() -> None:
    llm = FakeLLM("The ratio is 5.5% [1].")
    _pipeline(0.6, llm).answer("What is the FICTIONAL ratio?")
    assert llm.calls[0][1]["content"].endswith("Answer in English.")


def test_prompt_requires_the_not_found_signal() -> None:
    assert f"reply with exactly {NOT_FOUND_SIGNAL}" in SYSTEM_PROMPT.replace("\n", " ")


def test_evaluate_answers_scores_facts_citations_and_abstention() -> None:
    answerable = GoldenItem(
        id="q1",
        question="Quel est le ratio FICTIONAL ?",
        language="fr",
        answerable=True,
        theme="t",
        expected_section="Article 3",
        expected_facts=["5,5 %"],
        evidence=Evidence(file="bam/fictional.pdf", page=1, quote="5,5 % des risques"),
    )
    unanswerable = GoldenItem(
        id="u1", question="Quel est le taux FICTIONAL ?", language="fr", answerable=False, theme="u"
    )
    rows = evaluate_answers([answerable], _pipeline(0.6, FakeLLM("Le ratio est de 5,5 % [1].")))
    rows += evaluate_answers([unanswerable], _pipeline(0.6, FakeLLM(NOT_FOUND_SIGNAL)))
    s = summarize(rows)
    assert s["fact_recall"] == 1.0
    assert s["cites_document"] == 1.0 and s["cites_article"] == 1.0
    assert s["false_abstention"] == 0.0
    assert s["abstention_accuracy"] == 1.0


def _streamed(pipeline: RagPipeline, question: str) -> tuple[str, object]:
    pieces = list(pipeline.stream(question))
    text = "".join(p for p in pieces if isinstance(p, str))
    return text, pieces[-1]


def test_stream_yields_text_then_the_final_answer() -> None:
    llm = FakeLLM("Le ratio est de 5,5 % [1].")
    text, final = _streamed(_pipeline(0.6, llm), "Quel est le ratio FICTIONAL ?")
    assert text.strip() == "Le ratio est de 5,5 % [1]."
    assert getattr(final, "text", None) == "Le ratio est de 5,5 % [1]."
    assert [c.number for c in getattr(final, "citations", [])] == [1]


def test_stream_never_shows_the_not_found_signal() -> None:
    text, final = _streamed(_pipeline(0.6, FakeLLM(NOT_FOUND_SIGNAL)), "Quel taux FICTIONAL ?")
    assert text == ""
    assert getattr(final, "abstention", None) == "llm"


def test_stream_releases_text_that_only_looked_like_the_signal() -> None:
    llm = FakeLLM("No, la banque FICTIONAL ne peut pas [1].")  # starts like "NOT_FOUND"
    text, final = _streamed(_pipeline(0.6, llm), "La banque FICTIONAL peut-elle ?")
    assert text.strip() == "No, la banque FICTIONAL ne peut pas [1]."
    assert getattr(final, "abstention", None) == "none"


def test_stream_returns_guardrail_answers_directly() -> None:
    pieces = list(_pipeline(0.6, FakeLLM()).stream("Qui t'a créé ?"))
    assert len(pieces) == 1 and getattr(pieces[0], "guardrail", None) == "meta"


class _FailingRetriever:
    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        raise ConnectionError("network down (FICTIONAL)")


def test_a_failing_question_is_recorded_and_the_run_continues() -> None:
    items = [
        GoldenItem(
            id="u1", question="Taux FICTIONAL ?", language="fr", answerable=False, theme="u"
        ),
        GoldenItem(
            id="u2", question="Délai FICTIONAL ?", language="fr", answerable=False, theme="u"
        ),
    ]
    pipeline = RagPipeline(_FailingRetriever(), FakeLLM(), k=1)  # type: ignore[arg-type]
    rows = evaluate_answers(items, pipeline)
    assert [r.error is not None for r in rows] == [True, True]
    assert summarize(rows)["n"] == 0  # errors are excluded from the metrics
