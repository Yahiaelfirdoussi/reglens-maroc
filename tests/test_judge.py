"""Tests for the LLM-as-judge and cost reporting. Fake LLMs only; text is FICTIONAL."""

import json

import pytest

from reglens.evaluation.answers import evaluate_answers, summarize
from reglens.evaluation.golden import Evidence, GoldenItem
from reglens.evaluation.judge import Judge, build_judge_messages, parse_verdict
from reglens.generation.llm import FakeLLM
from reglens.models import Chunk, ScoredChunk
from reglens.rag import RagPipeline

ITEM = GoldenItem(
    id="q1",
    question="Quel est le ratio FICTIONAL ?",
    language="fr",
    answerable=True,
    theme="t",
    expected_section="Article 3",
    expected_facts=["5,5 %", "trimestriel"],
    evidence=Evidence(file="bam/f.pdf", page=1, quote="5,5 % des risques, contrôle trimestriel"),
)


def _source() -> ScoredChunk:
    chunk = Chunk(
        id="00000000-0000-0000-0000-000000000001",
        text="Article 3 Le ratio FICTIONAL est de 5,5 % des risques, contrôle trimestriel.",
        file="bam/f.pdf",
        index=0,
        page_start=1,
        page_end=1,
        issuer="BAM",
        title="FICTIONAL",
        url="https://example.test",
        section="Article 3",
    )
    return ScoredChunk(chunk=chunk, score=0.7)


def test_parse_verdict_clamps_and_tolerates_code_fences() -> None:
    raw = '```json\n{"claims": 4, "supported_claims": 3, "relevance": 9, "facts_conveyed": 5,'
    raw += ' "unsupported": ["x"], "missing_facts": []}\n```'
    verdict = parse_verdict(raw, facts_total=2)
    assert verdict.faithfulness == 0.75
    assert verdict.relevance == 5  # clamped to 1-5
    assert verdict.correctness == 1.0  # clamped to the number of expected facts
    assert parse_verdict('{"claims": 0}', 2).faithfulness == 1.0  # nothing claimed
    with pytest.raises(json.JSONDecodeError):
        parse_verdict("not json", 1)


def test_judge_prompt_carries_question_answer_sources_and_reference() -> None:
    _, user = build_judge_messages(ITEM, "Le ratio est de 5,5 % [1].", [_source()])
    for part in ("QUESTION:", "ANSWER:", "[1] (", "EXPECTED FACTS (2)", "OFFICIAL PASSAGE:"):
        assert part in user["content"]


class _Retriever:
    def retrieve(self, question: str, k: int) -> list[ScoredChunk]:
        return [_source()]


def test_evaluate_answers_with_judge_and_cost() -> None:
    answering = FakeLLM("Le ratio est de 5,5 % [1].")
    answering.last_usage = {"prompt_tokens": 1000, "completion_tokens": 100}  # type: ignore[attr-defined]
    verdict = {"claims": 1, "supported_claims": 1, "relevance": 4, "facts_conveyed": 1}
    judge = Judge(FakeLLM(json.dumps(verdict)))
    pipeline = RagPipeline(_Retriever(), answering, k=1)  # type: ignore[arg-type]
    rows = evaluate_answers([ITEM], pipeline, judge)
    assert rows[0].faithfulness == 1.0 and rows[0].relevance == 4 and rows[0].correctness == 0.5
    s = summarize(rows, price_in=2.0, price_out=8.0)
    assert s["cost_per_query_usd"] == pytest.approx((1000 * 2 + 100 * 8) / 1e6)
    assert summarize(rows)["cost_per_query_usd"] is None  # prices unknown


def test_judge_errors_do_not_stop_the_evaluation() -> None:
    pipeline = RagPipeline(_Retriever(), FakeLLM("Le ratio est de 5,5 % [1]."), k=1)  # type: ignore[arg-type]
    rows = evaluate_answers([ITEM], pipeline, Judge(FakeLLM("not json")))
    assert rows[0].faithfulness is None
    assert summarize(rows)["faithfulness"] is None
