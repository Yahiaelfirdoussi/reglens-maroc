"""Answer evaluation: fact recall, citations, abstention, latency and tokens.

Facts are matched after normalisation, so "5,5 %", "5.5%" and Arabic-Indic "٥٫٥٪" are the
same fact. Citations are checked against the evidence: the answer should cite a chunk from
the expected document (and ideally the expected article).
"""

import json
import re
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import structlog

from reglens.evaluation import metrics as m
from reglens.evaluation.golden import GoldenItem
from reglens.evaluation.judge import Judge
from reglens.evaluation.text_metrics import normalize
from reglens.rag import Answer, RagPipeline

log = structlog.get_logger(__name__)

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٪", "0123456789,%")
# A sentence keeps the citations written right after its final punctuation ("…100%.[1]").
_SENTENCE = re.compile(r"[^.!?؟\n]+(?:[.!?؟]+|$)(?:\s*\[\d+\])*")
_NUMBER_WORDS = {
    # French
    "un": "1",
    "une": "1",
    "deux": "2",
    "trois": "3",
    "quatre": "4",
    "cinq": "5",
    "six": "6",
    "sept": "7",
    "huit": "8",
    "neuf": "9",
    "dix": "10",
    "onze": "11",
    "douze": "12",
    "quinze": "15",
    "vingt": "20",
    "trente": "30",
    "quarante": "40",
    "quarante-cinq": "45",
    "cinquante": "50",
    "soixante": "60",
    "quatre-vingt-dix": "90",
    # English
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "ten": "10",
    "twelve": "12",
    "fifteen": "15",
    "thirty": "30",
    "forty": "40",
    "forty-five": "45",
    "sixty": "60",
    "ninety": "90",
}
_SPELLED_THEN_DIGITS = re.compile(
    r"((?:(?:"
    + "|".join(
        sorted(
            map(re.escape, [*_NUMBER_WORDS, "cent", "cents", "mille", "et"]), key=len, reverse=True
        )
    )
    + r")[\s-]*){1,5})\((\d[\d.,\s]*)\)"
)
_NUMBER_WORD = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _NUMBER_WORDS), key=len, reverse=True)) + r")\b"
)
_CITATION = re.compile(r"\[\d+\]")


def fact_form(text: str) -> str:
    """Comparison form for facts: digits unified, number words as digits, no spaces,
    decimal point == comma."""
    text = normalize(text).translate(_ARABIC_DIGITS).lower()
    # Legal style "quatre (4) mois", "deux cent mille (200.000) dirhams": keep the digits.
    text = _SPELLED_THEN_DIGITS.sub(lambda m: m.group(2), text)
    text = _NUMBER_WORD.sub(lambda m: _NUMBER_WORDS[m.group(1)], text)
    text = re.sub(r"(?<=\d)[ .,](?=\d{3}(?!\d))", "", text)  # thousands: 200.000 = 200 000
    text = re.sub(r"(?<=\d)[.,](?=\d)", ",", text)
    return re.sub(r"\s+", "", text)


def fact_found(fact: str, answer: str) -> bool:
    """Fact present in the answer; a number must not be part of a longer one ("5%" ≠ "25%")."""
    needle = fact_form(fact)
    if not needle:
        return False
    boundary = r"(?<![\d,])" if needle[0].isdigit() else ""
    tail = r"(?![\d,]*\d)" if needle[-1].isdigit() else ""
    return re.search(boundary + re.escape(needle) + tail, fact_form(answer)) is not None


_NUMBER_TOKEN = re.compile(r"\d+(?:,\d+)*%?")


def numbers_of(text: str) -> set[str]:
    """Numbers in a text after normalisation ("5,5 %" -> {"5,5%"}, "trente jours" -> {"30"})."""
    return set(_NUMBER_TOKEN.findall(fact_form(text)))


def is_numeric(fact: str) -> bool:
    return bool(numbers_of(fact))


def numbers_found(fact: str, answer: str) -> bool:
    """Every number of the fact appears in the answer, whatever the answer language
    ("30 jours ouvrables" is found in "30 business days" and in "30 يوم عمل")."""
    wanted = numbers_of(fact)
    return bool(wanted) and wanted <= numbers_of(answer)


def citation_coverage(answer: str) -> float:
    """Share of sentences (longer than a few words) that carry a [n] citation."""
    sentences = [s for s in _SENTENCE.findall(answer.strip()) if len(s.split()) >= 4]
    if not sentences:
        return 1.0 if _CITATION.search(answer) else 0.0
    return sum(bool(_CITATION.search(s)) for s in sentences) / len(sentences)


@dataclass(frozen=True)
class AnswerResult:
    id: str
    language: str
    difficulty: str
    answerable: bool
    abstention: str
    facts_found: int
    facts_total: int
    numeric_found: int  # facts containing a number: comparable in every answer language
    numeric_total: int
    cites_document: bool  # cites a chunk from an evidence document
    cites_article: bool  # cites a chunk labelled with the expected article
    citation_coverage: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    answer: str
    # LLM-as-judge (None when not judged: abstentions, judge off or judge error)
    faithfulness: float | None = None
    relevance: int | None = None
    correctness: float | None = None
    unsupported: tuple[str, ...] = ()
    missing_facts: tuple[str, ...] = ()


def score(item: GoldenItem, answer: Answer, latency_ms: float) -> AnswerResult:
    text = answer.text or ""
    facts = item.expected_facts + item.derived_facts
    cited = [answer.sources[c.number - 1].chunk for c in answer.citations]
    files = {e.file for e in item.all_evidence}
    return AnswerResult(
        id=item.id,
        language=item.language,
        difficulty=item.difficulty,
        answerable=item.answerable,
        abstention=answer.abstention,
        facts_found=sum(fact_found(f, text) for f in facts) if answer.abstention == "none" else 0,
        facts_total=len(facts),
        numeric_found=(
            sum(numbers_found(f, text) for f in facts if is_numeric(f))
            if answer.abstention == "none"
            else 0
        ),
        numeric_total=sum(is_numeric(f) for f in facts),
        cites_document=any(c.file in files for c in cited),
        cites_article=any(
            m.is_section_hit(c, e.file, item.expected_section)
            for c in cited
            for e in item.all_evidence
        ),
        citation_coverage=citation_coverage(text) if answer.abstention == "none" else 0.0,
        latency_ms=latency_ms,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        answer=text,
    )


def evaluate_answers(
    items: list[GoldenItem], pipeline: RagPipeline, judge: Judge | None = None
) -> list[AnswerResult]:
    results = []
    for item in items:
        if item.status == "rejected":
            continue
        start = time.perf_counter()
        answer = pipeline.answer(item.question)
        result = score(item, answer, (time.perf_counter() - start) * 1000)
        if judge is not None and item.answerable and answer.abstention == "none" and answer.text:
            try:
                verdict = judge.grade(item, answer.text, answer.sources)
            except Exception as error:  # a judge failure must not stop the evaluation
                log.warning("judge_failed", id=item.id, error=type(error).__name__)
            else:
                result = replace(
                    result,
                    faithfulness=verdict.faithfulness,
                    relevance=verdict.relevance,
                    correctness=verdict.correctness,
                    unsupported=tuple(verdict.unsupported),
                    missing_facts=tuple(verdict.missing_facts),
                )
        results.append(result)
    return results


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(
    rows: list[AnswerResult], price_in: float = 0.0, price_out: float = 0.0
) -> dict[str, float | None]:
    """Aggregate metrics; prices are USD per million tokens (0 = unknown, cost not reported)."""
    answerable = [r for r in rows if r.answerable]
    judged = [r for r in rows if r.faithfulness is not None]
    tokens_in = sum(r.prompt_tokens for r in rows)
    tokens_out = sum(r.completion_tokens for r in rows)
    priced = price_in > 0 or price_out > 0
    unanswerable = [r for r in rows if not r.answerable]
    answered = [r for r in answerable if r.abstention == "none"]
    facts = sum(r.facts_total for r in answerable)
    numeric = sum(r.numeric_total for r in answerable)
    latency = [r.latency_ms for r in rows if r.abstention != "low_score"]
    return {
        "n": len(rows),
        "fact_recall": sum(r.facts_found for r in answerable) / facts if facts else 0.0,
        "numeric_fact_recall": (
            sum(r.numeric_found for r in answerable) / numeric if numeric else 0.0
        ),
        "all_facts_rate": (
            sum(r.facts_found == r.facts_total for r in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        "cites_document": sum(r.cites_document for r in answered) / len(answered)
        if answered
        else 0.0,
        "cites_article": sum(r.cites_article for r in answered) / len(answered)
        if answered
        else 0.0,
        "citation_coverage": (
            sum(r.citation_coverage for r in answered) / len(answered) if answered else 0.0
        ),
        "false_abstention": (
            sum(r.abstention != "none" for r in answerable) / len(answerable) if answerable else 0.0
        ),
        "abstention_accuracy": (
            sum(r.abstention != "none" for r in unanswerable) / len(unanswerable)
            if unanswerable
            else 0.0
        ),
        "latency_p50_ms": m.percentile(latency, 50),
        "latency_p95_ms": m.percentile(latency, 95),
        "tokens_per_answer": (tokens_in + tokens_out) / len(rows) if rows else 0.0,
        "cost_per_query_usd": (
            (tokens_in * price_in + tokens_out * price_out) / 1e6 / len(rows)
            if priced and rows
            else None
        ),
        "judged": float(len(judged)),
        "faithfulness": _mean([r.faithfulness for r in judged if r.faithfulness is not None]),
        "relevance": _mean([float(r.relevance) for r in judged if r.relevance is not None]),
        "correctness": _mean([r.correctness for r in judged if r.correctness is not None]),
        "fully_faithful": _mean([float(r.faithfulness == 1.0) for r in judged]),
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _num(x: float | None, fmt: str = ".2f") -> str:
    return "n/a" if x is None else format(x, fmt)


def render_markdown(
    rows: list[AnswerResult],
    label: str,
    config: dict[str, object],
    price_in: float = 0.0,
    price_out: float = 0.0,
) -> str:
    lines = [
        f"# Answer evaluation: {label}",
        "",
        f"Generated {datetime.now(UTC).isoformat()} on {len(rows)} questions.",
        "",
        "Configuration: " + ", ".join(f"`{k}={v}`" for k, v in config.items()),
        "",
        "- **Numeric fact recall** (primary automatic metric): expected facts containing a "
        "number whose numbers all appear in the answer (number words count), comparable in "
        "every answer language. A wrongly refused question scores 0.",
        "- **Literal fact recall**: expected facts matched literally (French wording).",
        "- **Cites document / article**: answered questions citing a chunk from the evidence "
        "document / labelled with the expected article.",
        '- **False abstention / abstention accuracy**: answerable questions answered "not '
        'found" / unanswerable questions correctly answered "not found".',
        "- **Faithfulness / relevance / correctness** (LLM-as-judge, answered questions only): "
        "share of claims supported by the sources; 1-5 relevance to the question; share of "
        "expected facts conveyed in any language.",
        "",
        "| Slice | n | Numeric facts | Literal facts | Cites doc | Cites article | Citation "
        "coverage | False abst. | Abst. acc. | Faithfulness | Relevance | Correctness | p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def row(name: str, subset: list[AnswerResult]) -> str:
        s = summarize(subset)
        latency = s["latency_p50_ms"] or 0.0
        return (
            f"| {name} | {len(subset)} | {_pct(s['numeric_fact_recall'])} | "
            f"{_pct(s['fact_recall'])} | {_pct(s['cites_document'])} | "
            f"{_pct(s['cites_article'])} | {_pct(s['citation_coverage'])} | "
            f"{_pct(s['false_abstention'])} | {_pct(s['abstention_accuracy'])} | "
            f"{_pct(s['faithfulness'])} | {_num(s['relevance'])} | {_pct(s['correctness'])} | "
            f"{latency / 1000:.1f} s |"
        )

    lines.append(row("**All**", rows))
    for key in ("language", "difficulty"):
        for value in sorted({getattr(r, key) for r in rows}):
            lines.append(row(f"{key}={value}", [r for r in rows if getattr(r, key) == value]))
    s = summarize(rows, price_in, price_out)
    cost = s["cost_per_query_usd"]
    lines += [
        "",
        f"Average tokens per answer: {_num(s['tokens_per_answer'], '.0f')}. "
        f"Latency p95: {(s['latency_p95_ms'] or 0.0) / 1000:.1f} s. Cost per query: "
        + (f"${cost:.5f}" if cost is not None else "n/a (set REGLENS_LLM_PRICE_IN/OUT)")
        + ".",
        "",
        f"Judged answers: {_num(s['judged'], '.0f')}; fully faithful (no unsupported claim): "
        f"{_pct(s['fully_faithful'])}.",
        "",
        "Unsupported claims: "
        + (
            "; ".join(f"{r.id}: {' | '.join(r.unsupported)}" for r in rows if r.unsupported)
            or "none"
        ),
        "",
        "Wrong abstentions: "
        + (
            ", ".join(
                f"{r.id} ({'refused' if r.answerable else 'answered'})"
                for r in rows
                if (r.abstention != "none") == r.answerable
            )
            or "none"
        ),
    ]
    return "\n".join(lines) + "\n"


def write_answer_report(
    rows: list[AnswerResult],
    out_dir: Path,
    label: str,
    config: dict[str, object],
    price_in: float = 0.0,
    price_out: float = 0.0,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answers.json").write_text(
        json.dumps(
            {
                "label": label,
                "config": config,
                "summary": summarize(rows, price_in, price_out),
                "rows": [asdict(r) for r in rows],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    path = out_dir / "answers.md"
    path.write_text(render_markdown(rows, label, config, price_in, price_out), encoding="utf-8")
    return path
