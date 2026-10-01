"""LLM-as-judge: faithfulness, relevance and semantic correctness of an answer.

A separate, stronger model grades each answer (a model grading its own answers is lenient):

- **faithfulness** (0-1): share of the answer's claims supported by the sources it was given;
- **relevance** (1-5): does the answer address the question asked;
- **correctness** (0-1): share of the expected facts the answer conveys, in any language
  (translations and paraphrases count, unlike literal string matching).

The judge sees the question, the answer, the numbered sources and the reference (expected
facts and the official passage). It replies in JSON.
"""

import json
import re
from dataclasses import dataclass

from reglens.evaluation.golden import GoldenItem
from reglens.generation.llm import LLM, Message
from reglens.models import ScoredChunk

JUDGE_PROMPT = """You grade answers produced by a retrieval-augmented assistant on Moroccan \
financial regulation. Be strict and literal. Reply with a single JSON object and nothing else:
{"claims": <number of factual claims in the answer>,
 "supported_claims": <how many of them are fully supported by the SOURCES>,
 "unsupported": [<short quotes of unsupported claims>],
 "relevance": <1-5: 5 = directly and completely answers the QUESTION, 1 = off-topic>,
 "facts_conveyed": <how many EXPECTED FACTS the answer conveys, in any language or wording>,
 "missing_facts": [<expected facts not conveyed>]}
Rules: a claim is supported only if the SOURCES state it; general knowledge does not count.
An expected fact is conveyed if its meaning is present (translation or paraphrase is fine) and
any figure or deadline in it is exact."""


@dataclass(frozen=True)
class Verdict:
    faithfulness: float
    relevance: int
    correctness: float
    unsupported: list[str]
    missing_facts: list[str]


def build_judge_messages(
    item: GoldenItem, answer: str, sources: list[ScoredChunk]
) -> list[Message]:
    facts = item.expected_facts + item.derived_facts
    numbered = "\n\n".join(
        f"[{n}] ({s.chunk.reference or ''} {s.chunk.section or ''}) {s.chunk.text}"
        for n, s in enumerate(sources, start=1)
    )
    reference = "\n".join(e.quote for e in item.all_evidence)
    content = (
        f"QUESTION:\n{item.question}\n\nANSWER:\n{answer}\n\nSOURCES:\n{numbered}\n\n"
        f"EXPECTED FACTS ({len(facts)}):\n"
        + "\n".join(f"- {f}" for f in facts)
        + f"\n\nOFFICIAL PASSAGE:\n{reference}"
    )
    if item.notes:
        content += f"\n\nREVIEWER NOTE:\n{item.notes}"
    return [{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": content}]


def parse_verdict(raw: str, facts_total: int) -> Verdict:
    """Parse the judge's JSON (tolerating code fences); clamp values to their ranges."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(match.group(0) if match else raw)
    claims = max(int(data.get("claims", 0)), 0)
    supported = min(max(int(data.get("supported_claims", 0)), 0), claims)
    conveyed = min(max(int(data.get("facts_conveyed", 0)), 0), facts_total)
    return Verdict(
        faithfulness=supported / claims if claims else 1.0,
        relevance=min(max(int(data.get("relevance", 1)), 1), 5),
        correctness=conveyed / facts_total if facts_total else 1.0,
        unsupported=[str(x) for x in data.get("unsupported", [])][:5],
        missing_facts=[str(x) for x in data.get("missing_facts", [])][:5],
    )


class Judge:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return self._llm.name

    def grade(self, item: GoldenItem, answer: str, sources: list[ScoredChunk]) -> Verdict:
        facts_total = len(item.expected_facts + item.derived_facts)
        raw = self._llm.complete(build_judge_messages(item, answer, sources))
        return parse_verdict(raw, facts_total)
