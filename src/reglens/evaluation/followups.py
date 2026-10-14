"""Follow-up questions: does the conversation context find the right passage?

Each case pairs two approved golden questions: the ``context`` question is asked first (and
really answered), then the ``followup`` message, which only makes sense after it. The
expected passage is the ``target`` golden question's evidence, so no new regulatory content
is needed. ``standalone`` cases ask a complete golden question after an unrelated one: the
rewrite must leave them working. ``unanswerable`` cases follow up with a question the texts
do not answer (an unanswerable golden item): the assistant must still say "not found".
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel

from reglens.evaluation import metrics as m
from reglens.evaluation.golden import GoldenItem
from reglens.models import ScoredChunk
from reglens.rag import Answer, RagPipeline

log = structlog.get_logger(__name__)

SAME_AS_TARGET = "@target"  # the follow-up is the target golden question, verbatim


class FollowUpCase(BaseModel):
    id: str
    context: str  # golden id asked first
    followup: str  # the next message, or "@target"
    target: str  # golden id whose evidence answers the follow-up
    kind: Literal["follow_up", "standalone", "unanswerable"]


@dataclass
class FollowUpRow:
    id: str
    kind: str
    question: str
    alone_rank: int | None  # the follow-up searched on its own
    context_rank: int | None  # rewritten with the conversation
    standalone: str | None  # the rewritten question, when it changed
    status: str  # answered / not_found / blocked / error
    rewrite_ms: float
    alone_status: str = ""  # unanswerable cases: the status without the conversation


def load_cases(path: Path) -> list[FollowUpCase]:
    lines = path.read_text("utf-8").splitlines()
    return [FollowUpCase.model_validate(json.loads(line)) for line in lines if line.strip()]


def _rank(sources: list[ScoredChunk], target: GoldenItem) -> int | None:
    evidence = target.all_evidence
    hits = [any(m.is_passage_hit(s.chunk, e.file, e.quote) for e in evidence) for s in sources]
    return m.first_hit_rank(hits)


def _status(answer: Answer) -> str:
    if answer.guardrail != "ok":
        return "blocked"
    return "not_found" if answer.abstention != "none" else "answered"


def evaluate_followups(
    cases: list[FollowUpCase], golden: dict[str, GoldenItem], pipeline: RagPipeline
) -> list[FollowUpRow]:
    rows = []
    for case in cases:
        context, target = golden[case.context], golden[case.target]
        question = target.question if case.followup == SAME_AS_TARGET else case.followup
        try:
            first = pipeline.answer(context.question)
            alone = pipeline.retriever.retrieve(question, pipeline.top_k)
            answer = pipeline.answer(question, history=[(context.question, first.text or "")])
            alone_status = _status(pipeline.answer(question)) if case.kind == "unanswerable" else ""
        except Exception as error:  # record and continue (network)
            log.warning("followup_failed", id=case.id, error=type(error).__name__)
            rows.append(FollowUpRow(case.id, case.kind, question, None, None, None, "error", 0))
            continue
        rows.append(
            FollowUpRow(
                id=case.id,
                kind=case.kind,
                question=question,
                alone_rank=_rank(alone, target),
                context_rank=_rank(answer.sources, target),
                standalone=answer.standalone_question,
                status=_status(answer),
                rewrite_ms=answer.rewrite_ms,
                alone_status=alone_status,
            )
        )
    return rows


def _rr(rank: int | None) -> float:
    return 1 / rank if rank else 0.0


def summary(rows: list[FollowUpRow], kind: str) -> dict[str, float]:
    group = [r for r in rows if r.kind == kind]
    n = len(group) or 1
    refused = ("not_found", "blocked")
    return {
        "n": len(group),
        "hit_alone": sum(r.alone_rank is not None for r in group) / n,
        "hit_context": sum(r.context_rank is not None for r in group) / n,
        "mrr_alone": sum(_rr(r.alone_rank) for r in group) / n,
        "mrr_context": sum(_rr(r.context_rank) for r in group) / n,
        "answered": sum(r.status == "answered" for r in group) / n,
        "abstained_alone": sum(r.alone_status in refused for r in group) / n,
        "abstained_context": sum(r.status in refused for r in group) / n,
        "rewrite_ms_p50": m.percentile([r.rewrite_ms for r in group], 50),
    }


def render_markdown(rows: list[FollowUpRow], k: int) -> str:
    lines = [
        "# Follow-up questions",
        "",
        f"Each message is sent after a first, really answered question. Hit@{k} and MRR: the",
        "expected passage among the sources, searched on the message alone vs. rewritten with",
        "the conversation.",
        "",
        f"| Cases | n | Hit@{k} alone | Hit@{k} with context | MRR alone | MRR with context "
        "| Answered | Rewrite p50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, label in (("follow_up", "Follow-ups"), ("standalone", "Complete questions")):
        s = summary(rows, kind)
        lines.append(
            f"| {label} | {s['n']:.0f} | {s['hit_alone']:.1%} | {s['hit_context']:.1%} | "
            f"{s['mrr_alone']:.3f} | {s['mrr_context']:.3f} | {s['answered']:.1%} | "
            f"{s['rewrite_ms_p50']:.0f} ms |"
        )
    u = summary(rows, "unanswerable")
    if u["n"]:
        lines += [
            "",
            "| Unanswerable follow-ups | n | Abstained alone | Abstained with context |",
            "|---|---:|---:|---:|",
            f"| Not in the texts | {u['n']:.0f} | {u['abstained_alone']:.1%} | "
            f"{u['abstained_context']:.1%} |",
        ]
    lines += [
        "",
        "| Id | Message | Understood as | Alone | Context | Status |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        understood = r.standalone or "(unchanged)"
        if r.kind == "unanswerable":
            alone, context = r.alone_status, r.status
        else:
            alone, context = str(r.alone_rank or "miss"), str(r.context_rank or "miss")
        lines.append(f"| {r.id} | {r.question} | {understood} | {alone} | {context} | {r.status} |")
    return "\n".join(lines) + "\n"
