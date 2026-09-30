"""Evaluation runner: retrieval quality of the pipeline on the golden set."""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from reglens.evaluation import metrics as m
from reglens.evaluation.golden import GoldenItem
from reglens.retrieval.retriever import Retriever


@dataclass(frozen=True)
class ItemResult:
    id: str
    language: str
    theme: str
    difficulty: str
    passage_rank: int | None
    document_rank: int | None
    retrieval_ms: float
    top_files: list[str]


@dataclass
class Report:
    label: str
    config: dict[str, object]
    k: int
    created_at: str
    items: list[ItemResult] = field(default_factory=list)

    def summary(self, subset: list[ItemResult] | None = None) -> dict[str, float]:
        rows = self.items if subset is None else subset
        passage = [r.passage_rank for r in rows]
        document = [r.document_rank for r in rows]
        latency = [r.retrieval_ms for r in rows]
        return {
            "n": len(rows),
            "hit@1": m.hit_rate(passage, 1),
            "hit@3": m.hit_rate(passage, 3),
            f"hit@{self.k}": m.hit_rate(passage, self.k),
            f"mrr@{self.k}": m.mrr(passage, self.k),
            f"doc_hit@{self.k}": m.hit_rate(document, self.k),
            "latency_p50_ms": m.percentile(latency, 50),
            "latency_p95_ms": m.percentile(latency, 95),
        }

    def by(self, key: str) -> dict[str, dict[str, float]]:
        groups: dict[str, list[ItemResult]] = {}
        for row in self.items:
            groups.setdefault(str(getattr(row, key)), []).append(row)
        return {name: self.summary(rows) for name, rows in sorted(groups.items())}


def evaluate_retrieval(
    items: list[GoldenItem], retriever: Retriever, k: int, label: str, config: dict[str, object]
) -> Report:
    """Retrieve ``k`` chunks for every answerable question and rank the first relevant one."""
    report = Report(label=label, config=config, k=k, created_at=datetime.now(UTC).isoformat())
    for item in items:
        if not item.answerable or item.evidence is None or item.status == "rejected":
            continue
        start = time.perf_counter()
        results = retriever.retrieve(item.question, k)
        elapsed = (time.perf_counter() - start) * 1000
        passages = item.all_evidence
        passage = [
            any(m.is_passage_hit(r.chunk, e.file, e.quote) for e in passages) for r in results
        ]
        document = [any(r.chunk.file == e.file for e in passages) for r in results]
        report.items.append(
            ItemResult(
                id=item.id,
                language=item.language,
                theme=item.theme,
                difficulty=item.difficulty,
                passage_rank=m.first_hit_rank(passage),
                document_rank=m.first_hit_rank(document),
                retrieval_ms=elapsed,
                top_files=[r.chunk.file for r in results],
            )
        )
    return report


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_markdown(report: Report) -> str:
    k = report.k
    s = report.summary()
    lines = [
        f"# Evaluation report: {report.label}",
        "",
        f"Generated {report.created_at} on {s['n']:.0f} answerable questions.",
        "",
        "Configuration: " + ", ".join(f"`{key}={value}`" for key, value in report.config.items()),
        "",
        "A **passage hit** is a retrieved chunk from the expected document that contains at "
        f"least {m.PASSAGE_COVERAGE:.0%} of the evidence quote's words; a **document hit** only "
        "needs the right document.",
        "",
        f"| Slice | n | Hit@1 | Hit@3 | Hit@{k} | MRR@{k} | Doc hit@{k} |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def row(name: str, v: dict[str, float]) -> str:
        return (
            f"| {name} | {v['n']:.0f} | {_pct(v['hit@1'])} | {_pct(v['hit@3'])} | "
            f"{_pct(v[f'hit@{k}'])} | {v[f'mrr@{k}']:.3f} | {_pct(v[f'doc_hit@{k}'])} |"
        )

    lines.append(row("**All**", s))
    lines += [row(f"difficulty={name}", v) for name, v in report.by("difficulty").items()]
    lines += [row(f"lang={name}", v) for name, v in report.by("language").items()]
    lines += [row(f"theme={name}", v) for name, v in report.by("theme").items()]
    lines += [
        "",
        f"Retrieval latency: p50 {s['latency_p50_ms']:.0f} ms, p95 {s['latency_p95_ms']:.0f} ms.",
        "",
        f"Misses (no passage hit in the top {k}): "
        + (", ".join(r.id for r in report.items if r.passage_rank is None) or "none"),
    ]
    return "\n".join(lines) + "\n"


def write_report(report: Report, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": report.label,
        "config": report.config,
        "k": report.k,
        "created_at": report.created_at,
        "summary": report.summary(),
        "by_difficulty": report.by("difficulty"),
        "by_language": report.by("language"),
        "by_theme": report.by("theme"),
        "items": [asdict(item) for item in report.items],
    }
    json_path, md_path = out_dir / "report.json", out_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
