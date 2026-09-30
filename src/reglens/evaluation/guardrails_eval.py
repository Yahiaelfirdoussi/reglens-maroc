"""Guardrail evaluation: labelled off-topic / meta / greeting / in-scope questions.

Every golden-set question (unanswerable ones included) is also expected to pass: they are
in scope, and a false refusal there is the costliest error.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from reglens.evaluation.golden import load_golden
from reglens.guardrails import ScopeClassifier, Verdict, check


@dataclass(frozen=True)
class GuardCase:
    id: str
    question: str
    language: str
    expected: str  # "ok" | "meta" | "off_topic" | "greeting"


def load_cases(guard_set: Path, golden_set: Path | None) -> list[GuardCase]:
    cases = [
        GuardCase(**json.loads(line))
        for line in guard_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if golden_set is not None:
        cases += [
            GuardCase(item.id, item.question, item.language, "ok")
            for item in load_golden(golden_set)
        ]
    return cases


@dataclass
class GuardReport:
    rows: list[tuple[GuardCase, Verdict, float, float]]  # case, verdict, in/out scores

    def confusion(self) -> dict[str, Counter[str]]:
        table: dict[str, Counter[str]] = defaultdict(Counter)
        for case, verdict, _, _ in self.rows:
            table[case.expected][verdict] += 1
        return table

    def accuracy(self, expected: str) -> float:
        row = self.confusion()[expected]
        total = sum(row.values())
        return row[expected] / total if total else 0.0

    def errors(self) -> list[tuple[GuardCase, Verdict]]:
        return [(case, verdict) for case, verdict, _, _ in self.rows if verdict != case.expected]


def evaluate_guardrails(cases: list[GuardCase], scope: ScopeClassifier) -> GuardReport:
    rows: list[tuple[GuardCase, Verdict, float, float]] = []
    for case in cases:
        verdict = check(case.question, scope).verdict
        inside, outside = scope.scores(case.question)
        rows.append((case, verdict, inside, outside))
    return GuardReport(rows)


def render_markdown(report: GuardReport, model: str) -> str:
    classes = ["ok", "meta", "off_topic", "greeting"]
    table = report.confusion()
    lines = [
        "# Guardrail evaluation",
        "",
        f"Scope embeddings: `{model}`. Rows: expected class; columns: guardrail verdict.",
        "",
        "| Expected \\ Verdict | n | " + " | ".join(classes) + " | Accuracy |",
        "|---|---:|" + "---:|" * len(classes) + "---:|",
    ]
    for expected in classes:
        row = table.get(expected, Counter())
        n = sum(row.values())
        if not n:
            continue
        cells = " | ".join(str(row[c]) for c in classes)
        lines.append(f"| {expected} | {n} | {cells} | {report.accuracy(expected):.1%} |")
    ok_total = sum(table["ok"].values())
    false_refusals = ok_total - table["ok"]["ok"]
    lines += [
        "",
        f"False refusals on in-scope questions: {false_refusals}/{ok_total}.",
        "",
        "Errors: "
        + (
            "; ".join(f"{c.id} ({c.expected}→{v}): {c.question}" for c, v in report.errors())
            or "none"
        ),
    ]
    return "\n".join(lines) + "\n"
