"""Tests for golden-set validation. All documents and questions here are FICTIONAL."""

from pathlib import Path

import pymupdf
import pytest
from pydantic import ValidationError

from reglens.evaluation.golden import (
    Evidence,
    GoldenItem,
    compact,
    find_page,
    load_golden,
    validate,
    write_golden,
)

PAGE_TEXT = "Article 3 FICTIONAL\nLe ratio doit être au moins égal à 100 % sur 30 jours."


def _pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 72), "Page de garde FICTIONAL")
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 300), PAGE_TEXT)
    path = tmp_path / "bam" / "fictional.pdf"
    path.parent.mkdir()
    doc.save(path)
    return path


def _item(**overrides: object) -> GoldenItem:
    data: dict[str, object] = {
        "id": "q1",
        "question": "Quel ratio minimum FICTIONAL ?",
        "language": "fr",
        "answerable": True,
        "theme": "fictional",
        "expected_facts": ["100%", "30 jours"],
        "evidence": Evidence(
            file="bam/fictional.pdf", page=2, quote="au moins égal à 100 % sur 30"
        ),
    }
    data.update(overrides)
    return GoldenItem.model_validate(data)


def test_compact_ignores_spacing_and_typography() -> None:
    assert compact("100 %") == compact("100%")
    assert compact("l’article  3") == compact("l'article 3")


def test_valid_item_has_no_errors(tmp_path: Path) -> None:
    _pdf(tmp_path)
    item = _item(expected_facts=["100%"])
    assert validate(item, tmp_path) == []


def test_validate_reports_wrong_page_and_missing_fact(tmp_path: Path) -> None:
    _pdf(tmp_path)
    errors = validate(_item(), tmp_path)  # "30 jours" is not inside the quote
    assert errors == ["q1: fact '30 jours' not in quote"]
    wrong_page = _item(evidence=Evidence(file="bam/fictional.pdf", page=1, quote="100 %"))
    assert "q1: quote not found on page 1" in validate(wrong_page, tmp_path)


def test_find_page(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path)
    assert find_page(pdf, "égal à 100 %") == 2
    assert find_page(pdf, "absent FICTIONAL") is None


def test_unanswerable_items_carry_no_evidence() -> None:
    GoldenItem(id="u1", question="Q ?", language="en", answerable=False, theme="none")
    with pytest.raises(ValidationError):
        _item(answerable=False)
    with pytest.raises(ValidationError):
        _item(evidence=None)


def test_facts_may_come_from_extra_evidence(tmp_path: Path) -> None:
    _pdf(tmp_path)
    item = _item(
        expected_facts=["100%", "FICTIONAL"],
        derived_facts=["200 %"],
        notes="FICTIONAL: 200 % = twice the ratio",
        extra_evidence=[Evidence(file="bam/fictional.pdf", page=2, quote="Article 3 FICTIONAL")],
    )
    assert validate(item, tmp_path) == []
    assert len(item.all_evidence) == 2
    bad = item.model_copy(
        update={"extra_evidence": [Evidence(file="bam/fictional.pdf", page=1, quote="absent")]}
    )
    assert "q1: quote not found on page 1" in validate(bad, tmp_path)


def test_derived_facts_need_a_note() -> None:
    with pytest.raises(ValidationError, match="derived facts need a note"):
        _item(derived_facts=["200 %"])


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    write_golden(path, [_item()])
    assert load_golden(path) == [_item()]
