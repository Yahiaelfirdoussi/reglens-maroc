"""Tests for OCR text metrics. Example strings are FICTIONAL."""

import pytest

from reglens.evaluation.text_metrics import (
    bag_of_words_f1,
    cer_counts,
    levenshtein,
    normalize,
    number_counts,
    number_tokens,
    strip_page_number,
    wer_counts,
)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("", "", 0), ("abc", "abc", 0), ("abc", "", 3), ("kitten", "sitting", 3), ("ab", "ba", 2)],
)
def test_levenshtein(a: str, b: str, expected: int) -> None:
    assert levenshtein(a, b) == expected
    assert levenshtein(b, a) == expected


def test_normalize_ignores_layout_and_typography() -> None:
    assert normalize("l’article\n\n  premier —  FICTIONAL") == "l'article premier - FICTIONAL"


def test_cer_and_wer_counts() -> None:
    assert cer_counts("Article 12", "Article12") == (1, 10)
    assert wer_counts("Article 12 FICTIONAL", "Article12 FICTIONAL") == (2, 3)
    assert cer_counts("même\ntexte", "même texte") == (0, 10)


def test_bag_of_words_f1_ignores_order() -> None:
    assert bag_of_words_f1("colonne gauche colonne droite", "colonne droite colonne gauche") == 1.0
    assert bag_of_words_f1("a b", "c d") == 0.0


def test_number_tokens_and_recall() -> None:
    assert number_tokens("ratio de 100 % et 5,5% sur 30 jours") == {"100%": 1, "5,5%": 1, "30": 1}
    assert number_counts("100 % puis 30 jours", "100% puis 3O jours") == (1, 2)


def test_number_counts_ignore_the_page_number() -> None:
    reference = "7\nDélai de 30 jours FICTIONAL\n\n 7 \n"
    assert number_counts(reference, "Délai de 30 jours FICTIONAL", page_number=7) == (1, 1)
    assert number_counts(reference, "Délai de 30 jours FICTIONAL") == (1, 3)


def test_strip_page_number_keeps_table_cells() -> None:
    assert strip_page_number("Article 5 FICTIONAL\n12", 12) == "Article 5 FICTIONAL"
    assert strip_page_number("Pondération FICTIONAL\n20", 3) == "Pondération FICTIONAL\n20"
