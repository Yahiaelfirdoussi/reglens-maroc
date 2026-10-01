"""Tests for article-based chunking and the section metric. All text here is FICTIONAL."""

from reglens.evaluation.metrics import article_ids, is_section_hit
from reglens.ingestion.chunking import find_articles, legal_chunks
from reglens.models import Chunk

PREAMBLE = "Circulaire FICTIONAL n°99/Z/2099 relative aux essais. Vu la loi FICTIONAL ; "


def _article(number: str, words: int) -> str:
    body = " ".join(f"règle{i}" for i in range(words))
    return f"Article {number} {body}, conformément à l'article 1 ci-dessus. "


def test_find_articles_ignores_references_and_repairs_misreads() -> None:
    text = (
        "ARTICLE PREMIER. — Texte FICTIONAL. ART. 2.- Suite. "
        "Article 11 Voir l'article 3 ci-dessus. Article 42 Texte. Article 13 Fin. "
        "Article 1.20 Point. المادة 3 نص"
    )
    labels = [label for _, label in find_articles(text)]
    assert labels == [
        "Article premier",
        "Article 2",
        "Article 11",
        "Article 12",  # "42" repaired from its neighbours
        "Article 13",
        "Article 1.20",
        "Article 3",
    ]


def test_one_chunk_per_article_with_preamble() -> None:
    pages = [PREAMBLE + _article("premier", 60), _article("2", 60)]
    spans = legal_chunks(pages, size=600, overlap=100, min_size=50)
    assert [s.section for s in spans] == [None, "Article premier", "Article 2"]
    assert spans[1].text.startswith("Article premier")
    assert (spans[1].page_start, spans[2].page_start) == (1, 2)
    assert not any("Article 2" in s.text for s in spans[:2])  # never two articles mixed


def test_short_articles_are_merged_and_long_ones_split_with_overlap() -> None:
    short = "".join(_article(str(n), 3) for n in (1, 2, 3))
    long_body = " ".join(f"Phrase numéro {i} du texte FICTIONAL." for i in range(60))
    pages = [short + f"Article 4 {long_body}"]
    spans = legal_chunks(pages, size=400, overlap=80, min_size=120)
    labels = [s.section for s in spans]
    # Short articles are grouped only until the group reaches min_size.
    assert labels[0] == "Articles 1-2"
    assert len(spans[0].text) >= 120
    assert labels.count("Article 4") > 1
    parts = [s.text for s in spans if s.section == "Article 4"]
    assert all(len(p) <= 400 for p in parts)
    assert parts[0].split(".")[-2].strip() in parts[1]  # consecutive parts overlap
    assert all(p.rstrip().endswith(".") for p in parts[:-1])  # cut on sentence boundaries


def test_table_stays_with_its_introduction() -> None:
    table = "Notation AAA AA A BBB BB B Pondération 0 % 20 % 50 % 100 % 150 % 350 % " * 6
    text = f"Article 7 Les pondérations FICTIONAL sont les suivantes : {table.strip()} Fin."
    spans = legal_chunks([text], size=300, overlap=50)
    holder = next(s for s in spans if "sont les suivantes" in s.text)
    assert table.strip() in holder.text
    assert len(holder.text) > 300  # allowed to exceed the size cap, up to twice it


def test_documents_without_articles_fall_back_to_sentence_windows() -> None:
    body = " ".join(f"Principe FICTIONAL numéro {i}." for i in range(80))
    spans = legal_chunks([body], size=300, overlap=50)
    assert len(spans) > 1
    assert all(s.section is None and len(s.text) <= 300 for s in spans)


def test_article_ids_and_section_hits() -> None:
    assert article_ids("Articles 3-5") == {"3", "4", "5"}
    assert article_ids("Article premier") == {"1"}
    assert article_ids("Article 1.20") == {"1.20"}
    assert article_ids("Articles 4 and 5") == {"4", "5"}
    assert article_ids(None) == set()
    chunk = Chunk(
        id="x",
        text="t",
        file="bam/f.pdf",
        index=0,
        page_start=1,
        page_end=1,
        issuer="BAM",
        title="FICTIONAL",
        url="https://example.test",
        section="Articles 3-4",
    )
    assert is_section_hit(chunk, "bam/f.pdf", "Article 4")
    assert not is_section_hit(chunk, "bam/f.pdf", "Article 5")
    assert not is_section_hit(chunk, "bam/other.pdf", "Article 4")
