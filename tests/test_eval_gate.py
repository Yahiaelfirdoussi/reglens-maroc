"""CI evaluation gate: index the FICTIONAL fixture corpus through the real CLI, then fail the
build if retrieval quality drops below the gate. Offline: hash embedder, local Qdrant."""

import json
from pathlib import Path

import pymupdf
import pytest
from typer.testing import CliRunner

from reglens.cli import app
from reglens.config import get_settings
from reglens.evaluation.golden import load_golden, validate
from reglens.ingestion.metadata import write_sidecar
from reglens.models import DocumentMetadata

FIXTURES = Path(__file__).parent / "fixtures"
GATE = 0.85  # minimum hit@6 on the fixture set


@pytest.fixture()
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = tmp_path / "raw"
    data = json.loads((FIXTURES / "fictional_corpus.json").read_text(encoding="utf-8"))
    for document in data["documents"]:
        pdf = raw / document["file"]
        pdf.parent.mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open()
        for text in document["pages"]:
            doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
        doc.save(pdf)
        write_sidecar(
            pdf,
            DocumentMetadata(
                title=document["title"],
                issuer=document["issuer"],
                reference=document["reference"],
                url=f"https://example.test/{document['file']}",
                language="fr",
            ),
        )
    for key, value in {
        "REGLENS_EMBEDDING_MODEL": "hash",
        "REGLENS_QDRANT_PATH": str(tmp_path / "qdrant"),
        "REGLENS_CHUNKING": "legal",
        "REGLENS_CHUNK_SIZE": "1200",
        "REGLENS_CHUNK_OVERLAP": "200",
        "REGLENS_RETRIEVAL_MODE": "dense",
        "REGLENS_RERANKER": "",
        "REGLENS_LLM_MODEL": "",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield raw
    get_settings.cache_clear()


def test_fixture_golden_set_is_consistent(corpus: Path) -> None:
    items = load_golden(FIXTURES / "golden_fixtures.jsonl")
    assert [e for item in items for e in validate(item, corpus)] == []


def test_retrieval_quality_gate(corpus: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    ingest = runner.invoke(app, ["ingest", str(corpus)])
    assert ingest.exit_code == 0, ingest.output
    golden = str(FIXTURES / "golden_fixtures.jsonl")
    out = str(tmp_path / "report")
    passed = runner.invoke(app, ["eval", golden, "--out", out, "--min-hit-rate", str(GATE)])
    assert passed.exit_code == 0, passed.output
    assert (tmp_path / "report" / "report.json").exists()
    # The gate really fails when quality is below the threshold.
    failed = runner.invoke(app, ["eval", golden, "--out", out, "--min-hit-rate", "1.01"])
    assert failed.exit_code == 1
