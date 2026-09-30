"""Command-line interface: ``reglens ingest | ask | eval | serve``."""

from pathlib import Path
from typing import Annotated

import typer

from reglens.config import get_settings
from reglens.log import configure_logging

app = typer.Typer(help="RegLens Maroc — grounded answers on Moroccan financial regulation.")


@app.callback()
def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)


def _not_implemented(command: str, phase: int) -> None:
    typer.echo(f"`reglens {command}` is not implemented yet (roadmap phase {phase}).", err=True)
    raise typer.Exit(code=1)


@app.command()
def fetch(
    pages: Annotated[Path, typer.Option(help="YAML list of listing pages.")] = Path(
        "data/source_pages.yaml"
    ),
    out: Annotated[Path, typer.Option(help="Download directory.")] = Path("data/raw"),
    manifest: Annotated[Path, typer.Option(help="Corpus manifest to rebuild.")] = Path(
        "data/sources.yaml"
    ),
    list_only: Annotated[
        bool, typer.Option("--list", help="Only list discovered documents.")
    ] = False,
    issuer: Annotated[str | None, typer.Option(help="Keep one issuer (BAM, AMMC).")] = None,
    match: Annotated[str | None, typer.Option(help="Case-insensitive regex on titles.")] = None,
    limit: Annotated[int | None, typer.Option(help="Maximum documents to download.")] = None,
    selection: Annotated[
        Path | None, typer.Option(help="YAML allow-list of document URLs (curated corpus).")
    ] = None,
    delay: Annotated[float, typer.Option(help="Seconds between requests.")] = 1.0,
) -> None:
    """Discover and download official PDFs with YAML sidecar metadata."""
    from reglens.ingestion import fetch as fetching

    entries = {e.url: e for e in fetching.load_selection(selection)} if selection else {}
    urls = set(entries) if selection else None
    with fetching.make_client() as client:
        fetcher = fetching.Fetcher(client, delay=delay)
        found = fetching.discover(fetching.load_source_pages(pages), fetcher)
        selected = fetching.select(found, issuer=issuer, match=match, limit=limit, urls=urls)
        typer.echo(f"{len(found)} documents discovered, {len(selected)} selected.")
        if urls is not None:
            missing = set(urls) - {c.url for c in found}
            for url in sorted(missing):
                typer.echo(f"WARNING: selected URL no longer listed on the site: {url}", err=True)

        if list_only:
            for c in selected:
                published = c.published.isoformat() if c.published else "?"
                typer.echo(f"{c.issuer:<5} {c.reference or '?':<12} {published:<10}  {c.title}")
            return

        downloaded = [fetching.download(c, out, fetcher) for c in selected]
        for candidate, path in zip(selected, downloaded, strict=True):
            if path is not None and candidate.url in entries:
                fetching.apply_overrides(path, entries[candidate.url])
        count = fetching.write_manifest(out, manifest)
        failed = sum(path is None for path in downloaded)
        typer.echo(f"{len(selected) - failed} documents on disk, {failed} rejected.")
        typer.echo(f"Manifest {manifest} lists {count} documents.")


@app.command()
def ocr(
    path: Annotated[Path, typer.Argument(help="Directory of PDFs.")] = Path("data/raw"),
    language: Annotated[
        str | None,
        typer.Option(help="Tesseract language(s), e.g. fra+ara. Default: each sidecar's language."),
    ] = None,
    dpi: Annotated[int, typer.Option(help="Rendering resolution for OCR.")] = 300,
    workers: Annotated[int, typer.Option(help="Parallel OCR processes.")] = 4,
    force: Annotated[bool, typer.Option(help="Redo OCR even if cached.")] = False,
) -> None:
    """OCR scanned PDFs (no text layer) and record the text source in each sidecar."""
    from collections import Counter

    from reglens.ingestion.ocr import ocr_directory

    outcomes = ocr_directory(path, language=language, dpi=dpi, force=force, workers=workers)
    counts = Counter(o.status for o in outcomes)
    typer.echo(
        f"{len(outcomes)} PDFs: {counts['ocr']} OCR'd, {counts['cached']} already OCR'd, "
        f"{counts['native']} with a native text layer."
    )


@app.command("corpus-report")
def corpus_report(
    path: Annotated[Path, typer.Argument(help="Directory of PDFs.")] = Path("data/raw"),
    out: Annotated[Path, typer.Option(help="Markdown report.")] = Path("docs/corpus.md"),
    benchmark: Annotated[
        bool, typer.Option(help="Run the OCR accuracy benchmark on native-text documents.")
    ] = True,
    workers: Annotated[int, typer.Option(help="Parallel OCR processes.")] = 4,
) -> None:
    """Corpus statistics, metadata completeness and OCR accuracy (docs/corpus.md)."""
    import json

    from reglens.evaluation import corpus

    stats = corpus.document_stats(path)
    results = corpus.run_ocr_benchmark(path, workers=workers) if benchmark else None
    out.write_text(corpus.render_markdown(stats, results), encoding="utf-8")
    if results is not None:
        raw = Path("eval/report/ocr_benchmark.json")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps(corpus.benchmark_as_dicts(results), indent=2), encoding="utf-8")
    typer.echo(f"Report written to {out}.")


@app.command("golden-check")
def golden_check(
    path: Annotated[Path, typer.Argument(help="Golden set JSONL.")] = Path("eval/golden_set.jsonl"),
    raw: Annotated[Path, typer.Option(help="Directory of PDFs.")] = Path("data/raw"),
) -> None:
    """Check that every expected fact is backed by an exact quote on the cited page."""
    from collections import Counter

    from reglens.evaluation.golden import load_golden, validate

    items = load_golden(path)
    errors = [error for item in items for error in validate(item, raw)]
    for error in errors:
        typer.echo(f"ERROR {error}", err=True)
    answerable = sum(item.answerable for item in items)
    languages = Counter(item.language for item in items)
    statuses = Counter(item.status for item in items)
    typer.echo(
        f"{len(items)} questions ({answerable} answerable, {len(items) - answerable} "
        f"unanswerable) | languages {dict(languages)} | status {dict(statuses)}"
    )
    typer.echo(f"{len(errors)} evidence errors.")
    if errors:
        raise typer.Exit(code=1)


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Directory of PDFs + YAML sidecars.")],
) -> None:
    """Ingest documents into the vector store."""
    _not_implemented("ingest", 1)


@app.command()
def ask(question: Annotated[str, typer.Argument(help="Question in FR, AR or EN.")]) -> None:
    """Answer a question with citations."""
    _not_implemented("ask", 1)


@app.command("eval")
def evaluate(
    golden_set: Annotated[Path, typer.Argument(help="Path to a golden set JSONL file.")],
    judge: Annotated[bool, typer.Option(help="Use LLM-as-judge metrics.")] = False,
) -> None:
    """Run the evaluation on a golden set."""
    _not_implemented("eval", 1)


@app.command()
def serve() -> None:
    """Start the HTTP API."""
    _not_implemented("serve", 5)
