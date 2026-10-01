"""Command-line interface: ``reglens ingest | ask | eval | serve``."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from reglens.config import get_settings
from reglens.log import configure_logging

if TYPE_CHECKING:
    from reglens.retrieval.retriever import Retriever

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


def _retriever() -> "Retriever":
    from reglens.retrieval.embeddings import make_embedder
    from reglens.retrieval.reranker import make_reranker
    from reglens.retrieval.retriever import Retriever
    from reglens.retrieval.vector_store import QdrantStore, make_client

    settings = get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    client = make_client(settings.qdrant_url, settings.qdrant_path, api_key)
    store = QdrantStore(client, settings.collection_name)
    if store.count() == 0:
        typer.echo("The index is empty. Run `reglens ingest data/raw` first.", err=True)
        raise typer.Exit(code=1)
    return Retriever(
        make_embedder(
            settings.embedding_model, settings.model_cache_dir, settings.embedder_api_key()
        ),
        store,
        settings.retrieval_mode,
        make_reranker(settings.reranker, settings.model_cache_dir, settings.rerank_max_chars),
        settings.rerank_candidates,
        settings.rerank_language_set,
    )


def _config_snapshot() -> dict[str, object]:
    settings = get_settings()
    return {
        "embedding_model": settings.embedding_model,
        "chunking": settings.chunking,
        "contextual_header": settings.contextual_header,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "retrieval_mode": settings.retrieval_mode,
        "reranker": settings.reranker or "none",
        "rerank_candidates": settings.rerank_candidates,
        "rerank_languages": settings.rerank_languages or "all",
        "rerank_max_chars": settings.rerank_max_chars,
    }


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Directory of PDFs + YAML sidecars.")] = Path(
        "data/raw"
    ),
) -> None:
    """Chunk, embed and index every document (rebuilds the collection)."""
    from reglens.ingestion.pipeline import ingest as run_ingest
    from reglens.retrieval.embeddings import make_embedder
    from reglens.retrieval.vector_store import QdrantStore, make_client

    settings = get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    store = QdrantStore(
        make_client(settings.qdrant_url, settings.qdrant_path, api_key),
        settings.collection_name,
    )
    embedder = make_embedder(
        settings.embedding_model, settings.model_cache_dir, settings.embedder_api_key()
    )
    stats = run_ingest(
        path,
        embedder,
        store,
        settings.chunk_size,
        settings.chunk_overlap,
        strategy=settings.chunking,
        header=settings.contextual_header,
    )
    typer.echo(
        f"Indexed {stats.chunks} chunks from {stats.documents} documents with {embedder.name} "
        f"into collection {settings.collection_name}."
    )
    tokens = getattr(embedder, "tokens_used", 0)
    if tokens:
        typer.echo(f"Embedding API usage: {tokens:,} tokens.")
    if stats.truncated:
        typer.echo(
            f"WARNING: {stats.truncated} chunks exceed the embedding model's input window; "
            "their ends are ignored. Lower REGLENS_CHUNK_SIZE.",
            err=True,
        )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question in FR, AR or EN.")],
    k: Annotated[int | None, typer.Option("--k", "-k", help="Sources to retrieve.")] = None,
) -> None:
    """Answer a question with citations (sources only when no LLM is configured)."""
    from reglens.generation.llm import LiteLLMClient
    from reglens.rag import RagPipeline

    settings = get_settings()
    llm = None
    if settings.llm_model:
        key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        llm = LiteLLMClient(settings.llm_model, key, settings.llm_timeout_s)
    from reglens.guardrails import ScopeClassifier

    retriever = _retriever()
    scope = ScopeClassifier(retriever.embedder)
    answer = RagPipeline(
        retriever, llm, k or settings.top_k, scope, settings.abstain_min_score
    ).answer(question)

    if answer.guardrail != "ok" or answer.abstention != "none":
        typer.echo(answer.text or "")
        return
    if answer.text is None:
        typer.echo("No LLM configured (REGLENS_LLM_MODEL): showing the retrieved sources.\n")
    else:
        typer.echo(answer.text + "\n")
    for n, scored in enumerate(answer.sources, start=1):
        c = scored.chunk
        pages = f"p.{c.page_start}" + (f"-{c.page_end}" if c.page_end != c.page_start else "")
        where = f"{c.section}, {pages}" if c.section else pages
        typer.echo(f"[{n}] {c.issuer} {c.reference or '-'} {where} (score {scored.score:.3f})")
        typer.echo(f"    {c.text[:220]}...")


@app.command("eval")
def evaluate(
    golden_set: Annotated[Path, typer.Argument(help="Path to a golden set JSONL file.")] = Path(
        "eval/golden_set.jsonl"
    ),
    k: Annotated[int | None, typer.Option("--k", "-k", help="Retrieval depth.")] = None,
    label: Annotated[str, typer.Option(help="Name of this configuration.")] = "baseline",
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("eval/report"),
    judge: Annotated[bool, typer.Option(help="Use LLM-as-judge metrics (Phase 4).")] = False,
) -> None:
    """Measure retrieval quality (hit rate, MRR) on the golden set."""
    from reglens.evaluation.golden import load_golden
    from reglens.evaluation.runner import evaluate_retrieval, write_report

    if judge:
        typer.echo("LLM-as-judge metrics arrive in Phase 4; running retrieval metrics only.")
    settings = get_settings()
    depth = k or settings.top_k
    report = evaluate_retrieval(
        load_golden(golden_set), _retriever(), depth, label, _config_snapshot()
    )
    json_path, md_path = write_report(report, out)
    s = report.summary()
    typer.echo(
        f"{label}: hit@{depth} {s[f'hit@{depth}']:.1%} | MRR@{depth} {s[f'mrr@{depth}']:.3f} | "
        f"doc hit@{depth} {s[f'doc_hit@{depth}']:.1%} | n={s['n']:.0f}"
    )
    typer.echo(f"Report: {md_path} and {json_path}")


@app.command("eval-answers")
def eval_answers(
    golden_set: Annotated[Path, typer.Argument(help="Golden set JSONL.")] = Path(
        "eval/golden_set.jsonl"
    ),
    label: Annotated[str, typer.Option(help="Name of this configuration.")] = "answers",
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("eval/report"),
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
) -> None:
    """Generate answers for the golden set: fact recall, citations, abstention, latency."""
    from reglens.evaluation.answers import evaluate_answers, summarize, write_answer_report
    from reglens.evaluation.golden import load_golden
    from reglens.generation.llm import LiteLLMClient
    from reglens.rag import RagPipeline

    settings = get_settings()
    if not settings.llm_model:
        typer.echo("Set REGLENS_LLM_MODEL (and REGLENS_LLM_API_KEY) to evaluate answers.", err=True)
        raise typer.Exit(code=1)
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    llm = LiteLLMClient(settings.llm_model, key, settings.llm_timeout_s)
    retriever = _retriever()
    pipeline = RagPipeline(retriever, llm, settings.top_k, min_score=settings.abstain_min_score)
    items = load_golden(golden_set)[:limit]
    rows = evaluate_answers(items, pipeline)
    config = {
        **_config_snapshot(),
        "llm": settings.llm_model,
        "abstain_min_score": settings.abstain_min_score,
    }
    path = write_answer_report(rows, out / label, label, config)
    s = summarize(rows)
    typer.echo(
        f"{label}: numeric fact recall {s['numeric_fact_recall']:.1%} | "
        f"literal fact recall {s['fact_recall']:.1%} | "
        f"cites doc {s['cites_document']:.1%} | false abstention {s['false_abstention']:.1%} | "
        f"abstention acc. {s['abstention_accuracy']:.1%} | p50 {s['latency_p50_ms'] / 1000:.1f} s"
    )
    typer.echo(f"Report: {path}")


@app.command("eval-guardrails")
def eval_guardrails(
    guard_set: Annotated[Path, typer.Argument(help="Labelled guardrail cases (JSONL).")] = Path(
        "eval/guardrails_set.jsonl"
    ),
    golden_set: Annotated[Path, typer.Option(help="Golden set: every question must pass.")] = Path(
        "eval/golden_set.jsonl"
    ),
    out: Annotated[Path, typer.Option(help="Markdown report.")] = Path("eval/report/guardrails.md"),
) -> None:
    """Measure the input guardrails (meta, off-topic, greeting, false refusals)."""
    from reglens.evaluation.guardrails_eval import (
        evaluate_guardrails,
        load_cases,
        render_markdown,
    )
    from reglens.guardrails import ScopeClassifier
    from reglens.retrieval.embeddings import make_embedder

    settings = get_settings()
    embedder = make_embedder(
        settings.embedding_model, settings.model_cache_dir, settings.embedder_api_key()
    )
    report = evaluate_guardrails(load_cases(guard_set, golden_set), ScopeClassifier(embedder))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(report, embedder.name), encoding="utf-8")
    table = report.confusion()
    ok_total = sum(table["ok"].values())
    typer.echo(
        " | ".join(
            f"{k}: {report.accuracy(k):.1%}" for k in ("ok", "meta", "off_topic", "greeting")
        )
        + f" | false refusals {ok_total - table['ok']['ok']}/{ok_total}"
    )
    typer.echo(f"Report: {out}")


@app.command()
def serve() -> None:
    """Start the HTTP API."""
    _not_implemented("serve", 5)
