"""Command-line interface: ``reglens ingest | ask | eval | serve``."""

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from reglens.config import get_settings
from reglens.log import configure_logging

if TYPE_CHECKING:
    from reglens.evaluation.answers import AnswerResult
    from reglens.evaluation.golden import GoldenItem
    from reglens.rag import Answer, RagPipeline
    from reglens.retrieval.retriever import Retriever

app = typer.Typer(help="RegLens Maroc — grounded answers on Moroccan financial regulation.")


@app.callback()
def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)


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
    from reglens.service import EmptyIndexError, build_retriever

    try:
        return build_retriever()
    except EmptyIndexError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


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
    update: Annotated[
        bool,
        typer.Option(help="Only add new or changed documents (skip unchanged, keep the index)."),
    ] = False,
) -> None:
    """Chunk, embed and index documents (rebuilds the collection unless --update)."""
    from reglens.ingestion.pipeline import ingest as run_ingest
    from reglens.ingestion.pipeline import ingest_document
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
    if update:
        from collections import Counter

        outcomes: Counter[str] = Counter()
        for pdf in sorted(path.rglob("*.pdf")):
            status, _ = ingest_document(
                pdf,
                path,
                embedder,
                store,
                settings.chunk_size,
                settings.chunk_overlap,
                settings.chunking,
                settings.contextual_header,
            )
            outcomes[status] += 1
        typer.echo(
            f"{outcomes['added']} added, {outcomes['updated']} updated, "
            f"{outcomes['unchanged']} unchanged ({settings.collection_name})."
        )
        return
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


def _pipeline(k: int | None = None) -> "RagPipeline":
    """The full answering pipeline: guardrails, retrieval, abstention, LLM (if configured)."""
    from reglens.service import EmptyIndexError, build_pipeline

    try:
        return build_pipeline(k=k)
    except EmptyIndexError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


def _print_answer(answer: "Answer", all_sources: bool = False) -> None:
    """The answer, then its cited sources (or every retrieved source)."""
    if answer.guardrail != "ok" or answer.abstention != "none":
        typer.echo(answer.text or "")
        return
    if answer.text is None:
        typer.echo("No LLM configured (REGLENS_LLM_MODEL): showing the retrieved sources.")
        all_sources = True
    else:
        typer.echo(answer.text)
    cited = {c.number for c in answer.citations}
    shown = [(n, s) for n, s in enumerate(answer.sources, start=1) if all_sources or n in cited]
    if shown:
        typer.echo("\nSources:")
    for n, scored in shown:
        c = scored.chunk
        pages = f"p.{c.page_start}" + (f"-{c.page_end}" if c.page_end != c.page_start else "")
        where = f"{c.section}, {pages}" if c.section else pages
        typer.echo(f"[{n}] {c.issuer} {c.reference or '-'} · {where} · {c.title[:70]}")
        typer.echo(f"    « {c.text[:200].strip()}… »")
    timing = answer.retrieval_ms + (answer.generation_ms or 0.0)
    typer.echo(f"\n({timing / 1000:.1f} s)")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question in FR, AR or EN.")],
    k: Annotated[int | None, typer.Option("--k", "-k", help="Sources to retrieve.")] = None,
    all_sources: Annotated[
        bool, typer.Option(help="Show every retrieved source, not only the cited ones.")
    ] = False,
) -> None:
    """Answer one question with citations."""
    _print_answer(_pipeline(k).answer(question), all_sources)


@app.command()
def chat(
    all_sources: Annotated[
        bool, typer.Option(help="Show every retrieved source, not only the cited ones.")
    ] = False,
) -> None:
    """Ask questions one after another (setup loaded once). Empty line or Ctrl-D to quit."""
    pipeline = _pipeline()
    typer.echo("RegLens: ask a question in French, English or Arabic. Empty line to quit.\n")
    while True:
        try:
            question = input("» ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        _print_answer(pipeline.answer(question), all_sources)
        typer.echo("")


def _answer_rows(
    items: list["GoldenItem"], retriever: "Retriever", judge: bool
) -> tuple[list["AnswerResult"], dict[str, object]]:
    """Generate (and optionally judge) answers for golden items."""
    from reglens.evaluation.answers import evaluate_answers
    from reglens.evaluation.judge import Judge
    from reglens.generation.llm import LiteLLMClient
    from reglens.rag import RagPipeline

    settings = get_settings()
    if not settings.llm_model:
        typer.echo("Set REGLENS_LLM_MODEL (and REGLENS_LLM_API_KEY) to evaluate answers.", err=True)
        raise typer.Exit(code=1)
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    llm = LiteLLMClient(
        settings.llm_model, key, settings.llm_timeout_s, settings.llm_reasoning_effort
    )
    grader = None
    if judge:
        if not settings.judge_model:
            typer.echo("Set REGLENS_JUDGE_MODEL to use the LLM-as-judge.", err=True)
            raise typer.Exit(code=1)
        grader = Judge(LiteLLMClient(settings.judge_model, key, settings.llm_timeout_s))
    pipeline = RagPipeline(retriever, llm, settings.top_k, min_score=settings.abstain_min_score)
    config: dict[str, object] = {
        "llm": settings.llm_model,
        "judge": settings.judge_model if judge else "none",
        "reasoning_effort": settings.llm_reasoning_effort or "default",
        "abstain_min_score": settings.abstain_min_score,
    }
    return evaluate_answers(items, pipeline, grader), config


@app.command("eval")
def evaluate(
    golden_set: Annotated[Path, typer.Argument(help="Path to a golden set JSONL file.")] = Path(
        "eval/golden_set.jsonl"
    ),
    k: Annotated[int | None, typer.Option("--k", "-k", help="Retrieval depth.")] = None,
    label: Annotated[str, typer.Option(help="Name of this configuration.")] = "baseline",
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("eval/report"),
    answers: Annotated[bool, typer.Option(help="Also generate and score answers.")] = False,
    judge: Annotated[bool, typer.Option(help="Grade answers with the LLM-as-judge.")] = False,
    min_hit_rate: Annotated[
        float | None, typer.Option(help="Fail (exit 1) when hit@k is below this value.")
    ] = None,
) -> None:
    """Evaluate retrieval (and optionally answers) on a golden set; write report.md/json."""
    import json

    from reglens.evaluation.answers import render_markdown as render_answers
    from reglens.evaluation.answers import summarize
    from reglens.evaluation.golden import load_golden
    from reglens.evaluation.runner import evaluate_retrieval, write_report

    settings = get_settings()
    depth = k or settings.top_k
    items = load_golden(golden_set)
    retriever = _retriever()
    report = evaluate_retrieval(items, retriever, depth, label, _config_snapshot())
    json_path, md_path = write_report(report, out)
    s = report.summary()
    typer.echo(
        f"{label}: hit@{depth} {s[f'hit@{depth}']:.1%} | MRR@{depth} {s[f'mrr@{depth}']:.3f} | "
        f"doc hit@{depth} {s[f'doc_hit@{depth}']:.1%} | n={s['n']:.0f}"
    )

    if answers or judge:
        rows, answer_config = _answer_rows(items, retriever, judge)
        prices = (settings.llm_price_in, settings.llm_price_out)
        summary = summarize(rows, *prices)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload["answers"] = {
            "config": answer_config,
            "summary": summary,
            "rows": [asdict(r) for r in rows],
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(
            md_path.read_text(encoding="utf-8")
            + "\n"
            + render_answers(rows, label, {**_config_snapshot(), **answer_config}, *prices).replace(
                "# Answer evaluation", "## Answers", 1
            ),
            encoding="utf-8",
        )
        typer.echo(
            f"answers: numeric facts {summary['numeric_fact_recall']:.1%} | cites doc "
            f"{summary['cites_document']:.1%} | abstention acc. "
            f"{summary['abstention_accuracy']:.1%} | false abstention "
            f"{summary['false_abstention']:.1%}"
            + (
                f" | faithfulness {summary['faithfulness']:.1%} | correctness "
                f"{summary['correctness']:.1%}"
                if summary["faithfulness"] is not None and summary["correctness"] is not None
                else ""
            )
        )
    typer.echo(f"Report: {md_path} and {json_path}")

    hit = s[f"hit@{depth}"]
    if min_hit_rate is not None and hit < min_hit_rate:
        typer.echo(f"FAIL: hit@{depth} {hit:.1%} is below the gate {min_hit_rate:.1%}.", err=True)
        raise typer.Exit(code=1)


@app.command("eval-answers")
def eval_answers(
    golden_set: Annotated[Path, typer.Argument(help="Golden set JSONL.")] = Path(
        "eval/golden_set.jsonl"
    ),
    label: Annotated[str, typer.Option(help="Name of this configuration.")] = "answers",
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("eval/report"),
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
    judge: Annotated[bool, typer.Option(help="Grade answers with the LLM-as-judge.")] = False,
) -> None:
    """Generate answers only: facts, citations, abstention, judge, latency, cost."""
    from reglens.evaluation.answers import summarize, write_answer_report
    from reglens.evaluation.golden import load_golden

    settings = get_settings()
    rows, answer_config = _answer_rows(load_golden(golden_set)[:limit], _retriever(), judge)
    prices = (settings.llm_price_in, settings.llm_price_out)
    config = {**_config_snapshot(), **answer_config}
    path = write_answer_report(rows, out / label, label, config, *prices)
    s = summarize(rows, *prices)
    typer.echo(
        f"{label}: numeric facts {s['numeric_fact_recall']:.1%} | cites doc "
        f"{s['cites_document']:.1%} | abstention acc. {s['abstention_accuracy']:.1%} | "
        f"false abstention {s['false_abstention']:.1%}"
        + (
            f" | faithfulness {s['faithfulness']:.1%} | correctness {s['correctness']:.1%}"
            if s["faithfulness"] is not None and s["correctness"] is not None
            else ""
        )
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
def serve(
    host: Annotated[str | None, typer.Option(help="Interface to bind.")] = None,
    port: Annotated[int | None, typer.Option(help="Port.")] = None,
) -> None:
    """Start the HTTP API (FastAPI + uvicorn)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "reglens.api.main:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        log_config=None,  # structlog handles logging
    )
