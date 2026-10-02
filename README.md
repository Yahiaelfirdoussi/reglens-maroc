# RegLens Maroc

Multilingual (FR / AR / EN) RAG assistant for Moroccan financial regulation
(Bank Al-Maghrib, AMMC). Answers are grounded in official texts, cite the circular, article
and page, and abstain when the texts do not contain the answer.

> Status: retrieval, answers, guardrails and evaluation are built and measured on 113 test
> questions: the right passage is in the top 6 for 95.8% of answerable questions, answers
> are 96.3% faithful to their sources (LLM-as-judge), and unanswerable questions are
> refused. See [docs/results.md](docs/results.md) and [docs/corpus.md](docs/corpus.md).

## Quick start

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,llm,ui]"
.venv/bin/pytest -q                            # tests (offline: no model, no API key)
```

Configuration comes from environment variables prefixed `REGLENS_` (or a local `.env`);
`src/reglens/config.py` lists them with their defaults. Answering needs an embedding and an
LLM provider key, for example:

```bash
REGLENS_EMBEDDING_MODEL=openai/text-embedding-3-large
REGLENS_EMBEDDING_API_KEY=...
REGLENS_LLM_MODEL=openai/gpt-5.4-mini-2026-03-17
REGLENS_LLM_API_KEY=...
REGLENS_CHUNKING=legal
REGLENS_CHUNK_SIZE=1200
REGLENS_CHUNK_OVERLAP=200
```

## Use it

```bash
.venv/bin/reglens fetch --selection data/selection.yaml   # download the 47 official texts
.venv/bin/reglens ocr data/raw                             # OCR the scanned ones
.venv/bin/reglens ingest data/raw                          # chunk, embed, index
.venv/bin/reglens ask "Quel est le ratio de levier minimum ?"
.venv/bin/reglens chat                                     # several questions in a row
.venv/bin/streamlit run ui/app.py                          # chat interface, port 8501
.venv/bin/reglens eval --answers --judge                   # full evaluation report
```

## Layout

| Path | Purpose |
|---|---|
| `src/reglens/` | Library code (ingestion, retrieval, generation, evaluation, guardrails) |
| `ui/app.py` | Streamlit chat interface |
| `data/selection.yaml` | Curated corpus: official URLs and verified metadata |
| `data/raw/` | Source PDFs + `<file>.yaml` metadata sidecars (PDFs not committed) |
| `eval/golden_set.jsonl` | 113 reviewed evaluation questions with quoted evidence |
| `docs/` | Corpus report, results of every experiment |
| `tests/` | Unit tests and FICTIONAL fixtures (CI evaluation gate) |
