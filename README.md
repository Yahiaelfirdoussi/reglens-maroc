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
uv pip install --python .venv/bin/python -e ".[dev,llm,api,ui,local]"   # local: offline ONNX models
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
.venv/bin/reglens ingest data/raw                          # chunk, embed, index (--update: changed files only)
.venv/bin/reglens ask "Quel est le ratio de levier minimum ?"
.venv/bin/reglens chat                                     # several questions in a row
.venv/bin/streamlit run ui/app.py                          # chat interface, port 8501
.venv/bin/reglens eval --answers --judge                   # full evaluation report
```

## API

```bash
.venv/bin/reglens serve                     # http://127.0.0.1:8000, docs at /docs
curl -X POST localhost:8000/v1/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"question": "Quel est le ratio de levier minimum ?"}'
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/query` | Cited answer, sources, status (`answered`, `not_found`, `out_of_scope`) |
| `POST /v1/query/stream` | Same, as Server-Sent Events: `token` events, then one `answer` event |
| `POST /v1/ingest` | Admin key: upload a PDF + metadata; OCR'd if scanned; unchanged files skipped |
| `GET /health`, `GET /ready` | Liveness; readiness (index loaded, 503 with the reason otherwise) |
| `GET /metrics` | Prometheus: requests, stage latencies, cache, guardrails, abstentions, tokens |

Every response carries an `X-Request-ID`, also present on every JSON log line. Repeated
questions are answered from a time-limited cache, cleared on each ingest.

## Docker

```bash
docker compose up -d                                   # Qdrant + API (:8000) + interface (:8501)
docker compose run --rm api reglens ingest /data/raw   # index ./data/raw once
```

Images are slim, run as a non-root user and include Tesseract (French, Arabic) for scanned
uploads. CI builds both images on every push.

## Configuration

All settings are environment variables prefixed `REGLENS_` (a local `.env` is read too);
defaults live in `src/reglens/config.py`. The main ones:

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` / `EMBEDDING_API_KEY` | local MiniLM | `openai/text-embedding-3-large` is the measured best |
| `LLM_MODEL` / `LLM_API_KEY` | none | Any LiteLLM model; none = sources only |
| `LLM_REASONING_EFFORT` | provider default | `minimal` for faster answers |
| `JUDGE_MODEL` | none | LLM-as-judge for `reglens eval --judge` |
| `CHUNKING`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | `fixed`, 450, 90 | `legal`, 1200, 200 is the measured best |
| `RETRIEVAL_MODE`, `RERANKER` | `dense`, none | `hybrid` / `sparse`; local cross-encoder |
| `TOP_K`, `ABSTAIN_MIN_SCORE` | 6, 0.35 | Sources per answer; "not found" floor |
| `QDRANT_URL` / `QDRANT_PATH` | local `data/qdrant` | Qdrant server URL (Docker) or embedded folder |
| `API_KEYS`, `ADMIN_API_KEYS` | none (open) | Comma-separated `X-API-Key` values |
| `RATE_LIMIT_PER_MINUTE` | 30 | Per key (per IP when open) |
| `CORS_ORIGINS` | none | Comma-separated allowed origins |
| `ANSWER_CACHE_TTL_S`, `UPLOAD_MAX_MB` | 600, 20 | Answer cache lifetime; upload size cap |
| `LOG_JSON`, `LOG_LEVEL` | false, INFO | JSON logs in production |

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
