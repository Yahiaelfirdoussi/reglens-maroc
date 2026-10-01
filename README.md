# RegLens Maroc

Multilingual (FR / AR / EN) RAG assistant for Moroccan financial regulation
(Bank Al-Maghrib, AMMC, ACAPS). Answers are grounded in official texts, cite the
circular, article and page, and abstain when the texts do not contain the answer.

> Status: early development. Corpus, OCR, golden evaluation set and retrieval are done:
> the right passage is in the top 6 for 95.8% of 96 test questions (French 100%,
> English 96.6%, Arabic 85%), with article-based chunks and `text-embedding-3-large`;
> see [docs/corpus.md](docs/corpus.md) and [docs/results.md](docs/results.md).

## Quick start

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,llm,api,ui]"
.venv/bin/ruff check . && .venv/bin/mypy src   # lint + types
.venv/bin/pytest -q                            # tests (offline: no model, no API key)
.venv/bin/reglens --help
```

## Layout

| Path | Purpose |
|---|---|
| `src/reglens/` | Library code (ingestion, retrieval, generation, evaluation, api) |
| `data/raw/` | Source PDFs + `<file>.yaml` metadata sidecars |
| `data/sources.yaml` | Corpus manifest |
| `eval/golden_set.jsonl` | Golden evaluation questions |
| `docs/` | Architecture notes and evaluation results |
| `tests/` | Unit tests and FICTIONAL fixtures |
