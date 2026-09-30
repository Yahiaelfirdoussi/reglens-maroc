# RegLens Maroc

Multilingual (FR / AR / EN) RAG assistant for Moroccan financial regulation
(Bank Al-Maghrib, AMMC, ACAPS). Answers are grounded in official texts, cite the
circular, article and page, and abstain when the texts do not contain the answer.

> Status: early development. Phase 0 (corpus, OCR, golden evaluation set) is done;
> see [docs/corpus.md](docs/corpus.md) and [docs/results.md](docs/results.md).

## Quick start

```bash
make install   # creates .venv and installs the package with all extras
make lint
make test
reglens --help
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
