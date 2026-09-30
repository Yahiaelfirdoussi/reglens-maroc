# Architecture

_To be written as components land._ Planned flow:

```
PDF + YAML sidecar → extraction (PyMuPDF / OCR) → cleaning → legal-structure chunking
  → dense (bge-m3) + sparse (BM25) vectors → Qdrant
query → guardrails → hybrid search (RRF) → reranker → abstention check
  → prompt with <source> tags → LLM (LiteLLM) → citation validation → answer
```
