# Evaluation results

Every retrieval, chunking or prompt change is recorded here with before/after numbers,
measured on `eval/golden_set.jsonl`.

Metrics: a **passage hit** is a retrieved chunk from the expected document that contains at
least 50% of the evidence quote's words (every one of the 76 answerable questions has such a
chunk, so every miss is a real retrieval failure); **doc hit** only needs the right document.
Run with `reglens eval eval/golden_set.jsonl --label <name>`.

| Date | Configuration | Hit@1 | Hit@6 | MRR@6 | Doc hit@6 | Fact recall | Abstention acc. | Notes |
|---|---|---:|---:|---:|---:|---|---|---|
| 2026-09-30 | **Baseline**: fixed 450-char chunks (90 overlap), dense `paraphrase-multilingual-MiniLM-L12-v2` (384-d, ONNX), Qdrant cosine, k=6 | 35.5% | 69.7% | 0.464 | 80.3% | n/a | n/a | No LLM yet (retrieval only). p50 retrieval 15 ms. |

Baseline by question language (n = 38 FR, 22 EN, 16 AR):

| Language | Hit@6 | MRR@6 | Doc hit@6 |
|---|---:|---:|---:|
| French | 73.7% | 0.529 | 86.8% |
| English | 77.3% | 0.480 | 81.8% |
| Arabic | 50.0% | 0.289 | 62.5% |

What the 23 baseline misses have in common (inspected by hand):

- **Questions that name a reference** ("circulaire n°5/W/2017"): dense vectors do not match
  reference strings. Expected fix: contextual header + BM25 hybrid.
- **Arabic questions against French texts**: the small model is weak cross-lingually.
  Expected fix: stronger multilingual embeddings, query rewriting.
- **Amending circulars** (e.g. 2/W/2021 amends 14/G/2013): the base text outranks the
  amendment. Expected fix: contextual header, reranking.
- The model reads only 128 tokens (~480 French characters); 227 of 4,118 chunks (5.5%) are
  longer in tokens and their ends are ignored.

## OCR (Phase 0)

35 of 47 documents are scans. OCR quality is measured by rendering the 11 French
native-text documents (97 pages) to images, OCR'ing them, and comparing with their text layer
(`reglens corpus-report`, full table in [corpus.md](corpus.md)). This is an optimistic bound
for real scans.

| Date | Configuration | CER | WER | Number recall | Notes |
|---|---|---:|---:|---:|---|
| 2026-09-30 | Tesseract via PyMuPDF OCR-to-PDF, column-aware | see below | | | Table rows silently dropped |
| 2026-09-30 | **Tesseract CLI plain text, column-aware** | **2.9%** | **5.5%** | **94.0%** (1338/1424) | Current engine |

Change: PyMuPDF's OCR-to-PDF path dropped whole table rows (e.g. the risk-weight row
"0 % · 20 % · 50 % · 100 % · 150 %" of circular 26/G/2006, art. 11), while Tesseract's own
text output keeps them. Per-document impact, same benchmark:

| Document | CER before → after | Number recall before → after |
|---|---|---|
| BAM 26/G/2006 (risk-weight tables) | 3.9% → 2.6% | 78.8% → 94.5% |
| AMMC 01/23 (Bulletin officiel, two columns) | 9.5% → 5.2% | 97.7% → 90.1%\* |
| BAM 25/G/2006 | 5.8% → 5.9% | 100% → 100% |

\* Before/after number recall is not strictly comparable: the metric was corrected at the same
time to exclude the page's own number (the old version counted page numbers as facts). The
"before" totals also mixed in an Arabic document, so only per-document values are shown.

Known limits:

- Remaining number misses are mostly letterhead phone/fax numbers (BAM 2/G/10: 60%) and
  dense annex tables (AMMC 01/23).
- AMMC 02/20 is in Arabic (Bulletin officiel Arabic edition). Its text layer stores glyphs in
  visual order, so it is not a usable OCR reference (CER 47% against it); it is excluded from
  the totals and will need its own handling in Phase 1.
