# Evaluation results

Every retrieval, chunking or prompt change is recorded here with before/after numbers,
measured on `eval/golden_set.jsonl`.

| Date | Configuration | Hit rate@6 | MRR | Fact recall | Abstention acc. | Notes |
|---|---|---|---|---|---|---|
| | Baseline | | | | | |

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
