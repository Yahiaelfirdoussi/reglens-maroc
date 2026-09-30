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
| 2026-09-30 | + OpenAI `text-embedding-3-small` (1536-d), same chunks | 53.9% | 89.5% | 0.686 | 90.8% | n/a | n/a | Index: 498k tokens (~$0.01). p50 retrieval 305 ms. |
| 2026-09-30 | + OpenAI `text-embedding-3-large` (3072-d), same chunks | 71.1% | **97.4%** | **0.812** | 100.0% | n/a | n/a | Index: 498k tokens (~$0.07). p50 retrieval 394 ms. |

Hit@6 by question language (n = 38 FR, 22 EN, 16 AR):

| Language | Baseline (MiniLM) | OpenAI 3-small | OpenAI 3-large |
|---|---:|---:|---:|
| French | 73.7% | 100.0% | 97.4% |
| English | 77.3% | 90.9% | 100.0% |
| Arabic | 50.0% | 62.5% | **93.8%** |

Embedding comparison (only the embedding model changed; same 4,118 chunks, same questions):

- `text-embedding-3-large` removes most baseline failures: 2 misses left (q010, q041) against
  23 for the baseline, and every question finds the right document. The Arabic gap closes
  (50.0% → 93.8%): the small local model was the main cause of the cross-lingual weakness.
- `text-embedding-3-small` fixes French (100%) but leaves Arabic weak (62.5%).
- Costs: indexing is a one-off 498k tokens; each query adds one API call, which moves
  retrieval latency from ~15 ms (local) to ~300-400 ms (network) and requires a key and
  network access. The local model stays the offline/CI default.
- **Decision: `text-embedding-3-small` is the default.** Expected users ask mostly in French,
  where 3-small reaches 100% hit@6 (above 3-large's 97.4%) at about a seventh of the cost.
  The accepted trade-off is Arabic (62.5%); Phase 2 query rewriting (Arabic/English →
  French legal terms) targets that gap.
- Caveat: at 97.4% hit@6 the golden set is close to its ceiling. The questions reuse the
  wording of the articles they target; further Phase 2 gains will need harder questions
  (paraphrases, multi-article questions) to be measurable.

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
