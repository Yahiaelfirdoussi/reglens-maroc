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

## Harder questions (golden set v2)

20 hard questions were added (q077-q096): jargon the texts never use (CET1, LCR, ALCO),
everyday wording ("ma banque veut fermer mon compte"), Arabic paraphrases, and 10 documents
that had no question yet. 3 hard unanswerable questions keep the unanswerable share at 15%
(17/113). All 113 questions are reviewed and approved. One reviewer edit: q077 asks for the
CET1 requirement including the conservation buffer; its answer (8 %) is recorded as a derived
fact (5,5 % in Article 4 + 2,5 % in Article 5), with both articles as quoted evidence.

| Configuration | Hit@6 standard (n=76) | Hit@6 hard (n=20) | MRR@6 all (n=96) |
|---|---:|---:|---:|
| Baseline (local MiniLM) | 69.7% | 65.0% | 0.445 |
| OpenAI `text-embedding-3-small` (default) | 89.5% | 80.0% | 0.680 |

The hard set leaves room to measure Phase 2 changes. Hard misses with 3-small: q077 (CET1),
q080 (Arabic complaint), q086, q095 (Islamic-bank LCR).

## Guardrails

Measured with `reglens eval-guardrails` on 59 labelled cases (`eval/guardrails_set.jsonl`)
plus the 113 golden questions, which must all pass. Scope embeddings: `text-embedding-3-small`.
Reference phrases were written before the evaluation and not tuned on it.

| Expected | n | Correct |
|---|---:|---:|
| In scope (golden set + 10 look-alikes, e.g. "quel modèle de convention") | 123 | 100% (0 false refusals) |
| Questions about the system (model, provider, prompt, creators, overrides) | 22 | 100% |
| Greetings | 6 | 100% |
| Off-topic | 21 | 81.0% |

Off-topic misses (weather, a capital city, tourism, flu in Arabic) pass to retrieval, where
the Phase 3 abstention threshold and the system prompt's scope rule are the next layers.

## Phase 2, step 1: structure-aware chunking

One chunk per article (Titre/Chapitre/Article headings, BO "ARTICLE PREMIER" / "ART. 2",
AMMC "Article 1.20", Arabic "المادة"); long articles split on sentence boundaries with
200-character overlap; short neighbouring articles merged until 300 characters; a sentence
ending with ":" stays with the list or table it introduces (tables may grow to 2,400
characters); documents without articles fall back to sentence windows. Lower-case
"l'article 5 ci-dessus" references do not split; isolated OCR misreads of article numbers
are repaired from their neighbours (11, 42, 13 → 11, 12, 13). 44 of 47 documents have
gap-free article sequences (the rest: two amending circulars that quote only the articles
they change, and one missing heading).

Bigger chunks pass the passage metric more easily, so a fixed 1,200-character run isolates
the size effect from the structure effect. All runs: `text-embedding-3-small`, 96 questions.
**Section hit** = a top-6 chunk labelled with the expected article (only article chunks
carry labels).

| Configuration | Chunks | Hit@1 | Hit@3 | Hit@6 | MRR@6 | Hard hit@6 | FR hit@6 | AR hit@6 | Section hit@6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed 450 / 90 (before) | 4,118 | 54.2% | 80.2% | 87.5% | 0.680 | 80.0% | 97.9% | 60.0% | n/a |
| Fixed 1,200 / 200 (size only) | 1,492 | 67.7% | 79.2% | 86.5% | 0.747 | 75.0% | 95.7% | 50.0% | n/a |
| **By article ≤1,200 / 200** | 2,059 | 63.5% | **84.4%** | 87.5% | 0.732 | **85.0%** | **100%** | 55.0% | **77.1%** |

- Most of the MRR gain comes from chunk size (0.680 → 0.747), not structure. Structure adds a
  better top 3, better hard questions and perfect French retrieval, and it is the only
  option whose chunks can be cited by article (77.1% section hit; 89.4% on French).
- One question is worth ~1 point; differences of 1-2 points are within noise.
- Arabic stays weak whatever the chunking: the target of the query-rewriting step.
- **Decision:** article chunking (≤1,200 / 200) becomes the default.

## Phase 2, step 2: contextual header (rejected)

Each chunk embedded with a deterministic header before its text (the stored text stays the
official wording). Same setup as step 1: article chunks ≤1,200 / 200, `text-embedding-3-small`.

| Header before embedding | Hit@1 | Hit@3 | Hit@6 | MRR@6 | Doc hit@6 | Hard hit@6 | FR hit@6 | AR hit@6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **None (step 1, kept)** | 63.5% | 84.4% | **87.5%** | **0.732** | **90.6%** | **85.0%** | **100%** | 55.0% |
| Full: issuer \| reference \| title \| article | 61.5% | 78.1% | 83.3% | 0.701 | 86.5% | 75.0% | 89.4% | 60.0% |
| Short: issuer \| reference \| article | 57.3% | 77.1% | 84.4% | 0.680 | 88.5% | 80.0% | 97.9% | 45.0% |

- **Both headers lower every main metric**, so per the project rule the change is not kept
  (`REGLENS_CONTEXTUAL_HEADER` stays `off`; the option remains for later experiments).
- Why (inspected): every chunk of a document shares the same header, so chunks of one
  document look alike and the article body counts less. Results crowd into fewer
  documents (2.25 distinct documents in the top 6 vs 2.92 without a header). Example: q022
  (software deduction in amendment 2/W/2021) returns six chunks of the base circular
  14/G/2013, whose title matches the topic.
- The full header helped a few Arabic questions (q066, q086) but lost more elsewhere.
- Takeaway: metadata belongs in exact-match channels (BM25 over references, filters)
  rather than in the dense vector; this informs the hybrid-search step.
