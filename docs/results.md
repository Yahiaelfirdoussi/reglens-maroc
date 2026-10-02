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
- **Decision (later superseded by `3-large`, see step 4): `text-embedding-3-small` is the default.** Expected users ask mostly in French,
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

## Phase 2, step 3: hybrid search (BM25 + dense, RRF) — not adopted

Every chunk now also carries a BM25 sparse vector (Qdrant IDF modifier). The BM25 text is
the chunk plus its reference and article label: metadata goes to the exact-match channel,
not the dense vector. The tokenizer folds accents, drops French/English stop words and
normalises references ("n°5/W/2017", "5W2017", "5/W/17" → one token). Hybrid fuses the top
30 of each side with Reciprocal Rank Fusion. All three modes run on the same index
(article chunks ≤1,200 / 200, `text-embedding-3-small`), so only the search mode changes.

| Search mode | Hit@1 | Hit@3 | Hit@6 | MRR@6 | FR MRR | EN MRR | AR MRR | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Dense (kept)** | **64.6%** | **84.4%** | 87.5% | **0.737** | **0.872** | **0.747** | **0.406** | 285 ms |
| BM25 only | 33.3% | 49.0% | 57.3% | 0.415 | 0.716 | 0.195 | 0.027 | 64 ms |
| Hybrid (RRF) | 57.3% | 79.2% | 87.5% | 0.687 | 0.837 | 0.652 | 0.385 | 368 ms |

- Hybrid finds the same answers (hit@6 87.5%) but ranks them lower (MRR 0.737 → 0.687).
  BM25 cannot match across languages (EN MRR 0.195, AR 0.027 against French texts), and
  equal-weight fusion lets that noise push good dense results down. Even in French, BM25
  alone (0.716) is below dense (0.872).
- The dense run reproduces step 1 (MRR 0.737 vs 0.732; re-embedding noise).
- **Test-set gap:** only 2 of 96 questions name a reference (q015, q047), BM25's main
  strength, and dense already finds both. Reference-style questions ("Que dit l'article 6
  de la circulaire 4/W/2018 ?") should be added before hybrid is judged for good.
- **Decision:** dense stays the default (`REGLENS_RETRIEVAL_MODE=dense`). BM25 vectors stay
  in the index at no cost: hybrid is one setting away, and BM25 can feed reranker candidates.

## Phase 2, step 4: cross-encoder reranking

`jinaai/jina-reranker-v2-base-multilingual` (local ONNX via FastEmbed, 1.1 GB) re-scores the
first-stage candidates (question and chunk read together) and keeps the top 6. Same index
(article chunks, `text-embedding-3-small`, dense). The top 20 dense results contain the
right passage for 93.8% of questions (top 30: 95.8%), the ceiling for reranking; hybrid
candidates add no recall (also 93.8% at 20).

| Configuration | Hit@1 | Hit@3 | Hit@6 | MRR@6 | Doc hit@6 | Section hit@6 | FR hit@6 | EN hit@6 | AR hit@6 | Hard hit@6 | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense (before) | 64.6% | 84.4% | 87.5% | 0.737 | 90.6% | 77.1% | **100%** | 89.7% | 55.0% | **85.0%** | **0.3 s** |
| + rerank top 10 | 67.7% | 82.3% | 85.4% | 0.752 | 89.6% | 76.0% | 95.7% | 89.7% | 55.0% | 80.0% | 2.6 s |
| **+ rerank top 20** | **71.9%** | **88.5%** | **90.6%** | **0.804** | **95.8%** | **81.2%** | 97.9% | 89.7% | **75.0%** | 75.0% | 5.4 s |

- Reranking the top 20 gives the best results of Phase 2: MRR 0.737 → 0.804, hit@1 +7.3
  points, and the largest Arabic gain so far (hit@6 55% → 75%, MRR 0.406 → 0.650).
- **Flagged:** French hit@6 100% → 97.9% (1 question) and hard questions 85% → 75%
  (2 questions).
- Reranking only the top 10 is worse than no reranking: Arabic answers sit deep in the dense
  ranking, so the reranker needs the full 20 candidates.
- **Cost: latency.** On a laptop CPU, scoring 20 pairs of up to 1,200 characters takes about
  5 s per question (p50 5.4 s, p95 7.8 s), against 0.3 s for dense search alone.

### Making Arabic retrieval fast (step 4, continued)

Users ask mostly in French and English, and 5 s per question is too slow. The reranker's
gain is mostly Arabic, so three faster options were measured:

1. **Rerank Arabic questions only** (`REGLENS_RERANK_LANGUAGES=ar`, language detection
   from the guardrails; the model loads lazily, so French/English never pay for it):
   hit@6 91.7%, MRR 0.788, French/English unchanged at 0.3 s.
2. **Shorter reranker input** on Arabic questions, timed on this Intel laptop without
   other load (20 candidates): full text 7.1 s, first 600 characters 3.7 s, first 400
   characters 3.0 s (hit@6 unchanged at 75%), max 256 tokens 4.5 s. 3 s is the floor here.
3. **`text-embedding-3-large` on the article chunks, no reranker.** Earlier `3-large` had
   lifted Arabic from 62.5% to 93.8% on fixed chunks, so it was re-measured on the current
   chunking (indexing ~419k tokens, ~$0.05):

| Article chunks, 96 questions | Hit@1 | Hit@6 | MRR@6 | Doc hit@6 | Section hit@6 | FR hit@6 | EN hit@6 | AR hit@6 | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `3-small` | 64.6% | 87.5% | 0.737 | 90.6% | 77.1% | 100% | 89.7% | 55.0% | 0.3 s |
| `3-small` + reranker on Arabic only | n/a | 91.7% | 0.788 | n/a | n/a | 100% | 89.7% | 75.0% | 0.3 s (AR 3-7 s) |
| Route Arabic → `3-large`, FR/EN → `3-small` | n/a | 93.8% | 0.801 | n/a | n/a | 100% | 89.7% | 85.0% | 0.3 s |
| **`3-large` for everyone** | **74.0%** | **95.8%** | **0.828** | **99.0%** | **85.4%** | **100%** | **96.6%** | **85.0%** | 0.5 s |

(Rows 2-3 are computed exactly from per-question results of the measured runs.)

- **Decision: `text-embedding-3-large` for every question, no reranker.** It beats the
  reranker on Arabic (85% vs 75%) at a tenth of the latency, adds 7 points on English,
  keeps French at 100%, and needs one index instead of two. This **supersedes** the
  earlier choice of `3-small`: on article chunks the gap is much larger than on fixed
  chunks, and the extra cost is ~$0.05 per re-index (questions stay well under a cent
  per thousand).
- The reranker stays available (`REGLENS_RERANKER`, `REGLENS_RERANK_LANGUAGES`,
  `REGLENS_RERANK_MAX_CHARS`, lazy loading) but off.

## Phase 3: answers, citations and abstention

Setup: article chunks, `text-embedding-3-large`, top 6 sources, `gpt-5.4-mini-2026-03-17`
via LiteLLM (temperature 0). All 113 golden questions; `reglens eval-answers`.

Abstention works in two layers. A score floor (0.35) answers "not found" without calling the
LLM; it is deliberately low because top retrieval scores of answerable and unanswerable
questions overlap (answerable 0.41-0.81, unanswerable 0.23-0.63; the lowest answerable
scores are all Arabic, cross-lingual), so a single threshold cannot separate them. The LLM
then decides: it must reply exactly `NOT_FOUND` when the sources do not contain the answer,
and that signal becomes a fixed answer in the question's language.

| Slice | n | Numeric fact recall | Cites right document | Cites right article | Citation coverage | False abstention | Abstention accuracy | p50 latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **All** | 113 | **83.2%** | **97.8%** | **87.9%** | **98.9%** | 5.2% | **94.1%** | 1.6 s |
| French | 54 | 88.5% | 97.9% | 85.1% | 98.9% | 0.0% | 85.7% | 1.6 s |
| English | 34 | 76.9% | 100% | 92.6% | 98.1% | 6.9% | 100% | 1.5 s |
| Arabic | 25 | 76.5% | 94.1% | 88.2% | 100% | 15.0% | 100% | 1.7 s |
| Hard questions | 23 | 50.0% | 100% | 72.2% | 100% | 10.0% | 100% | 1.7 s |

Metric notes (each found by reading the answers, then fixed in the scorer):

- **Numeric fact recall** checks that every number of an expected fact appears in the answer
  ("30 jours ouvrables" is found in "30 business days"; "trente" = 30; "5,5 %" = "5.5%" =
  "٥٫٥٪"; "200.000" = "200 000"). Text facts are French wording, so literal matching
  under-counts correct English and Arabic answers (literal recall 57.5%); judging them needs
  the LLM-as-judge of Phase 4.
- Citation coverage counts a citation written after the final punctuation ("…100%.[1]").

Behaviour fixes made during the phase:

- English questions were answered in French (the sources' language): the question's
  language is now stated explicitly to the model.
- `NOT_FOUND` written after an explanation was missed: the signal is now detected anywhere.
- The prompt forbids answering a different question with related sources.

Abstention, question by question:

- **Unanswerable: 16/17 refused (94.1%).** The miss, u007 (deposit-guarantee cap, the
  deliberate near-miss), states that no capped amount is given and cites a source instead of
  replying `NOT_FOUND`. It invents no figure.
- **Answerable refused: 5/96 (5.2%).** 4 of them (q010, q051, q077, q086) are retrieval
  misses: the right passage was not in the sources, so "not found" is the faithful answer.
  Only q042 is a true model error (right passage ranked 2nd).
- Average 1,618 tokens per answer (about 1,500 in, 120 out).

## Phase 4: evaluation as a system

One command, `reglens eval eval/golden_set.jsonl --answers --judge`, writes
`eval/report/report.md` and `report.json` with retrieval, answer, citation, abstention,
judge, latency and token metrics. The LLM-as-judge is a stronger model than the answering
one (`gpt-5.4-2026-03-05` judging `gpt-5.4-mini-2026-03-17`), since a model grading its own
answers is lenient. It sees the question, answer, numbered sources and the reference
(expected facts and official passage) and returns JSON.

| Slice | n | Faithfulness | Fully faithful | Correctness (judge) | Relevance (1-5) | Numeric facts | Cites doc | Cites article | False abst. | Abst. acc. | p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **All** | 113 | **96.3%** | 94.6% | **94.6%** | **4.89** | 85.3% | 98.9% | 88.0% | 4.2% | 88.2% | 1.7 s |
| French | 54 | 95.6% | n/a | 95.7% | 4.91 | 90.4% | 100% | 87.2% | 0.0% | 85.7% | 1.8 s |
| English | 34 | 100% | n/a | 100% | 5.00 | 76.9% | 100% | 92.6% | 6.9% | 100% | 1.5 s |
| Arabic | 25 | 92.6% | n/a | 83.3% | 4.67 | 82.4% | 94.4% | 83.3% | 10.0% | 80.0% | 1.8 s |
| Hard | 23 | 99.7% | n/a | 97.2% | 4.94 | 60.0% | 100% | 72.2% | 10.0% | 66.7% | 1.8 s |

Retrieval in the same run: hit@6 95.8%, MRR@6 0.828. 92 answers judged (answerable and
answered). Average 1,622 tokens per answer; cost per query is reported once prices are set
(`REGLENS_LLM_PRICE_IN` / `REGLENS_LLM_PRICE_OUT`, USD per million tokens).

How to read it:

- **Judge correctness vs literal matching.** English answers reach 100% correctness with the
  judge against 21.7% literal fact recall: the facts are there, translated. The judge
  resolves the text-fact limitation noted in Phase 3.
- **Run-to-run variation.** Abstention accuracy was 94.1% in Phase 3 and 88.2% here: one
  question (u017) flipped. The model is not fully deterministic at temperature 0, and with
  17 unanswerable questions one question is worth ~6 points. Differences below that are
  noise; repeated runs would give an interval.
- **The judge also errs.** It flagged q024 (risk weight 50 % for BBB+ to BBB- sovereigns)
  as unsupported although the figure is in the source table; the flattened table row likely
  confused it. Faithfulness is, if anything, slightly underestimated.
- Remaining issues: hard questions (numeric facts 60%), Arabic correctness (83.3%), and
  the deposit-guarantee near-miss (u007) still answered with a hedge instead of `NOT_FOUND`.

**CI gate.** `tests/fixtures/` holds a FICTIONAL four-document corpus and 9 golden
questions. CI turns it into PDFs, indexes it through `reglens ingest` (offline hash
embedder, local Qdrant) and runs `reglens eval --min-hit-rate 0.85`, which exits with an
error below the gate. Fixture hit@6 today: 100% (8/8), so one regression is tolerated and
two fail the build.

## Interface and speed

`reglens chat` (questions in a row) and a Streamlit interface (`ui/app.py`: streamed answers,
grounded / not-found / out-of-scope badge, cited sources with article, page, quoted passage
and link to the official text, right-to-left Arabic).

The first question in the interface took 11.4 s. Profiling one question:

| Stage | Before | Fix |
|---|---:|---|
| LiteLLM import + pipeline build | 5.4 s + 4.6 s on the first question | done at page load, with one tiny warm-up call to each provider; LiteLLM's bundled price list instead of a download |
| Question embedded for the scope check, then again for retrieval | 0.40 + 0.32 s | embedded once (cache shared by guardrails and retrieval) |
| LLM answer | 1.7-3.1 s | streamed (first words ~1 s); `reasoning_effort=minimal` |
| Old comparison indexes loaded at startup | 177 MB | deleted (results recorded above); 60 MB |

Result for the same first question: **2.4 s, first words after 1.7 s** (page load ~10 s,
once). `reasoning_effort=minimal` was checked on all 113 questions with the judge:

| | Default effort (Phase 4) | `minimal` |
|---|---:|---:|
| Faithfulness | 96.3% | 97.1% |
| Correctness | 94.6% | 94.5% |
| Relevance | 4.89 | 4.90 |
| Numeric fact recall | 85.3% | 82.1% |
| Abstention accuracy | 88.2% | 94.1% |
| False abstention | 4.2% | 5.2% |

Every difference is about one question, within run-to-run noise, so `minimal` is kept. Most
of the speed gain comes from the warm-up, streaming and single embedding; `minimal` adds a
smaller, network-dependent saving. The answer evaluation now records a failing question
(network error) instead of losing the whole run, and network retries last about a minute.
