# Final Evaluation Results

All results used in the thesis report. Covers 5 embedding models evaluated on 3 benchmarks:

- **Custom legal dataset** (in-domain) — held-out test split, ~170K-page corpus, 879 queries
- **ARCChallenge** (external, general-domain) — MTEB science QA, ~9,350 corpus docs, 1,172 queries
- **BarExamQA** (external, legal-domain) — MTEB US bar exam, ~116 corpus docs, 117 queries

---

## Models

| # | Label | Corpus model | Query model | Adapter |
|---|---|---|---|---|
| 1 | openai | text-embedding-3-large | text-embedding-3-large | none |
| 2 | openai+adapter | text-embedding-3-large | text-embedding-3-large + full-rank adapter | full-rank (9.4M params) |
| 3 | voyage-4-large | voyage-4-large | voyage-4-large | none |
| 4 | voyage-4 | voyage-4-large | voyage-4 | none |
| 5 | voyage-4-lite | voyage-4-large | voyage-4-lite | none |

---

## 1. Custom Legal Dataset (In-Domain, Test Split)

Evaluated on the held-out test split (879 queries, no query or document overlap with training data). Corpus: ~170K pages from the project's legal document collection.

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | MAP | nDCG@10 |
|---|---|---|---|---|---|---|---|
| **openai+adapter** | **0.5017** | **0.7918** | **0.8703** | **0.9170** | **0.6300** | **0.6236** | **0.6789** |
| voyage-4-large | 0.4642 | 0.7281 | 0.8123 | 0.8578 | 0.5838 | 0.5729 | 0.6263 |
| voyage-4 | 0.4323 | 0.7258 | 0.7907 | 0.8441 | 0.5601 | 0.5507 | 0.6033 |
| voyage-4-lite | 0.4323 | 0.7144 | 0.7816 | 0.8328 | 0.5571 | 0.5471 | 0.5985 |
| openai | 0.3049 | 0.6030 | 0.7088 | 0.7759 | 0.4361 | 0.4296 | 0.4891 |

**Adapter impact**: +18.98pp nDCG@10 over the OpenAI baseline (0.4891 ? 0.6789), surpassing even voyage-4-large.

---

## 2. ARCChallenge (External, General Domain)

Public MTEB benchmark — science QA retrieval, ~9,350 corpus documents. Tests whether models generalise beyond the legal training domain.

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | MAP | nDCG@10 |
|---|---|---|---|---|---|---|---|
| **voyage-4-large** | **0.2645** | **0.5341** | **0.6263** | **0.6920** | **0.3886** | **0.3886** | **0.4383** |
| voyage-4 | 0.1630 | 0.4198 | 0.5230 | 0.6152 | 0.2825 | 0.2825 | 0.3308 |
| voyage-4-lite | 0.1263 | 0.3584 | 0.4565 | 0.5435 | 0.2318 | 0.2318 | 0.2760 |
| openai | 0.1203 | 0.3166 | 0.4053 | 0.4966 | 0.2176 | 0.2176 | 0.2522 |
| openai+adapter | 0.1203 | 0.3131 | 0.4010 | 0.5000 | 0.2164 | 0.2164 | 0.2501 |

**Adapter impact**: -0.21pp nDCG@10 (0.2522 ? 0.2501). Adapter is neutral on out-of-domain data.

---

## 3. BarExamQA (External, Legal Domain)

Public MTEB benchmark — US bar exam legal provision retrieval, ~116 corpus documents. Tests partial domain transfer to a different legal dataset.

| Model | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | MAP | nDCG@10 |
|---|---|---|---|---|---|---|---|
| **voyage-4-large** | **0.5043** | **0.7607** | **0.8462** | 0.9060 | **0.6185** | **0.6185** | **0.6672** |
| voyage-4 | 0.4103 | 0.7180 | 0.8291 | **0.9231** | 0.5547 | 0.5547 | 0.6136 |
| voyage-4-lite | 0.3675 | 0.7094 | 0.8120 | 0.8803 | 0.5268 | 0.5268 | 0.5893 |
| openai+adapter | 0.3761 | 0.6410 | 0.7521 | 0.8462 | 0.4966 | 0.4966 | 0.5484 |
| openai | 0.3675 | 0.6667 | 0.7436 | 0.8889 | 0.4963 | 0.4963 | 0.5459 |

**Adapter impact**: +0.25pp nDCG@10 (0.5459 ? 0.5484). Small positive transfer on legal-domain data.

---

## Summary: nDCG@10 Across All Benchmarks

| Model | Custom Legal (in-domain) | ARCChallenge (general) | BarExamQA (legal) |
|---|---|---|---|
| **openai+adapter** | **0.6789** | 0.2501 | 0.5484 |
| openai | 0.4891 | 0.2522 | 0.5459 |
| voyage-4-large | 0.6263 | **0.4383** | **0.6672** |
| voyage-4 | 0.6033 | 0.3308 | 0.6136 |
| voyage-4-lite | 0.5985 | 0.2760 | 0.5893 |

### Adapter delta (openai+adapter vs openai baseline)

| Benchmark | nDCG@10 delta | Interpretation |
|---|---|---|
| Custom Legal (in-domain) | **+18.98pp** | Strong improvement on training domain |
| BarExamQA (legal, external) | +0.25pp | Small positive transfer to related legal domain |
| ARCChallenge (general, external) | -0.21pp | Neutral — no degradation on general domain |

---

## Files in this folder

| File | Description |
|---|---|
| `eval_test_text_embedding_3_large.json` | OpenAI baseline — custom legal test split |
| `eval_test_text_embedding_3_large_adapted_full_rank_r256.json` | OpenAI + adapter — custom legal test split |
| `eval_test_voyage_4_large_2048.json` | Voyage-4-large — custom legal test split |
| `eval_test_voyage_4_large_2048_query_voyage_4.json` | Voyage-4 (asymmetric) — custom legal test split |
| `eval_test_voyage_4_large_2048_query_voyage_4_lite.json` | Voyage-4-lite (asymmetric) — custom legal test split |
| `mteb_results.json` | All 5 models on ARCChallenge + BarExamQA (full MTEB metrics) |
| `embedding_model_comparison.html` | Interactive visualization — charts and tables for all 3 benchmarks (open in browser) |
