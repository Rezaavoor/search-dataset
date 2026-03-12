# Run 3 Dataset Analysis — Mar 12

This analysis covers `vision_validated_relaxed.csv` in `output/verified/run 3/`.

The dataset was produced by:
1. **Vision-based filtering** (`vision_validate_dataset.py`, Gemini 2.5 Flash via Vertex AI) on the full 7,801-row generation output
2. **Rank curation** — removed 628 queries where `text-embedding-3-large` ranked the positive page beyond position 1,000 (corrupt text extraction + irretrievable pages)
3. **Structural curation** — removed 215 queries with structurally non-unique source pages: creditor matrix rows (144), service list/affidavit pages (38), and privilege log tables (33). These pages are indistinguishable within the corpus and provide no useful fine-tuning signal.

The corpus is `processed/pdf_page_store.sqlite`, containing **171,860 document pages** across **7,129 files**.

---

## Dataset composition

| Type | Count |
|------|-------|
| Single-hop (`single_hop_direct`) | 3,787 |
| Multi-hop (`multi_hop_ragas`) | 2,067 |
| **Total** | **5,854** |

### Curation history

| Stage | Rows | Removed | Reason |
|-------|------|---------|--------|
| Raw generation | 7,801 | — | |
| Vision filter (relaxed) | 6,697 | 1,104 | Gemini: bad query or non-answerable page |
| Rank > 1,000 removal | 6,069 | 628 | Corrupt extraction (219) + irretrievable boilerplate (409) |
| Structural non-unique removal | **5,854** | 215 | Creditor matrix (144) + service lists (38) + privilege logs (33) |

### Comparison to previous runs

| | Run 2 (text-only filter) | Run 3 (vision + curation) | Change |
|--|--------------------------|---------------------------|--------|
| Total queries | 2,450 | **5,854** | **+139%** |
| Single-hop | 1,875 | 3,787 | +102% |
| Multi-hop | 575 | 2,067 | +259% |

---

## Vision filter breakdown

| Query quality | Source answerability | Count |
|---------------|---------------------|-------|
| good | answerable | 5,029 |
| good | partial | 676 |
| mediocre | answerable | — |
| mediocre | partial | 149 |
| **Total** | | **5,854** |

- **5,029 rows** (85.9%) pass the strict threshold (`good` + `answerable`)
- Gemini 2.5 Flash confidence: mean **0.981**, median **1.000**, min **0.800**

---

## Retrieval evaluation — `text-embedding-3-large`

Evaluated against the full 171,860-page corpus using cosine similarity. Metrics recomputed programmatically after curation.

| Metric | Value |
|--------|-------|
| **Recall@1** | **0.3056** |
| Recall@5 | 0.5627 |
| Recall@10 | 0.6587 |
| Recall@20 | 0.7373 |
| MRR (unbounded) | 0.4249 |
| MRR@1 | 0.3056 |
| MRR@5 | 0.4025 |
| MRR@10 | 0.4155 |
| MRR@20 | 0.4210 |
| MAP | 0.4033 |
| nDCG@1 | 0.3056 |
| nDCG@5 | 0.4232 |
| nDCG@10 | 0.4550 |
| nDCG@20 | 0.4754 |

### Hard negative analysis

- 3,777 queries (64.5%) include hard negatives
- Mean hard negative rank: **~340** (hard negatives rank well below positives on average)
- Average hard negatives per query (when present): **3.50**

---

## Query metadata

### Query style (single-hop, 3,787 rows)

| Style | Count |
|-------|-------|
| Perfect grammar | ~1,280 |
| Poor grammar | ~1,260 |
| Web search like | ~1,247 |

### Persona distribution (single-hop)

| Persona | Count |
|---------|-------|
| Corporate Counsel | ~640 |
| Legal Researcher | ~635 |
| Contract Analyst | ~620 |
| Compliance Officer | ~615 |
| Litigation Associate | ~610 |
| Paralegal | ~587 |

---

## World distribution (multi-hop, 2,067 rows)

| World | Count |
|-------|-------|
| Claires | ~1,190 |
| Law worlds/415 | ~190 |
| Law worlds/433 | ~180 |
| Law worlds/423 | ~175 |
| Law worlds/416 | ~90 |
| Law worlds/419 | ~70 |
| Law worlds/418 | ~64 |
| Law worlds/425 | ~54 |
| Law worlds/417 | ~50 |

---

## Text length summary

| Field | Min | Mean | Median | Max |
|-------|-----|------|--------|-----|
| Query characters | 27 | ~149 | ~142 | 456 |
| Reference characters | 56 | ~658 | ~634 | 3,825 |

---

## Corpus table

| File type | Pages |
|-----------|-------|
| PDF | 135,139 |
| XLSX | 25,392 |
| DOCX | 9,793 |
| PPTX | 958 |
| CSV | 459 |
| TXT | 67 |
| JSON | 37 |
| DOC | 15 |
| **Total** | **171,860** |

All labeled positive sources in this dataset are PDFs.

---

## Curation notes

### Rank > 1,000 removal (628 queries)
- **Corrupt text extraction (219 queries, rank > 50,000)**: PDFs with garbled `doc_content` in SQLite. Gemini's vision filter passed them (reads rendered pages), but embeddings are meaningless. Files should be re-ingested via OCR.
- **Irretrievable boilerplate (409 queries, rank 1,001–50,000)**: Valid content but too similar to hundreds of sibling pages in the same corpus — court filing headers, creditor matrix headers, etc.

### Structural non-unique removal (215 queries)
Pages that are structurally indistinguishable within the corpus even when text is well-extracted. Training on these sends noise rather than signal:
- **Creditor matrix rows (144)**: pages of redacted creditor names/addresses; dozens of identical-looking pages within the same filing
- **Service list / affidavit pages (38)**: mailing list pages listing counsel addresses, formatted identically across filings
- **Privilege log tables (33)**: Bates-number tables; structurally identical across all 50+ privilege log documents in the Epic v. Apple case

The remaining **745 queries** with rank 101–1,000 (after removing the above) are kept as **hard but valid fine-tuning examples** — the model retrieves plausible but wrong pages, which is exactly the signal needed to improve retrieval quality.

---

## Files

| File | Description |
|------|-------------|
| `vision_validated_relaxed.csv` | Dataset — 5,854 rows, CSV |
| `vision_validated_relaxed.json` | Dataset — 5,854 rows, JSON records |
| `eval_vision_validated_relaxed_text_embedding_3_large.json` | Retrieval evaluation — text-embedding-3-large, Recall@K / MRR / MAP / nDCG |
