# Run 3 Dataset Analysis — Mar 12

This analysis covers `vision_validated_relaxed.csv` in `output/verified/run 3/`.

The dataset was produced by running **vision-based post-hoc filtering** (`vision_validate_dataset.py`, Gemini 2.5 Flash via Vertex AI) on the full 7,801-row unfiltered generation output (`output/full_10_000.csv`). It uses the **relaxed** filter threshold: `query_quality` in `{good, mediocre}` AND `source_answerability` in `{answerable, partial}`.

The corpus backing this dataset is the SQLite page store at `processed/pdf_page_store.sqlite`, containing **171,860 document pages** across **7,129 files**.

---

## Dataset composition


| Type                             | Count     |
| -------------------------------- | --------- |
| Single-hop (`single_hop_direct`) | 4,324     |
| Multi-hop (`multi_hop_ragas`)    | 2,373     |
| **Total**                        | **6,697** |


- **2,865** unique positive source filenames
- **2,781** unique positive `rel_path` values
- 0 errored queries during evaluation
- No queries have >50 positive pages (no boilerplate contamination)

### Comparison to previous runs


|               | Run 2 (text-only filter) | Run 3 (vision filter, relaxed) | Change    |
| ------------- | ------------------------ | ------------------------------ | --------- |
| Total queries | 2,450                    | **6,697**                      | **+173%** |
| Single-hop    | 1,875                    | 4,324                          | +131%     |
| Multi-hop     | 575                      | 2,373                          | +313%     |


The vision filter recovers roughly 4,250 queries that the old text-only LLM filter incorrectly dropped — primarily because it could not see tables, charts, or scanned page content.

---

## Vision filter breakdown


| Query quality | Source answerability | Count     |
| ------------- | -------------------- | --------- |
| good          | answerable           | 5,604     |
| good          | partial              | 912       |
| mediocre      | answerable           | 71        |
| mediocre      | partial              | 110       |
| **Total**     |                      | **6,697** |


- **5,675 rows** (84.7%) pass the strict threshold (`good` + `answerable`) — available as `vision_validated_strict.csv` in `output/`
- **1,022 rows** included only in relaxed output (partial answerability or mediocre quality)
- Gemini 2.5 Flash confidence: mean **0.981**, median **1.000**, min **0.800**

---

## Retrieval evaluation — `text-embedding-3-large`

Evaluated against the full 171,860-page corpus using cosine similarity.


| Metric          | Value      |
| --------------- | ---------- |
| **Recall@1**    | **0.2671** |
| Recall@5        | 0.4919     |
| Recall@10       | 0.5758     |
| Recall@20       | 0.6445     |
| MRR (unbounded) | 0.3716     |
| MRR@1           | 0.2671     |
| MRR@5           | 0.3518     |
| MRR@10          | 0.3632     |
| MRR@20          | 0.3680     |
| MAP             | 0.3527     |
| nDCG@1          | 0.2671     |
| nDCG@5          | 0.3699     |
| nDCG@10         | 0.3977     |
| nDCG@20         | 0.4156     |


### Hard negative analysis

- 4,379 queries (65.4%) include hard negatives
- Mean hard negative rank: **332.7** (hard negatives rank well below positives on average)
- Hard negative outranks positive: **41.6%** of queries with hard negatives — these are the genuinely challenging cases

---

## Query metadata

### Query style (single-hop, 4,324 rows)


| Style           | Count |
| --------------- | ----- |
| Perfect grammar | 1,450 |
| Poor grammar    | 1,446 |
| Web search like | 1,428 |


### Query length (single-hop)


| Length | Count |
| ------ | ----- |
| Long   | 1,467 |
| Medium | 1,431 |
| Short  | 1,426 |


### Persona distribution (single-hop)


| Persona              | Count |
| -------------------- | ----- |
| Corporate Counsel    | 748   |
| Legal Researcher     | 742   |
| Contract Analyst     | 724   |
| Compliance Officer   | 723   |
| Litigation Associate | 711   |
| Paralegal            | 676   |


---

## World distribution (multi-hop, 2,373 rows)


| World          | Count |
| -------------- | ----- |
| Claires        | 1,416 |
| Law worlds/415 | 207   |
| Law worlds/433 | 195   |
| Law worlds/423 | 187   |
| Law worlds/416 | 105   |
| Law worlds/419 | 77    |
| Law worlds/418 | 72    |
| Law worlds/425 | 60    |
| Law worlds/417 | 54    |


---

## Hard negatives

- **4,379 queries** include hard negatives (**65.4%** of the dataset)
- Average hard negatives per query (when present): **3.66**
- Average hard negatives per query (all rows): **2.39**

---

## Text length summary


| Field                | Min | Mean  | Median | Max   |
| -------------------- | --- | ----- | ------ | ----- |
| Query characters     | 27  | 149.9 | 142.0  | 456   |
| Reference characters | 56  | 658.8 | 634.0  | 3,825 |


---

## Corpus table


| File type | Pages       |
| --------- | ----------- |
| PDF       | 135,139     |
| XLSX      | 25,392      |
| DOCX      | 9,793       |
| PPTX      | 958         |
| CSV       | 459         |
| TXT       | 67          |
| JSON      | 37          |
| DOC       | 15          |
| **Total** | **171,860** |


All labeled positive sources in this dataset are PDFs.

---

## Files


| File                                                        | Description                                                                |
| ----------------------------------------------------------- | -------------------------------------------------------------------------- |
| `vision_validated_relaxed.csv`                              | Dataset — 6,697 rows, CSV (relaxed vision filter)                          |
| `vision_validated_relaxed.json`                             | Dataset — 6,697 rows, JSON records                                         |
| `eval_vision_validated_relaxed_text_embedding_3_large.json` | Retrieval evaluation — text-embedding-3-large, Recall@K / MRR / MAP / nDCG |


