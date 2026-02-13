# Final Verified Dataset

**464 queries** evaluated against a **165,508-page** PDF corpus using `text-embedding-3-large`.

## Dataset composition

| Type | Count |
|------|-------|
| Single-hop | 395 |
| Multi-hop | 69 |
| **Total** | **464** |

- 431 unique positive source files
- 253 queries include hard negatives (avg 3.0 per query)

## Filtering applied

Starting from ~2,000 generated queries, the following were removed:

1. **Rank > 100** — 114 queries where the positive page was buried deep, indicating vague queries or mislabeled ground truth.
2. **Exact duplicates** — 3 duplicate pairs with conflicting positives.
3. **Severe general queries** — 24 queries with recall@20=0 where the top-20 was dominated by sibling documents from the same file family.
4. **Moderate general queries** — 28 queries at rank > 5 where 10+ of the top-20 results came from the same file family as the positive.
5. **LLM quality filter** — applied before the above steps via the generation pipeline.

## Evaluation results (`text-embedding-3-large`)

| Metric | Value |
|--------|-------|
| recall@1 | 0.457 |
| recall@5 | 0.825 |
| recall@10 | 0.884 |
| recall@20 | 0.922 |
| MRR | 0.614 |
| MAP | 0.606 |
| NDCG@10 | 0.669 |

## Rank distribution

| Range | Queries | % |
|-------|---------|---|
| 1 | 212 | 45.7% |
| 2–5 | 171 | 36.8% |
| 6–20 | 45 | 9.7% |
| 21–100 | 36 | 7.8% |

Median rank: **2** · P95: **33** · Max: **100**

## Query length vs. performance

Longer, more specific queries retrieve better:

| Length | Count | Avg rank | recall@1 | recall@10 |
|--------|-------|----------|----------|-----------|
| Short (<80 chars) | 49 | 7.5 | 0.327 | 0.837 |
| Medium (80–200) | 321 | 6.6 | 0.461 | 0.875 |
| Long (>200) | 94 | 3.6 | 0.511 | 0.936 |

## Hard negative analysis

- 54.5% of queries have hard negatives
- In 19.8% of those, a hard negative outranks the positive
- Mean hard negative rank: 99.1

## Files

| File | Description |
|------|-------------|
| `final_dataset.json` | Dataset (464 rows, JSON records) |
| `final_dataset.csv` | Dataset (464 rows, CSV) |
| `eval_final_dataset_text_embedding_3_large.json` | Full per-query evaluation results |
