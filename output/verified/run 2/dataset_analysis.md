# Run 2 Dataset Analysis - Mar 10

This analysis is for `full_10_000_filtered.csv` in `output/verified/run 2/`.

The filtered dataset contains **2,450 queries** grounded in the current SQLite corpus in `pdf_page_store`, which now contains **171,860 document pages** across **7,129 files**.

## Dataset composition

| Type | Count |
|------|-------|
| Single-hop | 1,875 |
| Multi-hop | 575 |
| **Total** | **2,450** |

- 1,530 unique positive source filenames
- 1,433 unique positive `rel_path` values
- All labeled positives in this dataset point to PDF sources

## Query metadata coverage

The single-hop rows carry explicit persona/style/length metadata; the 575 multi-hop rows are `unknown` for those fields.

### Query style

| Style | Count |
|-------|-------|
| Web search like | 682 |
| Perfect grammar | 665 |
| Poor grammar | 528 |
| Unknown | 575 |

### Query length label

| Length | Count |
|--------|-------|
| Short | 637 |
| Medium | 656 |
| Long | 582 |
| Unknown | 575 |

### Persona distribution

| Persona | Count |
|---------|-------|
| Corporate Counsel | 328 |
| Compliance Officer | 325 |
| Legal Researcher | 316 |
| Litigation Associate | 315 |
| Contract Analyst | 300 |
| Paralegal | 291 |
| Unknown | 575 |

## World distribution

| World | Count |
|-------|-------|
| None (single-hop) | 1,875 |
| Claires | 220 |
| Law worlds/415 | 81 |
| Law worlds/433 | 69 |
| Law worlds/423 | 68 |
| Law worlds/416 | 35 |
| Law worlds/419 | 35 |
| Law worlds/418 | 24 |
| Law worlds/425 | 24 |
| Law worlds/417 | 19 |

## Hard negatives

- 1,460 queries include hard negatives (**59.6%** of the dataset)
- Average hard negatives per query, when present: **3.15**
- No rows are missing a labeled source file
- No rows have an empty reference answer

## Text length summary

| Field | Min | Mean | Median | Max |
|-------|-----|------|--------|-----|
| Query characters | 21 | 144.3 | 135.0 | 427 |
| Reference characters | 68 | 659.8 | 633.5 | 3,213 |

## Current corpus table

Current `pdf_page_store` row counts:

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

This is a multi-format corpus overall, but the positive sources referenced by `full_10_000_filtered.csv` are all PDFs.

## Notes

- This file profiles the dataset itself, not retrieval performance.
- There is currently no matching evaluation JSON for `full_10_000_filtered.csv` in `output/verified/run 2/`, so Recall/MRR/MAP metrics are not reported here.

## Files

| File | Description |
|------|-------------|
| `full_10_000_filtered.json` | Dataset (2,450 rows, JSON records) |
| `full_10_000_filtered.csv` | Dataset (2,450 rows, CSV) |