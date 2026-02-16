## Legal Document Search Evaluation Toolkit

This project builds **synthetic Q&A evaluation datasets** from a corpus of legal documents (PDF, DOCX, XLSX, and more). It is designed end-to-end for evaluating retrieval/search/RAG pipelines: each row includes a **question** (`user_input`), a **grounded answer** (`reference`), the **supporting context** (`reference_contexts`), and optionally **hard negatives** for IR evaluation.

The toolkit covers the full lifecycle: **corpus ingestion** → **embedding** → **profiling** → **dataset generation** (single-hop + multi-hop) → **quality filtering** → **hard negative mining** → **search evaluation** → **dataset validation**.

---

## Project structure

```
# --- Corpus preparation ---
ingest_corpus.py                # Ingest documents into SQLite page store (PDF, DOCX, XLSX, PPTX, TXT, CSV, JSON)
embed_corpus.py                 # Compute page embeddings (multi-key parallelism, resume-safe, cloud + open-source models)
profile_corpus.py               # Generate per-file LLM profiles (multi-endpoint parallelism)

# --- Dataset generation ---
generate_synthetic_dataset.py   # Main pipeline: single-hop + multi-hop Q&A generation (world mode)
generate_single_hop.py          # Standalone parallel single-hop generator (no KG, multi-endpoint)

# --- Evaluation & validation ---
evaluate_search.py              # Evaluate embedding model retrieval quality (Recall@K, MRR, MRR@K, nDCG@K, MAP)
validate_dataset.py             # LLM-as-a-judge validation of generated datasets
run_mleb.py                     # Run Massive Legal Embedding Benchmark (MLEB) with MTEB

# --- Modules ---
modules/
  config.py                     # Constants and defaults
  utils.py                      # Shared helpers (hashing, JSON, paths, text)
  db.py                         # SQLite page store operations + multi-model embedding storage
  loaders.py                    # Format-specific page extractors (PDF, DOCX, XLSX, etc.)
  azure_doc_intel.py            # Azure Document Intelligence OCR client (batch, retry, per-page spans)
  transforms.py                 # RAGAS transform patches (SafeHeadlineSplitter, etc.)
  llm_setup.py                  # LLM/embedding setup (OpenAI / Azure OpenAI)
  profiles.py                   # PDF profile generation & formatting
  synthesizers.py               # Query synthesizers + distribution building
  single_hop.py                 # Reusable single-hop generation (no KG needed, parallel + checkpointed)
  hard_negatives.py             # Hard negative mining (BM25 + embedding + RRF + tiered LLM judge + SQLite loader)
  quality_filter.py             # LLM-as-a-judge quality filter (drops low-quality rows during generation)
  testset.py                    # KG building, testset gen/save, persona management, source mapping

# --- Data ---
output/                         # Generated datasets + evaluation results
processed/                      # Cached artifacts (KG, personas, SQLite store, profiles, checkpoints)
```

---

## Quick start

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies include `sentence-transformers`, `transformers`, and `mteb` for open-source embedding model support. OCR mode additionally requires `azure-ai-documentintelligence`.

### Configure credentials

- Copy `.env.example` → `.env`
- Configure **either** OpenAI **or** Azure OpenAI (auto-detected from env vars).
- For multi-endpoint Azure parallelism (used by `profile_corpus.py`, `generate_single_hop.py`, `generate_synthetic_dataset.py`, `embed_corpus.py`), add `AZURE_OPENAI_API_KEY_2` + `AZURE_OPENAI_ENDPOINT_2`, etc. (up to `_9`).
- For OCR-based PDF extraction (`--ocr`), set `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` and `AZURE_DOCUMENT_INTELLIGENCE_KEY`.

---

## Corpus preparation pipeline

Before generating a dataset, prepare the corpus:

### 1. Ingest documents into SQLite

```bash
python ingest_corpus.py --input-dir search-dataset
```

Extracts all documents (PDF, DOCX, XLSX, PPTX, TXT, CSV, JSON) into the SQLite page store (`processed/pdf_page_store.sqlite`). Each page becomes one row with `doc_content`, `rel_path`, `filename`, `page_number`, and metadata.

#### OCR mode (Azure Document Intelligence)

For scanned PDFs, image-heavy documents, or complex table layouts, use OCR-based extraction instead of pypdf's text-layer extraction:

```bash
# OCR all PDFs using Azure Document Intelligence
python ingest_corpus.py --input-dir search-dataset/ --ocr

# OCR into a separate DB for A/B comparison
python ingest_corpus.py --input-dir search-dataset/ \
    --db-path processed/pdf_page_store_ocr.sqlite --ocr --reprocess
```

The `--ocr` flag routes PDF extraction through Azure Document Intelligence's `prebuilt-layout` model, which returns per-page markdown. Non-PDF formats always use their standard extractors (unchanged).

How it works (`modules/azure_doc_intel.py`):
- **Batching**: PDFs larger than 30 pages are split into sub-PDFs using pypdf, processed in parallel (up to 6 concurrent Azure requests), then merged
- **Retry**: exponential backoff on HTTP 429 / transient errors (up to 5 retries)
- **Per-page span mapping**: the full markdown response is sliced back to per-page text using the page-span offsets returned by the API

Requires `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` and `AZURE_DOCUMENT_INTELLIGENCE_KEY` env vars (see `.env.example`). Pricing is ~$1.50 per 1,000 pages.

> **OCR vs pypdf benchmark**: On a corpus of ~128k PDF pages (primarily text-layer PDFs), OCR showed small but consistent improvements: recall@1 +2.4%, MAP +1.4%, MRR +1.0%, nDCG@10 +1.0%. Gains would likely be larger on scanned/image-heavy documents. Full results in `output/verified/OCR_vs_no_OCR_comparison_analysis.md`.

### 2. Compute page embeddings

```bash
python embed_corpus.py --embedding-model text-embedding-3-large
```

Computes and stores page embeddings for the specified model. Multi-key parallelism and resume-safe (picks up where it left off if interrupted).

#### Open-source / HuggingFace models

`embed_corpus.py` also supports local SentenceTransformer models via the `hf` provider:

```bash
# Single open-source model
python embed_corpus.py --provider hf --model bge-m3

# All registered open-source models sequentially
python embed_corpus.py --models all-hf

# Comma-separated list of models
python embed_corpus.py --models bge-m3,stella_en_400m_v5,jina-embeddings-v3
```

Built-in HuggingFace model aliases:

| Alias | HuggingFace ID |
|-------|----------------|
| `stella_en_400m_v5` | `dunzhang/stella_en_400M_v5` |
| `jina-embeddings-v3` | `jinaai/jina-embeddings-v3` |
| `snowflake-arctic-embed-m-v2.0` | `Snowflake/snowflake-arctic-embed-m-v2.0` |
| `bge-m3` | `BAAI/bge-m3` |

Any SentenceTransformer-compatible model can also be used by passing its HuggingFace ID directly (e.g., `--model BAAI/bge-large-en-v1.5`). Provider is auto-detected when the model name contains `/` or matches a known alias.

### 3. Generate per-file profiles

```bash
python profile_corpus.py
```

Generates LLM-based metadata per file (title, doc type, summary, topics, key entities, likely user intents). Profiles are stored in the SQLite store and reused by the query generation pipeline to make queries more realistic. Multi-endpoint Azure parallelism with retry/backoff.

---

## Dataset generation

The main entrypoint is `generate_synthetic_dataset.py`. It uses **world mode**, which generates:

1. **Single-hop queries** from the **full corpus** (no KG needed — fast, parallelised across Azure endpoints)
2. **Multi-hop queries** from **per-world KGs** (one KG per subfolder)

```bash
python generate_synthetic_dataset.py \
  --input-dir search-dataset \
  --multi-hop-worlds Claires "Law worlds/415" "Law worlds/416" \
  --single-hop-size 300 \
  --multi-hop-size 100 \
  --hard-negatives \
  --no-pdf-profiles \
  --output combined_dataset
```

How it works:
- **Single-hop**: randomly samples PDFs from the full SQLite store, picks the best page per file, generates a Q&A pair using the LLM. No KG needed. When multiple Azure endpoints are configured, generation runs in parallel (one LLM per thread).
- **Multi-hop**: for each world subfolder, collects its PDFs, builds (or loads cached) a per-world KG, generates multi-hop queries using only multi-hop RAGAS synthesizers.
- **Quality filtering**: by default (`--filter`), an LLM-as-a-judge pass drops queries that fail query quality or source answerability checks (see [Quality filtering](#quality-filtering) below).
- **Hard negatives**: mines from the full corpus via SQLite (BM25 + embedding retrieval, RRF merge, tiered LLM judge).
- **Output**: a combined CSV/JSON with `synthesizer_name` (`single_hop_direct` or `multi_hop_ragas`) and `world` column for multi-hop rows.

When `--multi-hop-worlds` is omitted, it defaults to `["."]` (the entire input directory as a single world).

### Incremental / resumable generation

All major pipeline stages are **checkpointed** and resume-safe:

- **Single-hop**: progress saved to `processed/progress/{output}__single_hop_progress.json` — completed queries are preserved and generation resumes from where it left off.
- **Multi-hop**: per-world checkpoints at `processed/progress/{output}__multi_hop_world__{world}.json` — completed worlds are skipped on restart.
- **Hard negatives**: checkpoint at `processed/progress/{output}__hard_negatives_progress.json`.

If a run is interrupted (Ctrl+C, crash, rate-limit exhaustion), simply re-run the same command — it picks up from the last checkpoint.

### Standalone single-hop generator

For maximum throughput on single-hop queries, use the dedicated parallel script:

```bash
python generate_single_hop.py \
  --num-queries 500 \
  --seed 42 \
  --output single_hop_dataset
```

Features: multi-endpoint Azure parallelism, exponential backoff, graceful Ctrl+C shutdown, resume-safe, built-in hard negative mining.

### Common flags

Input/output:
- `--input-dir PATH` — corpus root directory
- `--output NAME` — output file base name (saved to `output/`)
- `--output-formats csv json parquet`
- `--specific-folders folderA folderB` — only load PDFs from these subdirectories
- `--files path/to/one.pdf` — specific files to include
- `--max-pdfs N` — cap number of files
- `--no-recursive` — don't search subdirectories

LLM/provider:
- `--model gpt-4o-mini` — LLM model (or Azure deployment name)
- `--provider auto|openai|azure`

Query generation:
- `--testset-size N` — number of samples (default: 50)
- `--single-hop-size N` — single-hop queries from full corpus (default: `testset-size / 2`)
- `--multi-hop-size N` — total multi-hop queries split across worlds (default: `testset-size / 2`)
- `--multi-hop-worlds world1 world2 ...` — world subfolder paths relative to `--input-dir`; supports nested paths like `"Law worlds/415"` (default: `["."]`)
- `--seed N` — random seed for single-hop file sampling
- `--standalone-queries` / `--no-standalone-queries` — corpus-level standalone queries (default: enabled)
- `--corpus-size-hint N` — approximate corpus size for prompt context (default: 7000)
- `--query-llm-context "..."` — extra LLM guidance for query generation
- `--query-mix ...` — override synthesizer distribution (e.g., `single_hop_entities=0.5`)
- `--list-query-synthesizers` — print available synthesizer names
- `--num-personas N` — number of personas to generate from KG (default: 3)
- `--personas-path PATH` — path to a JSON/JSONL file with pre-defined personas

PDF profiles:
- `--pdf-profiles` / `--no-pdf-profiles` — per-PDF LLM profiles (default: enabled)
- `--pdf-profile-max-pages N` (default: 3)
- `--pdf-profile-max-chars-per-page N` (default: 2500)

Quality filtering:
- `--filter` / `--no-filter` — LLM quality filter (default: **enabled**)

Hard negatives:
- `--hard-negatives` — enable hard negative mining
- `--num-bm25-negatives N` (default: 10)
- `--num-embedding-negatives N` (default: 10)

Store/embedding:
- `--pdf-store-db PATH` — override SQLite store location
- `--pdf-store-embeddings` / `--no-pdf-store-embeddings` — compute and store page embeddings during sync (default: disabled)
- `--no-content-embeddings` — skip page_content embeddings in KG (summary only)

Caching:
- `--processed-dir processed` — where intermediate caches are stored
- `--no-cache` — disable cache reuse for KG/personas
- `--reprocess` — ignore caches and rebuild everything

---

## Quality filtering

When `--filter` is enabled (the default), the pipeline runs an **LLM-as-a-judge quality filter** after generation and before hard negative mining. This is handled by `modules/quality_filter.py`.

### What it checks

1. **Source validity** — rows referencing `"unknown"` sources are dropped immediately.
2. **Query Quality (QC)** — is the query standalone, specific, and suitable for an IR dataset? Queries that are vague, referential ("this case"), or unanswerable from a search perspective are removed.
3. **Source Answerability (SA)** — does the labeled positive page actually answer the query? The filter fetches the real page content from SQLite (not just `reference_contexts`) and checks faithfulness.

Rows that fail any check are dropped from the output. This is an integrated filter (removes rows) — in contrast, `validate_dataset.py` is a post-hoc report that does not modify the dataset.

### Output

When filtering is enabled:
- The filtered dataset is saved as `{output}_filtered.csv` (and other requested formats)
- Console logs report how many rows were dropped and why

---

## Pipeline steps

Step counts adjust dynamically based on enabled features and number of worlds.

**Typical run** (`--multi-hop-worlds Claires "Law worlds/415"` with `--hard-negatives` and `--filter`):

```
[Step  1/11] Collecting PDF paths
[Step  2/11] Setting up LLM & embeddings
[Step  3/11] Syncing SQLite PDF store
[Step  4/11] Loading documents from SQLite store
[Step  5/11] Generating single-hop queries (full corpus)
[Step  6/11] Multi-hop for world: Claires (1/2)
[Step  7/11] Multi-hop for world: Law worlds/415 (2/2)
[Step  8/11] Quality filtering
[Step  9/11] Mining hard negatives
[Step 10/11] Saving results
[Step 11/11] Done
```

---

## Hard negative mining

When `--hard-negatives` is enabled, the pipeline mines challenging negative examples for each query. Hard negatives are passages that are **similar** to the query but **do not** contain enough information to answer it.

### Strategy

1. **Candidate retrieval**: BM25 (lexical) + embedding (semantic) retrieval from the full corpus via SQLite (no KG needed)
2. **RRF merge**: Reciprocal Rank Fusion (`score = Σ 1/(k+rank)`) combines both ranked lists into a single ordering — candidates appearing in only one list still participate
3. **Proximity exclusion**: positive pages ± 2 adjacent pages are excluded (replaces the old whole-file exclusion, so other pages from the same document can still be selected)
4. **Near-duplicate filtering**: candidates with cosine similarity ≥ 0.90 to any positive page are dropped
5. **Tiered LLM judge**: each candidate is evaluated for relevance, answerability, and **topical similarity**:
   - **Tier 1** (ideal): relevant=yes, answerable=no, topical_similarity=high
   - **Tier 2** (acceptable): answerable=no, relevant=yes|uncertain, topical_similarity=high|medium
   - **Reject**: answerable=yes with valid evidence, or relevant=no
6. **Per-file cap**: max 2 negatives per source file (up from the previous 1)
7. **Selection**: tier 1 negatives are preferred; remaining slots filled from tier 2

Because the miner prioritizes correctness, some queries may produce **no** hard negatives (`[]`).

### Tunable parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_bm25_negatives` | 10 | BM25 hard negatives per query (CLI: `--num-bm25-negatives`) |
| `num_embedding_negatives` | 10 | Embedding hard negatives per query (CLI: `--num-embedding-negatives`) |
| `max_judge_calls_per_query` | 20 | Max LLM judge calls per query |
| `bm25_candidate_multiplier` | 15 | Multiplier for BM25 candidate pool size |
| `embedding_candidate_multiplier` | 15 | Multiplier for embedding candidate pool size |
| `near_duplicate_cosine_threshold` | 0.90 | Cosine similarity threshold for near-duplicate exclusion |
| `embedding_min_similarity` | 0.25 | Minimum similarity for embedding candidates |

The first two are exposed as CLI flags; the rest use config defaults from `modules/config.py`.

### Output

Hard negatives are saved as a `hard_negatives` column: a JSON-encoded list of `"<filename> (page N)"` strings.

---

## SQLite page store

All document pages are stored in a **single-table SQLite database** (`processed/pdf_page_store.sqlite`).

- **Row granularity**: 1 row per document page
- **Supported formats**: PDF, DOCX, XLSX, PPTX, TXT, CSV, JSON, XLS, DOC
- **Core fields**: `doc_content`, `rel_path`, `filename`, `file_type`, `page_number`, `content_chars`
- **Derived fields**:
  - `summary` — cheap extractive snippet (offline, deterministic)
  - `emb__{model}` — page embeddings per model (multi-model storage)
  - `pdf_profile_json` — per-file LLM profile
  - `ragas_headlines_json`, `ragas_summary` — cached RAGAS extractions (skip LLM calls on KG rebuild)

### Multi-model embedding storage

Multiple embedding models are stored side by side. Each model gets its own column pair:

```
emb__text_embedding_3_large       BLOB
emb_dims__text_embedding_3_large  INTEGER
```

Model names are sanitized (e.g., `text-embedding-3-large` → `text_embedding_3_large`). Columns are added dynamically via `ALTER TABLE`. Both cloud models (OpenAI, Azure, Bedrock) and open-source models (SentenceTransformer/HuggingFace) are stored in the same schema.

### Caching

Intermediate artifacts under `processed/`:
- `processed/kg/` — RAGAS Knowledge Graphs (gzipped JSON, keyed by document fingerprint + model config)
- `processed/personas/` — generated personas
- `processed/meta/` — metadata JSON per cached artifact
- `processed/progress/` — incremental generation checkpoints (single-hop, multi-hop, hard negatives)
- `processed/pdf_page_store.sqlite` — the main page store

KG caches use stable keys derived from absolute paths + file sizes + modification times. Use `--reprocess` to force a rebuild.

---

## Evaluating embedding models

`evaluate_search.py` measures retrieval quality using a generated dataset. It reuses pre-computed page embeddings from SQLite.

```bash
python evaluate_search.py \
  --dataset output/combined_dataset.csv \
  --embedding-model text-embedding-3-large \
  --top-k 1 5 10 20
```

Output is written to `output/eval_{dataset_stem}_{model}.json` by default (e.g., `eval_combined_dataset_text_embedding_3_large.json`). Override with `--output`.

### Metrics

| Metric | Description |
|--------|-------------|
| **Recall@K** | Did any correct source page appear in the top K results? |
| **MRR** | Mean Reciprocal Rank (unbounded) — 1/rank of the first correct page, averaged |
| **MRR@K** | Truncated MRR — same as MRR but 0 if the first positive is beyond rank K |
| **nDCG@K** | Normalised Discounted Cumulative Gain at K (binary relevance) |
| **MAP** | Mean Average Precision — precision at each relevant rank, averaged |
| **HN rank** | Where do hard negatives rank vs positives? |
| **HN outranks %** | How often does a hard negative appear before any positive? |

### Supported providers

| Provider | `--provider` | Env vars needed | Example model |
|----------|-------------|-----------------|---------------|
| OpenAI | `openai` | `OPENAI_API_KEY` | `text-embedding-3-large` |
| Azure OpenAI | `azure` | `AZURE_OPENAI_*` | `text-embedding-3-large` |
| AWS Bedrock | `bedrock` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_BEDROCK_REGION` | `eu.cohere.embed-v4:0` |

Bedrock uses a direct API wrapper (`_DirectBedrockCohereEmbedder`) that bypasses `langchain_aws` to avoid parameter compatibility issues with Cohere Embed v4. Pages with empty or whitespace-only content are automatically skipped during embedding.

If a model doesn't have embeddings in SQLite yet, the script auto-embeds all corpus pages (one-time, resume-safe).

> **Note**: open-source models can be embedded with `embed_corpus.py --provider hf` and stored in SQLite. However, `evaluate_search.py` currently only supports cloud providers (`openai`, `azure`, `bedrock`) for query-time embedding. To evaluate an open-source model, pre-embed the corpus with `embed_corpus.py` and use the stored embeddings.

---

## Validating datasets

`validate_dataset.py` uses an LLM-as-a-judge to evaluate each query:

1. **Query Quality** — Is the query appropriate for a law-focused IR dataset?
2. **Source Answerability** — Does the **labeled positive page(s)** truly answer the query? (fetches actual page content from SQLite rather than relying on `reference_contexts`, which may contain text from different pages)
3. **Hard Negative Quality** — Is each hard negative a true hard negative?

The validator handles both single-hop and multi-hop rows transparently:
- Strips `<N-hop>` prefixes from reference contexts before sending to the LLM judge
- Resolves source display from `source_file_with_page`, `source_files_with_pages_readable`, or `source_files_with_pages` (JSON) — whichever is populated

This is a **post-hoc reporting tool** — it does not modify the dataset. For integrated filtering that drops rows during generation, see [Quality filtering](#quality-filtering).

### Output

- **JSON report** — full results including a `by_synthesizer` breakdown (pass/fail counts per synthesizer type)
- **CSV report** — one row per evaluated query with a `synthesizer` column for filtering
- **Console summary** — includes a "BY SYNTHESIZER TYPE" section showing per-type query quality and source answerability rates

```bash
python validate_dataset.py \
  --dataset output/combined_dataset.csv \
  --output validation_report
```

---

## Outputs and schema

Output files are saved to `output/` in the formats you request (`--output-formats csv json parquet`).

### Core columns

| Column | Description |
|--------|-------------|
| `user_input` | The generated question |
| `reference` | The grounded answer |
| `reference_contexts` | Context strings used to ground the answer |
| `synthesizer_name` | `single_hop_direct` or `multi_hop_ragas` |
| `world` | World subfolder name (multi-hop rows only) |

### Source mapping columns

| Column | Description |
|--------|-------------|
| `source_files` | JSON list of filenames |
| `source_files_with_pages` | JSON list like `"file.pdf (page 5)"` |
| `page_numbers` | JSON list of 1-indexed page numbers |
| `source_files_readable` | Comma-separated string for quick inspection |
| `source_files_with_pages_readable` | Comma-separated string |

For **single-hop** queries, source mapping is exact (the source page is known at generation time). For **multi-hop** queries, source mapping is resolved by matching reference contexts against the KG's DOCUMENT nodes (which carry exact source + page metadata) using a 5-strategy matching pipeline:

1. **Exact match** — context text matches node `page_content` exactly
2. **Whitespace-normalised exact match** — collapses all whitespace to single spaces before comparison (handles RAGAS inserting extra spaces/newlines)
3. **Containment** — context is a substring of a node's text, or vice-versa; picks the best overlap
4. **Whitespace-normalised containment** — same as above but with normalised text
5. **Head-chunk fallback** — first 100 normalised characters of context found in a node

### Hard negative column

When `--hard-negatives` is enabled:
- `hard_negatives` — JSON list of `"<filename> (page N)"` strings

---

## How RAGAS works here

RAGAS generates the dataset in two phases:

### Phase A: Documents → Knowledge Graph

1. Documents are loaded from SQLite and converted to KG DOCUMENT nodes
2. `default_transforms(...)` applies LLM-based extractors + embedding steps:
   - **HeadlinesExtractor** → headline-based splitting into CHUNK nodes
   - **SummaryExtractor** → per-node summaries
   - **NERExtractor** + **ThemesExtractor** → entities and themes
   - **EmbeddingExtractor** → summary + page_content embeddings
   - **CosineSimilarityBuilder** → similarity edges between nodes
   - **OverlapScoreBuilder** → entity overlap edges
3. This repo patches the pipeline with `SafeHeadlineSplitter` (filters out docs without usable headlines) and adds page_content embeddings for 100% node coverage

### Phase B: KG → Scenarios → Q&A samples

1. **Personas**: clustered from document summaries via embeddings + LLM
2. **Synthesizers**: single-hop (entity/theme-driven) + multi-hop (abstract via similarity, specific via entity overlap)
3. **Scenarios**: sample nodes, themes, personas from the KG
4. **Generation**: LLM produces (question, answer) faithful to the selected context(s)

Phase A runs once per world (building a smaller, tractable KG), and only multi-hop synthesizers are used. Single-hop queries are generated separately from the full corpus without any KG.

---

## Troubleshooting

### "No API credentials found"

Set either `OPENAI_API_KEY`, or `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` (plus deployment names). See `.env.example`.

### "Documents appears to be too short (100 tokens or less)"

RAGAS's default transforms need longer docs. Provide longer source documents or implement custom transforms.

### "rank_bm25 is required for BM25 hard negative mining"

Install with `pip install rank_bm25`, or set `--num-bm25-negatives 0` to skip BM25 candidates.

### Slow or expensive runs

Cost drivers: LLM calls during KG transforms, persona generation, Q/A generation, quality filtering, page embedding.

Ways to reduce:
- Lower `--testset-size`, `--single-hop-size`, `--multi-hop-size`
- Use `--max-pdfs` or narrower `--specific-folders`
- Use a smaller/cheaper model
- Use `--no-pdf-profiles` if profiles are already computed via `profile_corpus.py`
- Use `--no-content-embeddings` to skip page_content embeddings (not recommended)
- Use `--no-filter` to skip the LLM quality filter (faster but lower dataset quality)
- Configure multiple Azure endpoints (`AZURE_OPENAI_API_KEY_2`, etc.) for parallel single-hop generation

### KG cache misses

The KG cache key includes absolute file paths + modification times. If files are touched, copied, or the path changes, the cache won't match. Use `--reprocess` to force a rebuild, or create symlinks in `processed/kg/` to map the new key to an existing cache file.
