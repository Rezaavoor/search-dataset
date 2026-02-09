## Synthetic Q&A Dataset Generator (RAGAS)

This project generates a **synthetic Q&A testset** from your **PDF documents** using **RAGAS**. It is designed for building evaluation datasets for retrieval/search/RAG pipelines: each row includes a **question** (`user_input`), a **grounded answer** (`reference`), the **supporting context** (`reference_contexts`), and optionally **hard negatives** for IR evaluation.

The main entrypoint is `generate_synthetic_dataset.py`.

---

## Project structure

```
generate_synthetic_dataset.py   # Generate synthetic Q&A dataset from PDFs
evaluate_search.py              # Evaluate embedding model search quality
modules/
  config.py                     # Constants and defaults
  utils.py                      # Shared helpers (hashing, JSON, paths, text)
  db.py                         # SQLite PDF page store operations
  transforms.py                 # RAGAS transform patches (SafeHeadlineSplitter, etc.)
  llm_setup.py                  # LLM/embedding setup (OpenAI / Azure OpenAI)
  profiles.py                   # PDF profile generation & formatting
  synthesizers.py               # Query synthesizers + distribution building
  hard_negatives.py             # Hard negative mining (BM25, embedding, LLM judge)
  testset.py                    # KG building, testset gen/save, persona management
output/                         # Generated datasets + evaluation results
processed/                      # Cached intermediate artifacts (KG, personas, SQLite store)
```

---

## Quick start

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure credentials

- Copy `.env.example` → `.env`
- Configure **either** OpenAI **or** Azure OpenAI (the script auto-detects based on which variables are set).

### Run

```bash
python generate_synthetic_dataset.py \
  --input-dir . \
  --testset-size 50 \
  --output synthetic_dataset \
  --output-formats csv json
```

Output files are saved to the `output/` folder (e.g., `output/synthetic_dataset.csv`).

Common flags:
- `--specific-folders folderA folderB`
- `--files path/to/one.pdf path/to/another.pdf`
- `--no-recursive` (don't search subdirectories)
- `--max-pdfs N` (cap the number of PDF files; default: no limit)
- `--standalone-queries` / `--no-standalone-queries` (default: enabled; generate standalone, corpus-level search queries)
- `--corpus-size-hint N` (used only when `--standalone-queries` is enabled; default: 7000)
- `--query-llm-context "..."` (extra guidance appended to the default corpus guidance)
- `--pdf-profiles` / `--no-pdf-profiles` (default: enabled; generate per-PDF LLM profiles and inject them into query generation)
- `--pdf-profile-max-pages N` (default: 3; lower to reduce cost)
- `--pdf-profile-max-chars-per-page N` (default: 2500; lower to reduce cost)
- `--provider auto|openai|azure`
- `--model gpt-4o-mini` (or your Azure deployment name)
- `--processed-dir processed` (where intermediate caches are stored)
- `--pdf-store-db PATH` (override SQLite store location; default: `processed/pdf_page_store.sqlite`)
- `--no-cache` (disable cache reuse for KG/personas)
- `--reprocess` (ignore caches and rebuild everything)

### Hard negative mining (for IR evaluation)

```bash
python generate_synthetic_dataset.py \
  --input-dir . \
  --testset-size 50 \
  --hard-negatives \
  --num-bm25-negatives 5 \
  --num-embedding-negatives 5 \
  --output synthetic_dataset_with_negatives
```

Hard negatives are saved to a single `hard_negatives` column as a JSON list of
`"<filename>.pdf (page N)"` strings. Some queries may have **no** hard negatives (`[]`)
if the miner cannot find negatives reliably.

Hard negative flags:
- `--hard-negatives` (enable hard negative mining)
- `--num-bm25-negatives N` (BM25 candidate mining budget per query, default: 5)
- `--num-embedding-negatives N` (embedding candidate mining budget per query, default: 5)
- `--no-content-embeddings` (disable page_content embeddings, use summary_embedding only)

---

## Pipeline steps

The script runs a clear step-by-step pipeline with progress indicators:

```
[Step 1/9] Collecting PDF paths
[Step 2/9] Setting up LLM & embeddings
[Step 3/9] Syncing SQLite PDF store
[Step 4/9] Loading documents from SQLite store
[Step 5/9] Building PDF profiles
[Step 6/9] Building knowledge graph
[Step 7/9] Generating personas
[Step 8/9] Generating testset
[Step 9/9] Saving results
```

When hard negatives are enabled, a "Mining hard negatives" step is added before saving. The step count adjusts dynamically based on which optional features (PDF profiles, hard negatives) are enabled.

---

## SQLite PDF page store

All PDF pages are extracted and stored in a **single-table SQLite database** (`pdf_page_store`). This is always active — the store is automatically created and synced on every run.

- **Row granularity**: 1 row per **PDF page** (matches `PyPDFLoader`)
- **Stored fields** (from PDFs): `doc_content` + loader metadata (`source`, `page`, etc.)
- **Derived fields**:
  - `summary`: cheap extractive snippet (offline, deterministic)
  - `embedding_f32` + `emb__{model}`: page embeddings stored per model (see "Multi-model embedding storage" below). Computed with `--pdf-store-embeddings`. Reused by the KG transform pipeline for DOCUMENT-level nodes and by `evaluate_search.py` for retrieval evaluation.
  - `pdf_profile_json`: per-PDF profile (one per PDF; computed when `--pdf-profiles` is enabled)
  - `ragas_headlines_json`, `ragas_summary`: RAGAS LLM extractions cached for reuse across KG builds (skips LLM calls when cached)
  - `ragas_entities_json`, `ragas_themes_json`: RAGAS chunk-level extractions persisted after KG build for inspection (not loaded back — see caching notes below)

**Default DB path:** `processed/pdf_page_store.sqlite` (override with `--pdf-store-db`).

On each run, the store automatically:
- Inserts any **missing or stale** PDFs (based on file size + modification time)
- Backfills extractive summaries for pages that don't have them
- Optionally computes page embeddings (`--pdf-store-embeddings`)
- Generates PDF profiles as needed (`--pdf-profiles`)

Use `--reprocess` to force re-extraction of all PDFs into the store.

### Reusing cached artifacts

Intermediate artifacts are saved under `processed/` and reused on subsequent runs:
- `processed/kg/`: the processed RAGAS Knowledge Graph (after transforms)
- `processed/personas/`: generated personas
- `processed/meta/`: metadata JSON describing cache keys + inputs for each artifact
- `processed/pdf_page_store.sqlite`: SQLite store of extracted PDF pages + derived fields

This is useful when you want to regenerate testsets (e.g., different `--testset-size`) without re-running the expensive transform step.

### What is cached in SQLite vs the KG file

The SQLite store caches **page-level** (DOCUMENT node) extractions that map 1:1 to a page row:

| Extraction | SQLite columns | Reused on KG rebuild? |
|------------|----------------|----------------------|
| Headlines | `ragas_headlines_json` | Yes — skips LLM call |
| Summary | `ragas_summary` | Yes — skips LLM call |
| Page embedding | `emb__{model}` | Yes — skips embedding API call (DOCUMENT nodes only) |

**Chunk-level** data (entities, themes, chunk embeddings) is NOT loaded from SQLite because multiple chunks can exist per page and the per-page key would cause collisions. This data is written to SQLite for inspection but the **KG file cache** (`processed/kg/*.json.gz`) is what prevents recomputation on re-runs with the same PDFs.

---

## How RAGAS works here (what gets called, and why)

RAGAS generates the synthetic dataset in two big phases:

- **Phase A: Build a Knowledge Graph (KG)** from your documents via a **transform pipeline** (LLM-based extractors + embedding-based steps).
- **Phase B: Generate samples** by sampling nodes/relationships from the KG and using the **LLM** to write the final **(question, answer)** grounded in the chosen context(s).

This repo wires those phases together using:
- `ragas.testset.TestsetGenerator`
- `ragas.testset.transforms.default_transforms(...)` (plus patches described below)
- `testset.to_pandas()` for export

---

## Phase A: Documents → Knowledge Graph (Transforms)

### 1) Initialize LLM + embeddings and load documents

LLM and embedding models are set up once (Step 2) and reused for all subsequent operations including store sync, profile generation, KG building, and testset generation. PDF pages are loaded from the SQLite page store (auto-synced in Step 3). Each page becomes a LangChain `Document` with `metadata["source"]` (absolute path) and `metadata["page"]` (0-indexed).

### 2) LLM + embedding setup

`setup_llm_and_embeddings(...)` creates:
- an **LLM** (for summarization, NER/themes extraction, and finally Q/A generation)
- an **embedding model** (for summary embeddings + content embeddings + similarity edges)

Both are wrapped for RAGAS using:
- `LangchainLLMWrapper`
- `LangchainEmbeddingsWrapper`

Azure note:
- `AzureChatOpenAI` is instantiated with `temperature=1` because some Azure deployments only support that value.

### 3) RAGAS converts documents into graph nodes

Inside RAGAS, each input document becomes a node like:
- `Node(type=DOCUMENT, properties={"page_content": ..., "document_metadata": ...})`

### 4) Default transforms (chosen based on document length)

RAGAS's `default_transforms(...)` chooses one of two pipelines based on how many docs fall into token-length bins:
- \(101–500\) tokens
- \(501+\) tokens

If your docs are mostly \(\le 100\) tokens, RAGAS raises an error ("documents too short") and you'll need longer inputs or custom transforms.

### 5) What transforms are used (LLM vs embeddings)

RAGAS transforms are applied **in order**. Each transform either:
- adds **properties** to nodes (e.g., `summary`, `entities`, `themes`, `summary_embedding`, `page_content_embedding`), or
- adds **relationships** (e.g., `child`, `next`, `summary_similarity`, `content_similarity`, `entities_overlap`), or
- **filters/removes nodes** from the graph.

#### Long-doc pipeline (roughly "many docs are 501+ tokens")

Typical transform sequence:
- **LLM**: `HeadlinesExtractor` → adds `headlines` (used for splitting)
- **Filter**: `HeadlinesRequiredFilter` → removes documents without usable headlines
- **Splitter**: `HeadlineSplitter` → creates `CHUNK` nodes + `child`/`next` relationships
- **LLM**: `SummaryExtractor` → adds `summary` (usually on DOCUMENT nodes)
- **LLM**: `CustomNodeFilter` → removes low "question potential" chunks
- **Parallel step**:
  - **Embeddings**: `EmbeddingExtractor(embed_property_name="summary")` → adds `summary_embedding`
  - **LLM**: `ThemesExtractor` → adds `themes`
  - **LLM**: `NERExtractor` → adds `entities`
- **Parallel step**:
  - **Embeddings/math**: `CosineSimilarityBuilder(property_name="summary_embedding")` → adds `summary_similarity` edges
  - **String overlap**: `OverlapScoreBuilder(property_name="entities")` → adds `entities_overlap` edges + `overlapped_items`
- **Embeddings**: `EmbeddingExtractor(property_name="page_content_embedding", embed_property_name="page_content")` → adds `page_content_embedding`
- **Embeddings/math**: `CosineSimilarityBuilder(property_name="page_content_embedding")` → adds `content_similarity` edges

#### Medium-doc pipeline (roughly "many docs are 101–500 tokens")

Same idea, but it typically:
- skips headline-based splitting
- runs extractors over DOCUMENT nodes more directly
- still produces `summary_embedding` + `page_content_embedding` + similarity/overlap relationships

### 6) Why this repo patches the headline splitter

In practice, some documents won't yield `headlines` (or RAGAS may decide not to extract them for shorter docs).
The stock `HeadlineSplitter` raises if `headlines` is missing.

This repo replaces it with `SafeHeadlineSplitter` (in `modules/transforms.py`) and adds a headline-required filter:
- if `headlines` is missing/empty (or none of the extracted headlines match the text), the document is **filtered out** (no queries are generated from it)
- otherwise, standard headline splitting behavior is used

This makes the pipeline more robust on legal docs with inconsistent heading formatting.

### 7) Page content embeddings (improved KG connectivity)

By default, this repo adds **page_content embeddings** to ALL nodes (not just those with summaries). This provides:

| Benefit | Description |
|---------|-------------|
| **100% embedding coverage** | All nodes get `page_content_embedding`, not just the ~50% with summaries |
| **Richer similarity edges** | `content_similarity` edges connect ALL semantically related content |
| **Better multi-hop questions** | RAGAS can find related content across all documents and chunks |
| **Enables hard negative mining** | Full corpus coverage for finding confusing passages |

Without this improvement, only nodes with summaries participate in similarity-based relationships, which limits multi-hop question generation.

To disable this feature (use summary embeddings only):
```bash
python generate_synthetic_dataset.py --no-content-embeddings ...
```

---

## Phase B: Knowledge Graph → Scenarios → Synthetic Q/A samples

After transforms, the Knowledge Graph contains:
- nodes with `page_content` plus extracted properties like `summary`, `themes`, `entities`, `summary_embedding`, `page_content_embedding`
- relationships like:
  - `child` / `next` (document structure)
  - `summary_similarity` (embedding-based "these docs are about similar things")
  - `content_similarity` (embedding-based similarity on actual page content)
  - `entities_overlap` (NER/overlap-based "these chunks share entities")

RAGAS then generates test samples like this:

### 1) Personas (embeddings + LLM)

If no personas are provided, RAGAS:
- clusters document summaries using **`summary_embedding`**
- uses the **LLM** to generate a small list of personas from representative summaries

Personas influence query style and framing (e.g., "law clerk" vs "compliance analyst").

This repo exposes two knobs:
- `--num-personas N`: generate more personas (default: 3)
- `--personas-path personas.json`: supply your own personas (skips persona generation)

### 2) Pick query synthesizers ("query distribution")

By default, RAGAS uses up to three synthesizers (it may skip ones that don't fit your KG):
- **Single-hop specific**: one context chunk, entity/theme-driven
- **Multi-hop abstract**: multi-context queries using *theme* connections across similar docs
- **Multi-hop specific**: multi-context queries using *entity overlap* connections

### 3) Scenario generation (mostly KG sampling, some LLM calls)

Synthesizers create *scenarios* by selecting:
- nodes (contexts)
- themes/entities to focus on
- persona + query style + query length

Some synthesizers call the LLM to map personas to themes/entities (to make the persona choice meaningful).

### 4) Sample generation (LLM writes the final Q/A)

For each scenario, the LLM is prompted to produce:
- a **question** (`user_input`)
- an **answer** (`reference`) that is **faithful to the provided context only**

RAGAS stores the exact context strings used in:
- `reference_contexts` (list of one chunk for single-hop, multiple tagged chunks for multi-hop)

---

## Hard Negative Mining (for IR Evaluation)

When `--hard-negatives` is enabled, the script mines challenging negative examples for each query. Hard negatives are passages that:
- Are **similar** to the query (lexically or semantically)
- But **do not** contain enough information to answer the query

This is essential for training and evaluating retrieval systems, as random negatives are too easy to distinguish.

### Mining approaches

| Approach | Method | What it captures |
|----------|--------|------------------|
| **BM25** | Lexical similarity (term overlap) | Passages with similar vocabulary |
| **Embedding** | Semantic similarity (cosine distance) | Passages about similar topics |

In this repo, BM25/embedding retrieval is used to generate *candidates* at the **page** level.
Candidates are then filtered for reliability:
- Exclude any page that matches a positive `(filename,page)`
- Exclude any page from the same **PDF file** as the positive(s)
- Exclude near-duplicates via embedding cosine similarity against positive page(s)
- Validate with an LLM judge: keep only pages that are **relevant** but **not answerable**

Because the miner prioritizes correctness, some queries may produce **no** hard negatives (`[]`).

### Output columns

When hard negatives are enabled, the output includes:
- `hard_negatives`: JSON-encoded list of strings like `"<filename>.pdf (page N)"` (stored as a string in the exported CSV/JSON)

### Example output row

```json
{
  "user_input": "What are the penalties for late filing?",
  "reference": "Late filing penalties include...",
  "reference_contexts": ["<positive context from doc>"],
  "hard_negatives": "[\"filing_rules.pdf (page 12)\", \"another_doc.pdf (page 5)\"]",
  "source_files_with_pages": "[\"filing_rules.pdf (page 5)\"]",
  "source_files_with_pages_readable": "filing_rules.pdf (page 5)"
}
```

---

## Outputs and schema

Output files are saved to the `output/` directory in the formats you request:
- `--output-formats csv json parquet`

Typical core columns:
- `user_input`: the generated question
- `reference`: the generated grounded answer
- `reference_contexts`: list of context strings used to ground the answer (may be serialized as a list-like string in CSV depending on your pandas/RAGAS versions)

Hard negative columns (when `--hard-negatives` is used):
- `hard_negatives`: JSON-encoded list of `"<filename>.pdf (page N)"` strings (stored as a string in the exported CSV/JSON)

Additional columns may appear depending on synthesizer/version (e.g., persona/style metadata).

### Source mapping columns added by this repo

When saving, this repo tries to map each `reference_contexts` entry back to the original loaded docs, then adds:
- `source_files`: JSON-encoded list of filenames (stored as a string in the exported CSV/JSON)
- `source_files_with_pages`: JSON-encoded list like `"file.pdf (page 5)"` (stored as a string in the exported CSV/JSON)
- `page_numbers`: JSON-encoded list of page numbers (1-indexed; `null` for non-paginated docs; stored as a string in the exported CSV/JSON)
- `source_files_readable`, `source_files_with_pages_readable`: comma-separated strings for quick inspection

This mapping is **heuristic string matching** against loaded `Document.page_content`, so it's "best effort."

---

## Evaluating embedding model search quality

`evaluate_search.py` measures how well an embedding model retrieves the correct source pages for each query in a generated dataset. It reuses the pre-computed page embeddings stored in SQLite -- no need to re-embed the corpus.

### Quick start

```bash
python evaluate_search.py \
  --dataset output/dataset_with_negatives_v11.csv \
  --embedding-model text-embedding-3-large \
  --top-k 1 5 10 20
```

Results are saved to `output/eval_text_embedding_3_large.json` and a summary is printed to stdout.

### What it does

1. Loads the dataset CSV (queries + ground truth source pages + hard negatives)
2. Loads page embeddings from SQLite for the specified model
3. Embeds all queries using the same model (the only API call)
4. Ranks all corpus pages by cosine similarity to each query
5. Computes metrics: Recall@K, MRR, MAP, hard negative rank analysis

### Metrics

| Metric | Description |
|--------|-------------|
| **Recall@K** | Did any correct source page appear in the top K results? |
| **MRR** | Mean Reciprocal Rank -- 1/rank of the first correct page, averaged |
| **MAP** | Mean Average Precision -- precision at each relevant rank, averaged |
| **HN rank** | Where do hard negatives rank vs positives? |
| **HN outranks %** | How often does a hard negative appear before any positive? |

### Flags

- `--dataset` (required): Path to the generated dataset CSV
- `--embedding-model` (required): Model name or deployment ID (e.g., `text-embedding-3-large`, `eu.cohere.embed-v4:0`)
- `--top-k`: K values to evaluate (default: `1 5 10 20`)
- `--pdf-store-db`: Custom SQLite path (default: `processed/pdf_page_store.sqlite`)
- `--provider`: `auto` / `openai` / `azure` / `bedrock` (default: auto-detect from env vars)
- `--output`: Custom output path (default: `output/eval_{model}.json`)

### Supported embedding providers

| Provider | `--provider` | Env vars needed | Example model |
|----------|-------------|-----------------|---------------|
| OpenAI | `openai` | `OPENAI_API_KEY` | `text-embedding-3-large` |
| Azure OpenAI | `azure` | `AZURE_OPENAI_*` | `text-embedding-3-large` |
| AWS Bedrock | `bedrock` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_BEDROCK_REGION` | `eu.cohere.embed-v4:0` |

With `--provider auto`, the script detects the provider from environment variables (Azure > OpenAI > Bedrock).

### Evaluating a new model

When you specify a model that doesn't have embeddings in SQLite yet, the script automatically:
1. Adds model-specific columns to the SQLite table
2. Embeds all corpus pages using that model (one-time cost, with retry/backoff for throttling)
3. Stores embeddings after each batch (resume-safe -- if interrupted, re-run picks up where it left off)
4. Proceeds with evaluation

```bash
# OpenAI model
python evaluate_search.py \
  --dataset output/dataset_with_negatives_v11.csv \
  --embedding-model text-embedding-3-small

# AWS Bedrock / Cohere model
python evaluate_search.py \
  --dataset output/dataset_with_negatives_v11.csv \
  --embedding-model eu.cohere.embed-v4:0 \
  --provider bedrock

# Second run with same model: instant (embeddings loaded from SQLite)
python evaluate_search.py \
  --dataset output/dataset_with_negatives_v11.csv \
  --embedding-model eu.cohere.embed-v4:0 \
  --provider bedrock
```

---

## Multi-model embedding storage

The SQLite store supports **multiple embedding models** side by side. Each model gets its own column pair:

```
emb__text_embedding_3_large       BLOB
emb_dims__text_embedding_3_large  INTEGER
emb__text_embedding_3_small       BLOB
emb_dims__text_embedding_3_small  INTEGER
```

Model names are sanitized for column names (e.g., `text-embedding-3-large` becomes `text_embedding_3_large`). Columns are added dynamically via `ALTER TABLE` when a new model is first used.

The legacy columns (`embedding_f32`, `embedding_model`, `embedding_dims`) are preserved for backward compatibility. When `--pdf-store-embeddings` is used in the generation pipeline, embeddings are written to both legacy and model-specific columns.

**Important:** If you add new PDFs with `--pdf-store-embeddings`, the new pages get embeddings for the current model only. Other model columns remain NULL for those pages. The evaluation script handles this by auto-computing missing embeddings.

---

## Customizing the generation

### Change transform behavior

This script currently uses:
- `default_transforms(...)` from RAGAS, then patches (in `modules/transforms.py`):
  - `HeadlineSplitter` → `SafeHeadlineSplitter` (no fallback chunking)
  - adds a `HeadlinesExtractor` + filter step to **skip documents without usable headlines**
  - adds `EmbeddingExtractor` for `page_content` → `page_content_embedding`
  - adds `CosineSimilarityBuilder` for `page_content_embedding` → `content_similarity` edges

If you want full control, you can:
- build your own transform list (extractors/splitters/relationship builders)
- replace the `transforms = ...` block in `build_knowledge_graph(...)` (in `modules/testset.py`) and call `apply_transforms(kg, your_transforms, ...)`

### Change query mix

RAGAS accepts `query_distribution=...` (a list of `(synthesizer, probability)` pairs).

This script starts from the default RAGAS distribution, and when `--standalone-queries` is enabled (default) it patches the synthesizer prompts to encourage **standalone, corpus-level** search queries. Disable this behavior with `--no-standalone-queries` if you want unmodified RAGAS prompts.

This repo also exposes:
- `--query-mix ...`: override the synthesizer mix (names or `name=weight` pairs)
- `--list-query-synthesizers`: print available synthesizer names

### PDF profiles (improving query realism across many PDFs)

When `--pdf-profiles` is enabled (default), the script generates a **PDF-level profile** for each PDF (stored in the SQLite page store) and injects it into query generation as additional context. This helps the LLM write questions that sound like what a person would search for when looking across **thousands of documents**, not just "what's on this page."

Profiles are persisted in the SQLite DB (one profile per PDF) so reruns across different subsets can reuse them.

Profile inputs include:
- the PDF's **folder path** (e.g., `Claires/…`)
- the **filename stem** (e.g., `222_Notice_of_Appearance.._Filed_by_Wichita_County._(Lerew_Mollie)`)
- a short excerpt from the first few pages

Notes:
- The profiler metadata is used to improve **query framing only**; answers are still expected to be grounded in the provided context excerpt(s).
- If you want to reduce cost/latency, lower `--pdf-profile-max-pages` and/or `--pdf-profile-max-chars-per-page`, or disable with `--no-pdf-profiles`.

---

## Troubleshooting

### "No API credentials found"

Set either:
- `OPENAI_API_KEY`, or
- `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` (plus deployment names)

See `.env.example`.

### "Documents appears to be too short (100 tokens or less)"

RAGAS's default transform selection expects enough longer docs.
Options:
- provide longer source docs (or chunk them differently before feeding them), or
- implement custom transforms that work with short docs.

### "rank_bm25 is required for BM25 hard negative mining"

BM25 mining is optional. If you see this error and you don't need BM25 candidates, set:
`--num-bm25-negatives 0`.

To enable BM25 mining, install the BM25 library:
```bash
pip install rank_bm25
```

Or reinstall all requirements:
```bash
pip install -r requirements.txt
```

### Slow or expensive runs

Cost drivers:
- LLM calls during transforms (summary/NER/themes/filtering)
- persona generation
- Q/A generation for each sample
- Page content embedding (relatively cheap)

Ways to reduce:
- lower `--testset-size`
- reduce the number of loaded PDFs via `--max-pdfs` or narrower `--specific-folders`
- use a smaller/cheaper model
- use `--no-content-embeddings` to skip page_content embedding (not recommended)

### Cache invalidation

The caching system uses stable cache keys derived from:
- Source PDF fingerprints (absolute path + size + modification time)
- Provider/model/embedding id
- Pipeline configuration changes (e.g., toggling `--no-content-embeddings`)

To force a rebuild, use `--reprocess`.
