## Synthetic Q&A Dataset Generator (RAGAS)

This project generates a **synthetic Q&A testset** from your **PDF documents** using **RAGAS**. It is designed for building evaluation datasets for retrieval/search/RAG pipelines: each row includes a **question** (`user_input`), a **grounded answer** (`reference`), the **supporting context** (`reference_contexts`), and optionally **hard negatives** for IR evaluation.

The main entrypoint is `generate_synthetic_dataset.py`.

---

## Project structure

```
generate_synthetic_dataset.py   # Main entrypoint (argparse + step-by-step orchestration)
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
output/                         # Generated CSV/JSON/Parquet datasets
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
[Step  1/10] Collecting PDF paths
[Step  2/10] Syncing SQLite PDF store
[Step  3/10] Loading documents from SQLite store
[Step  4/10] Setting up LLM & embeddings
[Step  5/10] Building PDF profiles
[Step  6/10] Building knowledge graph
[Step  7/10] Generating personas
[Step  8/10] Generating testset
[Step  9/10] Mining hard negatives
[Step 10/10] Saving results
```

The step count adjusts dynamically based on which optional features (PDF profiles, hard negatives) are enabled.

---

## SQLite PDF page store

All PDF pages are extracted and stored in a **single-table SQLite database** (`pdf_page_store`). This is always active — the store is automatically created and synced on every run.

- **Row granularity**: 1 row per **PDF page** (matches `PyPDFLoader`)
- **Stored fields** (from PDFs): `doc_content` + loader metadata (`source`, `page`, etc.)
- **Derived fields**:
  - `summary`: cheap extractive snippet (offline, deterministic)
  - `embedding_f32`: optional float32 BLOB, computed with `--pdf-store-embeddings` (stored for reuse; the RAGAS transform pipeline still computes its own embeddings as needed)
  - `pdf_profile_json`: per-PDF profile (one per PDF; computed when `--pdf-profiles` is enabled)
  - `ragas_headlines_json`, `ragas_summary`: RAGAS LLM extractions cached for reuse across runs

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

### 1) Load documents from SQLite store

PDF pages are loaded from the SQLite page store (auto-synced in Step 2). Each page becomes a LangChain `Document` with `metadata["source"]` (absolute path) and `metadata["page"]` (0-indexed).

### 2) Initialize LLM + embeddings

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
