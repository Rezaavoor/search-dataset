## Synthetic Q&A Dataset Generator (RAGAS)

This project generates a **synthetic Q&A testset** from your **PDF documents** using **RAGAS**. It is designed for building evaluation datasets for retrieval/search/RAG pipelines: each row includes a **question** (`user_input`), a **grounded answer** (`reference`), the **supporting context** (`reference_contexts`), and optionally **hard negatives** for IR evaluation.

The main entrypoint is `generate_synthetic_dataset.py`.

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

Common flags:
- `--file-types pdf` (non-PDF entries are ignored)
- `--specific-folders folderA folderB`
- `--files path/to/one.pdf path/to/another.pdf`
- `--no-recursive` (don't search subdirectories)
- `--max-pdfs N` (cap the number of PDF files before loading pages; default: no limit)
- `--max-files N` (cap extracted *LangChain Documents*; for PDFs this usually means **pages**; default: no limit)
- `--standalone-queries` / `--no-standalone-queries` (default: enabled; generate standalone, corpus-level search queries)
- `--corpus-size-hint N` (used only when `--standalone-queries` is enabled; default: 7000)
- `--query-llm-context "..."` (extra guidance appended to the default corpus guidance)
- `--provider auto|openai|azure`
- `--model gpt-4o-mini` (or your Azure deployment name)
- `--processed-dir processed` (where intermediate caches are stored)
- `--no-cache` (disable cache reuse)
- `--reprocess` (ignore caches and rebuild them)

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
`"<filename>.pdf (page N)"` strings. (In the exported CSV/JSON, this is stored as a JSON-encoded string.) Some queries may have **no** hard negatives (`[]`)
if the miner cannot find negatives reliably.

Hard negative flags:
- `--hard-negatives` (enable hard negative mining)
- `--num-bm25-negatives N` (BM25 candidate mining budget per query, default: 5)
- `--num-embedding-negatives N` (embedding candidate mining budget per query, default: 5)
- `--no-content-embeddings` (disable page_content embeddings, use summary_embedding only)

### Reusing processed artifacts

By default the script saves intermediate artifacts to `processed/` and reuses them on subsequent runs:
- `processed/docs/`: extracted PDF pages (LangChain `Document`s)
- `processed/kg/`: the processed RAGAS Knowledge Graph (after transforms)
- `processed/personas/`: generated personas
- `processed/meta/`: metadata JSON describing cache keys + inputs for each artifact

This is useful when you want to regenerate testsets (e.g., different `--testset-size`) without re-running the expensive transform step.

---

## How RAGAS works here (what gets called, and why)

RAGAS generates the synthetic dataset in two big phases:

- **Phase A: Build a Knowledge Graph (KG)** from your documents via a **transform pipeline** (LLM-based extractors + embedding-based steps).
- **Phase B: Generate samples** by sampling nodes/relationships from the KG and using the **LLM** to write the final **(question, answer)** grounded in the chosen context(s).

This repo wires those phases together in `generate_synthetic_dataset.py` using:
- `ragas.testset.TestsetGenerator`
- `ragas.testset.transforms.default_transforms(...)` (plus patches described below)
- `testset.to_pandas()` for export

---

## Phase A: Documents → Knowledge Graph (Transforms)

### 1) Load documents (LangChain loaders)

`load_documents(...)` loads **PDF files only** into LangChain `Document` objects (non-PDF inputs are skipped).

Important detail:
- PDFs loaded via `PyPDFLoader` are typically **one `Document` per page** and include `metadata["page"]` (0-indexed). This affects `--max-files` and source mapping.
- Use `--max-pdfs` if you want to cap the number of PDF files instead of pages.

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
- **Embeddings**: `EmbeddingExtractor(property_name="page_content_embedding", embed_property_name="page_content")` → adds `page_content_embedding` (NEW)
- **Embeddings/math**: `CosineSimilarityBuilder(property_name="page_content_embedding")` → adds `content_similarity` edges (NEW)

#### Medium-doc pipeline (roughly "many docs are 101–500 tokens")

Same idea, but it typically:
- skips headline-based splitting
- runs extractors over DOCUMENT nodes more directly
- still produces `summary_embedding` + `page_content_embedding` + similarity/overlap relationships

### 6) Why this repo patches the headline splitter

In practice, some documents won't yield `headlines` (or RAGAS may decide not to extract them for shorter docs).
The stock `HeadlineSplitter` raises if `headlines` is missing.

This repo replaces it with `SafeHeadlineSplitter` and adds a headline-required filter:
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

`generate_synthetic_dataset.py` saves a `Testset` via `testset.to_pandas()` into the formats you request:
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
- `default_transforms(...)` from RAGAS, then patches:
  - `HeadlineSplitter` → `SafeHeadlineSplitter` (no fallback chunking)
  - adds a `HeadlinesExtractor` + filter step to **skip documents without usable headlines**
  - adds `EmbeddingExtractor` for `page_content` → `page_content_embedding`
  - adds `CosineSimilarityBuilder` for `page_content_embedding` → `content_similarity` edges

If you want full control, you can:
- build your own transform list (extractors/splitters/relationship builders)
- replace the `transforms = ...` block in `build_knowledge_graph(...)` and call `apply_transforms(kg, your_transforms, ...)`

### Change query mix

RAGAS accepts `query_distribution=...` (a list of `(synthesizer, probability)` pairs).

This script starts from the default RAGAS distribution, and when `--standalone-queries` is enabled (default) it patches the synthesizer prompts to encourage **standalone, corpus-level** search queries. Disable this behavior with `--no-standalone-queries` if you want unmodified RAGAS prompts.

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
- reduce the number of loaded PDFs/pages via `--max-pdfs`, `--max-files`, or narrower folders
- use a smaller/cheaper model
- use `--no-content-embeddings` to skip page_content embedding (not recommended)

### Cache invalidation

The caching system uses stable cache keys derived from:
- Source PDF fingerprints (absolute path + size + modification time)
- Provider/model/embedding id
- Pipeline configuration changes (e.g., toggling `--no-content-embeddings`)

To force a rebuild, use `--reprocess`.
