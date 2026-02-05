## Synthetic Q&A Dataset Generator (RAGAS)

This project generates a **synthetic Q&A testset** from your **PDF documents** using **RAGAS**. It is designed for building evaluation datasets for retrieval/search/RAG pipelines: each row includes a **question** (`user_input`), a **grounded answer** (`reference`), and the **supporting context** (`reference_contexts`) pulled from the source docs.

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
- `--max-files 200` (note: this caps loaded *LangChain Documents*, often meaning PDF pages)
- `--provider auto|openai|azure`
- `--model gpt-4o-mini` (or your Azure deployment name)
- `--processed-dir processed` (where intermediate caches are stored)
- `--no-cache` (disable cache reuse)
- `--reprocess` (ignore caches and rebuild them)

### Reusing processed artifacts

By default the script saves intermediate artifacts to `processed/` and reuses them on subsequent runs:
- `processed/docs/`: extracted PDF pages (LangChain `Document`s)
- `processed/kg/`: the processed RAGAS Knowledge Graph (after transforms)
- `processed/personas/`: generated personas

This is useful when you want to regenerate testsets (e.g., different `--testset-size`) without re-running the expensive transform step.

---

## How RAGAS works here (what gets called, and why)

RAGAS generates the synthetic dataset in two big phases:

- **Phase A: Build a Knowledge Graph (KG)** from your documents via a **transform pipeline** (LLM-based extractors + embedding-based steps).
- **Phase B: Generate samples** by sampling nodes/relationships from the KG and using the **LLM** to write the final **(question, answer)** grounded in the chosen context(s).

This repo wires those phases together in `generate_synthetic_dataset.py` using:
- `ragas.testset.TestsetGenerator`
- `ragas.testset.transforms.default_transforms(...)` (plus a small patch described below)
- `testset.to_pandas()` for export

---

## Phase A: Documents → Knowledge Graph (Transforms)

### 1) Load documents (LangChain loaders)

`load_documents(...)` loads **PDF files only** into LangChain `Document` objects (non-PDF inputs are skipped).

Important detail:
- PDFs loaded via `PyPDFLoader` are typically **one `Document` per page** and include `metadata["page"]` (0-indexed). This affects `--max-files` and source mapping.

### 2) Initialize LLM + embeddings

`setup_llm_and_embeddings(...)` creates:
- an **LLM** (for summarization, NER/themes extraction, and finally Q/A generation)
- an **embedding model** (for summary embeddings + similarity edges)

Both are wrapped for RAGAS using:
- `LangchainLLMWrapper`
- `LangchainEmbeddingsWrapper`

Azure note:
- `AzureChatOpenAI` is instantiated with `temperature=1` because some Azure deployments only support that value.

### 3) RAGAS converts documents into graph nodes

Inside RAGAS, each input document becomes a node like:
- `Node(type=DOCUMENT, properties={"page_content": ..., "document_metadata": ...})`

### 4) Default transforms (chosen based on document length)

RAGAS’s `default_transforms(...)` chooses one of two pipelines based on how many docs fall into token-length bins:
- \(101–500\) tokens
- \(501+\) tokens

If your docs are mostly \(\le 100\) tokens, RAGAS raises an error (“documents too short”) and you’ll need longer inputs or custom transforms.

### 5) What transforms are used (LLM vs embeddings)

RAGAS transforms are applied **in order**. Each transform either:
- adds **properties** to nodes (e.g., `summary`, `entities`, `themes`, `summary_embedding`), or
- adds **relationships** (e.g., `child`, `next`, `summary_similarity`, `entities_overlap`), or
- **filters/removes nodes** from the graph.

#### Long-doc pipeline (roughly “many docs are 501+ tokens”)

Typical transform sequence:
- **LLM**: `HeadlinesExtractor` → adds `headlines` (used for splitting)
- **Splitter**: `HeadlineSplitter` → creates `CHUNK` nodes + `child`/`next` relationships
- **LLM**: `SummaryExtractor` → adds `summary` (usually on DOCUMENT nodes)
- **LLM**: `CustomNodeFilter` → removes low “question potential” chunks
- **Parallel step**:
  - **Embeddings**: `EmbeddingExtractor(embed_property_name="summary")` → adds `summary_embedding`
  - **LLM**: `ThemesExtractor` → adds `themes`
  - **LLM**: `NERExtractor` → adds `entities`
- **Parallel step**:
  - **Embeddings/math**: `CosineSimilarityBuilder(property_name="summary_embedding")` → adds `summary_similarity` edges
  - **String overlap**: `OverlapScoreBuilder(property_name="entities")` → adds `entities_overlap` edges + `overlapped_items`

#### Medium-doc pipeline (roughly “many docs are 101–500 tokens”)

Same idea, but it typically:
- skips headline-based splitting
- runs extractors over DOCUMENT nodes more directly
- still produces `summary_embedding` + similarity/overlap relationships

### 6) Why this repo patches the headline splitter

In practice, some documents won’t yield `headlines` (or RAGAS may decide not to extract them for shorter docs).
The stock `HeadlineSplitter` raises if `headlines` is missing.

This repo replaces it with `SafeHeadlineSplitter` and adds a headline-required filter:
- if `headlines` is missing/empty (or none of the extracted headlines match the text), the document is **filtered out** (no queries are generated from it)
- otherwise, standard headline splitting behavior is used

This makes the pipeline more robust on legal docs with inconsistent heading formatting.

---

## Phase B: Knowledge Graph → Scenarios → Synthetic Q/A samples

After transforms, the Knowledge Graph contains:
- nodes with `page_content` plus extracted properties like `summary`, `themes`, `entities`, `summary_embedding`
- relationships like:
  - `child` / `next` (document structure)
  - `summary_similarity` (embedding-based “these docs are about similar things”)
  - `entities_overlap` (NER/overlap-based “these chunks share entities”)

RAGAS then generates test samples like this:

### 1) Personas (embeddings + LLM)

If no personas are provided, RAGAS:
- clusters document summaries using **`summary_embedding`**
- uses the **LLM** to generate a small list of personas from representative summaries

Personas influence query style and framing (e.g., “law clerk” vs “compliance analyst”).

### 2) Pick query synthesizers (“query distribution”)

By default, RAGAS uses up to three synthesizers (it may skip ones that don’t fit your KG):
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

## Outputs and schema

`generate_synthetic_dataset.py` saves a `Testset` via `testset.to_pandas()` into the formats you request:
- `--output-formats csv json parquet`

Typical core columns:
- `user_input`: the generated question
- `reference`: the generated grounded answer
- `reference_contexts`: list of context strings used to ground the answer

Additional columns may appear depending on synthesizer/version (e.g., persona/style metadata).

### Source mapping columns added by this repo

When saving, this repo tries to map each `reference_contexts` entry back to the original loaded docs, then adds:
- `source_files`: JSON list of filenames
- `source_files_with_pages`: JSON list like `"file.pdf (page 5)"`
- `page_numbers`: JSON list of page numbers (1-indexed; `null` for non-paginated docs)
- `source_files_readable`, `source_files_with_pages_readable`: comma-separated strings for quick inspection

This mapping is **heuristic string matching** against loaded `Document.page_content`, so it’s “best effort.”

---

## Customizing the generation

### Change transform behavior

This script currently uses:
- `default_transforms(...)` from RAGAS, then patches:
  - `HeadlineSplitter` → `SafeHeadlineSplitter` (no fallback chunking)
  - adds a `HeadlinesExtractor` + filter step to **skip documents without usable headlines**

If you want full control, you can:
- build your own transform list (extractors/splitters/relationship builders)
- pass it into `generate_with_langchain_docs(..., transforms=your_transforms)`

### Change query mix

RAGAS accepts `query_distribution=...` (a list of `(synthesizer, probability)` pairs). The script currently relies on the default distribution.

---

## Troubleshooting

### “No API credentials found”

Set either:
- `OPENAI_API_KEY`, or
- `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` (plus deployment names)

See `.env.example`.

### “Documents appears to be too short (100 tokens or less)”

RAGAS’s default transform selection expects enough longer docs.
Options:
- provide longer source docs (or chunk them differently before feeding them), or
- implement custom transforms that work with short docs.

### Slow or expensive runs

Cost drivers:
- LLM calls during transforms (summary/NER/themes/filtering)
- persona generation
- Q/A generation for each sample

Ways to reduce:
- lower `--testset-size`
- reduce the number of loaded docs/pages via `--max-files` or narrower folders
- use a smaller/cheaper model

