# Embedding Adapter Training

A self-contained pipeline for training a **query-only linear embedding adapter** that improves retrieval quality without re-embedding the document corpus. The adapter learns a linear transformation applied only to query embeddings at search time; document embeddings in the SQLite store are never touched.

This follows the methodology from the [Chroma Research technical report on Embedding Adapters](https://research.trychroma.com/embedding-adapters) (Sanjeev & Troynikov, 2024), which demonstrated up to 70% nDCG improvement from as few as 1,500 labeled query–document pairs.

---

## Prerequisites

Before running anything here, the main pipeline must have produced:

- `processed/pdf_page_store.sqlite` — corpus pages with pre-computed `text-embedding-3-large` embeddings (run `embed_corpus.py`)
- `output/verified/run 3/vision_validated_relaxed.csv` — the vision-validated evaluation dataset (run `vision_validate_dataset.py`)

---

## Files

```
adapter/
  config.py       # All hyperparameters — the only file you need to edit
  preprocess.py   # Step 1: build triplets + cache query embeddings
  train.py        # Step 2: split, train, evaluate
  model.py        # Adapter model classes + checkpoint utilities
  utils.py        # Shared helpers (parse_page_ref, fmt_page_ref)
  data/           # Runtime output (gitignored)
    query_embeddings.npy    # Cached query embeddings (float32, N×3072)
    query_texts.json        # Query strings index-aligned with embeddings
    triplets.csv            # Triplet metadata (query_idx, pos_ref, neg_ref, ...)
    best_model.pt           # Best checkpoint (self-describing)
    training_log.jsonl      # Per-epoch metrics + final test results
```

---

## Quickstart

```bash
# from the project root
python adapter/preprocess.py   # ~30s + <$0.10 in OpenAI API calls
python adapter/train.py        # ~30s to load SQLite, then training
```

`preprocess.py` is idempotent: re-running it skips both the embedding API call and triplet generation if the output files already exist. Pass `--recompute` to force a full rebuild.

---

## Step 1 — Preprocessing (`preprocess.py`)

Reads the dataset CSV, builds triplet training records, and caches query embeddings.

**Triplet construction** for each query:

- **Soft negatives** (every query): `N_SOFT_NEGS` random pages sampled from the full ~165K-page SQLite corpus, excluding the query's positive pages and its hard negatives. Soft negatives are sampled independently per positive so multi-hop queries (with 2–3 positive pages) get diverse negatives for each anchor.
- **Hard negatives** (queries with `hard_negatives` column populated): one triplet per hard negative, cross-produced with each positive page. Queries with zero hard negatives receive only soft-negative triplets.

This yields approximately **~50K triplets** total (6,697 queries × 5 soft + 4,379 queries with hard negs × avg 3.7 hard).

**Query embedding** uses the same OpenAI / Azure OpenAI credentials as the rest of the project (auto-detected from `.env`). Progress is checkpointed after every API batch; if interrupted, re-running resumes from where it left off.

**Output** written to `adapter/data/`:

| File | Description |
|------|-------------|
| `query_embeddings.npy` | float32 array, shape `(N_queries, 3072)` |
| `query_texts.json` | query strings, index-aligned with the array above |
| `triplets.csv` | `query_idx, pos_ref, neg_ref, neg_type, source_file` |

---

## Step 2 — Training (`train.py`)

Loads the preprocessed data, splits it, trains the adapter, and evaluates on the held-out test set.

### File-level train / val / test split

Triplets are split **70 / 15 / 15 at the source-file level**: all queries whose positive page comes from the same source file are placed entirely in one split. This prevents leakage from same-document queries appearing in both train and test.

### Training objective

The adapter is trained with **cosine-based triplet loss**. All three embeddings (adapted query, positive, negative) are L2-normalised before `nn.TripletMarginLoss(p=2)` so the loss operates in cosine space, consistent with cosine-similarity retrieval at inference time.

```
L = max(0, sim(q̂, d⁻) − sim(q̂, d⁺) + α)
```

where `q̂ = adapter(q)` and `α` is the margin hyperparameter.

### Early stopping

Val triplet loss is computed after every epoch. The best checkpoint is saved to `adapter/data/best_model.pt` whenever val loss improves. Training stops after `EARLY_STOPPING_PATIENCE` consecutive non-improving epochs. The test evaluation reloads the best checkpoint (not the final epoch state).

### Test evaluation

After training, the best model is evaluated against the full ~165K-page corpus on held-out test queries. Both a **baseline** (no adapter) and the **adapted** model are evaluated, and a comparison table is printed:

```
Metric         Baseline    Adapted      Delta       Δ%
────────────────────────────────────────────────────────
recall@1         0.4210     0.4720    +0.0510    +12.1%
recall@5         0.6830     0.7150    +0.0320     +4.7%
recall@10        0.7390     0.7680    +0.0290     +3.9%
recall@20        0.7920     0.8110    +0.0190     +2.4%
mrr              0.5140     0.5620    +0.0480     +9.3%
map              0.5140     0.5620    +0.0480     +9.3%
ndcg@10          0.5580     0.6020    +0.0440     +7.9%
```

Metrics: **Recall@K**, **MRR** (unbounded), **MAP**, **nDCG@10**.

---

## Configuration (`config.py`)

All hyperparameters live in a single file. Change values here; no other file needs editing.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `ADAPTER_TYPE` | `"low_rank"` | `"low_rank"` or `"full_rank"` |
| `LOW_RANK_DIM` | `128` | Rank `r` for low-rank adapter; ignored for full-rank |
| `LEARNING_RATE` | `3e-3` | Chroma's most frequently best LR |
| `MARGIN` | `0.3` | Triplet loss margin α |
| `BATCH_SIZE` | `256` | |
| `NUM_EPOCHS` | `10` | |
| `WEIGHT_DECAY` | `1e-4` | AdamW weight decay |
| `EARLY_STOPPING_PATIENCE` | `3` | Epochs without val improvement before stopping |
| `GRAD_CLIP_NORM` | `1.0` | Max gradient norm; set to `None` to disable |
| `N_SOFT_NEGS` | `5` | Soft negatives per (query, positive) pair |
| `RANDOM_SEED` | `42` | Controls split + soft-neg sampling |
| `TRAIN_RATIO` | `0.70` | File-level split |
| `VAL_RATIO` | `0.15` | |
| `TEST_RATIO` | `0.15` | |
| `TOP_K_EVAL` | `[1,5,10,20]` | Recall@K values |
| `DEVICE` | `"auto"` | Auto-selects CUDA → MPS → CPU |

---

## Adapter architectures (`model.py`)

### `LowRankAdapter` (default, recommended)

```
q_out = q + up(down(q))
```

- `down`: d→r linear (no bias, Kaiming init)
- `up`: r→d linear (bias; **weight initialised to zeros**)

Because `up.weight = 0` at initialisation, the adapter is an exact identity and learns a rank-r correction — same principle as LoRA. Residual connection prevents degradation if the adapter under-trains.

| Rank r | Parameters | Suggested minimum data |
|--------|-----------|------------------------|
| 64 | ~397K | ~300 queries |
| **128** | **~789K** | **~500 queries** ← default |
| 256 | ~1.57M | ~1,000 queries |
| full | ~9.4M | ~5,000 queries |

### `FullRankAdapter`

`nn.Linear(d, d)` with weight initialised to the identity matrix. Full expressive power (~9.4M params at d=3072), but risks overfitting on smaller datasets. Use if you have many training queries or as part of a sweep.

### Checkpoint format

Both adapters are saved as self-describing checkpoints:

```python
{"state_dict": ..., "adapter_type": "low_rank", "low_rank_dim": 128, "emb_dim": 3072}
```

Load without needing the original config:

```python
from adapter.model import load_adapter
adapter = load_adapter("adapter/data/best_model.pt")
```

---

## Inference

At query time, apply the adapter before your normal cosine-similarity search:

```python
import numpy as np
import torch
from adapter.model import load_adapter

adapter = load_adapter("adapter/data/best_model.pt").eval()

def adapt_query(query_embedding: np.ndarray) -> np.ndarray:
    """Apply adapter + L2-normalise. Input: raw embedding from OpenAI API."""
    q = torch.from_numpy(query_embedding).unsqueeze(0)
    with torch.no_grad():
        adapted = adapter(q).squeeze(0).numpy()
    norm = np.linalg.norm(adapted)
    return adapted / norm if norm > 0 else adapted
```

Document embeddings in the SQLite store remain unchanged.

---

## Hyperparameter sweep suggestions

Chroma Research's ablation results suggest starting with:

| Hyperparameter | Values to try |
|----------------|---------------|
| `LEARNING_RATE` | `1e-4, 3e-4, 1e-3, 3e-3, 1e-2` |
| `MARGIN` | `0.3, 1.0, 3.0` |
| `LOW_RANK_DIM` | `64, 128, 256, full_rank` |
| `N_SOFT_NEGS` | `3, 5, 10` |
| `BATCH_SIZE` | `64, 128, 256` |

Each sweep run only requires re-running `train.py` (no need to re-run `preprocess.py` unless `N_SOFT_NEGS` changes).

---

## Output files

| File | Created by | Description |
|------|-----------|-------------|
| `adapter/data/query_embeddings.npy` | `preprocess.py` | Cached query embeddings |
| `adapter/data/triplets.csv` | `preprocess.py` | Triplet metadata |
| `adapter/data/query_texts.json` | `preprocess.py` | Query strings for debugging |
| `adapter/data/best_model.pt` | `train.py` | Best checkpoint by val loss |
| `adapter/data/training_log.jsonl` | `train.py` | Per-epoch losses + final test metrics |

---

## References

- Sanjeev & Troynikov (2024). [Embedding Adapters](https://research.trychroma.com/embedding-adapters). Chroma Technical Report.
- Chroma GitHub: [suvansh/ChromaAdaptEmbed](https://github.com/suvansh/ChromaAdaptEmbed)
- Vejendla (2025). Drift-Adapter: Near Zero-Downtime Embedding Model Upgrades. EMNLP 2025.
