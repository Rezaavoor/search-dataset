#!/usr/bin/env python3
"""Train a query-only embedding adapter using cosine-based triplet loss.

Loads preprocessed data from adapter/data/, performs a file-level 70/15/15
train/val/test split, trains the adapter with early stopping on val loss,
then evaluates against the full corpus reporting Recall@K, MRR, MAP, nDCG@10.

Memory note: loading all ~165K page embeddings requires ~2 GB RAM.

Usage:
    python adapter/train.py
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_ADAPTER_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _ADAPTER_DIR.parent
sys.path.insert(0, str(_ADAPTER_DIR))

import config as cfg
from model import build_adapter, load_adapter, save_adapter
from utils import parse_page_ref


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if cfg.DEVICE != "auto":
        return torch.device(cfg.DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Load corpus embeddings into a single contiguous matrix
# (memory-efficient: one ~2 GB array instead of 165K separate arrays)
# ---------------------------------------------------------------------------

def _load_corpus(db_path: Path) -> Tuple[List[Tuple[str, int]], Dict[Tuple[str, int], int], np.ndarray]:
    """Load all page embeddings from SQLite into a single float32 matrix.

    Returns:
        corpus_keys    list of (filename, page_number) — row order of corpus_matrix
        key_to_idx     dict mapping (filename, page_number) -> row index
        corpus_matrix  np.ndarray shape (N_corpus, EMB_DIM), float32
    """
    print(f"  Loading page embeddings from SQLite  (~2 GB, please wait)...")
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT filename, page_number, "
        "emb__text_embedding_3_large, emb_dims__text_embedding_3_large "
        "FROM pdf_page_store "
        "WHERE emb__text_embedding_3_large IS NOT NULL"
    ).fetchall()
    conn.close()

    corpus_keys: List[Tuple[str, int]] = []
    vecs: List[np.ndarray] = []
    for filename, page_num, blob, dims in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        if dims and len(vec) == int(dims):
            corpus_keys.append((str(filename), int(page_num)))
            vecs.append(vec)

    corpus_matrix = np.stack(vecs).astype(np.float32)  # (N_corpus, D)
    key_to_idx = {k: i for i, k in enumerate(corpus_keys)}
    print(f"  Loaded {len(corpus_keys):,} page embeddings  shape={corpus_matrix.shape}")
    return corpus_keys, key_to_idx, corpus_matrix


# ---------------------------------------------------------------------------
# File-level train / val / test split
# ---------------------------------------------------------------------------

def _file_level_split(
    triplets_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split triplets 70/15/15 at the source-file level.

    All triplets whose source_file maps to the same split bucket stay together,
    preventing leakage from same-document queries.
    """
    unique_files = sorted(triplets_df["source_file"].unique())
    rng = np.random.RandomState(cfg.RANDOM_SEED)
    shuffled = np.array(unique_files, dtype=object)
    rng.shuffle(shuffled)

    n_train = int(len(shuffled) * cfg.TRAIN_RATIO)
    n_val   = int(len(shuffled) * cfg.VAL_RATIO)

    train_files = set(shuffled[:n_train])
    val_files   = set(shuffled[n_train: n_train + n_val])
    test_files  = set(shuffled[n_train + n_val:])

    train_df = triplets_df[triplets_df["source_file"].isin(train_files)].reset_index(drop=True)
    val_df   = triplets_df[triplets_df["source_file"].isin(val_files)].reset_index(drop=True)
    test_df  = triplets_df[triplets_df["source_file"].isin(test_files)].reset_index(drop=True)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TripletDataset(Dataset):
    """Resolves triplet CSV records to embedding vectors.

    Stores index triples (q_idx, pos_corpus_idx, neg_corpus_idx) so that
    the underlying arrays are shared, not duplicated.
    """

    def __init__(
        self,
        triplets_df: pd.DataFrame,
        query_embeddings: np.ndarray,
        key_to_idx: Dict[Tuple[str, int], int],
    ) -> None:
        self.query_embeddings = query_embeddings
        self.key_to_idx = key_to_idx
        self.records: List[Tuple[int, int, int]] = []
        skipped = 0

        for _, row in triplets_df.iterrows():
            q_idx   = int(row["query_idx"])
            pos_key = parse_page_ref(str(row["pos_ref"]))
            neg_key = parse_page_ref(str(row["neg_ref"]))

            if pos_key is None or neg_key is None:
                skipped += 1
                continue
            if pos_key not in key_to_idx or neg_key not in key_to_idx:
                skipped += 1
                continue

            self.records.append((q_idx, key_to_idx[pos_key], key_to_idx[neg_key]))

        if skipped:
            print(f"    Skipped {skipped:,} triplets (page not found in corpus lookup)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # corpus_matrix is passed at evaluation time via the closure; here we
        # return indices so the DataLoader can batch them efficiently.
        # (We actually store numpy refs — see note in train() about corpus_matrix.)
        q_idx, pos_idx, neg_idx = self.records[idx]
        return (
            torch.from_numpy(self.query_embeddings[q_idx].copy()),
            torch.tensor(pos_idx, dtype=torch.long),
            torch.tensor(neg_idx, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train(
    adapter: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    corpus_matrix: np.ndarray,
    device: torch.device,
    log_path: Path,
    model_path: Path,
) -> None:
    """Train the adapter with early stopping on validation triplet loss."""
    corpus_t = torch.from_numpy(corpus_matrix).to(device)  # (N, D) on device

    adapter   = adapter.to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    loss_fn = nn.TripletMarginLoss(margin=cfg.MARGIN, p=2, reduction="mean")

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        # ---- Train --------------------------------------------------------
        adapter.train()
        total_train_loss = 0.0
        n_train_batches  = 0

        for q_batch, pos_idx_batch, neg_idx_batch in train_loader:
            q_batch       = q_batch.to(device)
            pos_idx_batch = pos_idx_batch.to(device)
            neg_idx_batch = neg_idx_batch.to(device)

            p_batch = corpus_t[pos_idx_batch]   # (B, D)
            n_batch = corpus_t[neg_idx_batch]   # (B, D)

            # L2-normalise all three before loss so TripletMarginLoss(p=2)
            # is equivalent to cosine-based triplet loss — consistent with
            # cosine-similarity retrieval at inference time.
            optimizer.zero_grad()
            q_adapted = _l2_normalize(adapter(q_batch))
            p_norm    = _l2_normalize(p_batch)
            n_norm    = _l2_normalize(n_batch)

            loss = loss_fn(q_adapted, p_norm, n_norm)
            loss.backward()
            if cfg.GRAD_CLIP_NORM is not None:
                nn.utils.clip_grad_norm_(adapter.parameters(), cfg.GRAD_CLIP_NORM)
            optimizer.step()

            total_train_loss += loss.item()
            n_train_batches  += 1

        train_loss = total_train_loss / max(n_train_batches, 1)

        # ---- Validate -----------------------------------------------------
        adapter.eval()
        total_val_loss = 0.0
        n_val_batches  = 0

        with torch.no_grad():
            for q_batch, pos_idx_batch, neg_idx_batch in val_loader:
                q_batch       = q_batch.to(device)
                pos_idx_batch = pos_idx_batch.to(device)
                neg_idx_batch = neg_idx_batch.to(device)

                p_batch = corpus_t[pos_idx_batch]
                n_batch = corpus_t[neg_idx_batch]

                q_adapted = _l2_normalize(adapter(q_batch))
                p_norm    = _l2_normalize(p_batch)
                n_norm    = _l2_normalize(n_batch)

                total_val_loss += loss_fn(q_adapted, p_norm, n_norm).item()
                n_val_batches  += 1

        val_loss = total_val_loss / max(n_val_batches, 1)

        # ---- Logging ------------------------------------------------------
        is_best = val_loss < best_val_loss
        marker  = " ✓" if is_best else ""
        print(
            f"  Epoch {epoch:>3}/{cfg.NUM_EPOCHS} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"best={best_val_loss:.6f}{marker}"
        )

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": round(train_loss, 7),
                "val_loss":   round(val_loss, 7),
            }) + "\n")

        # ---- Checkpoint & early stopping ----------------------------------
        if is_best:
            best_val_loss    = val_loss
            patience_counter = 0
            save_adapter(adapter, model_path, {
                "adapter_type": cfg.ADAPTER_TYPE,
                "low_rank_dim": cfg.LOW_RANK_DIM,
                "emb_dim":      cfg.EMB_DIM,
            })
        else:
            patience_counter += 1
            if patience_counter >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}  (patience={cfg.EARLY_STOPPING_PATIENCE})")
                break

    print(f"  Best val loss: {best_val_loss:.6f}")


# ---------------------------------------------------------------------------
# Retrieval evaluation (full corpus)
# ---------------------------------------------------------------------------

def _evaluate_retrieval(
    test_df: pd.DataFrame,
    query_embeddings: np.ndarray,
    corpus_keys: List[Tuple[str, int]],
    key_to_idx: Dict[Tuple[str, int], int],
    corpus_matrix_norm: np.ndarray,   # already L2-normalised
    adapter: Optional[nn.Module],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate retrieval metrics for all unique test queries.

    For each query, ranks all ~165K corpus pages by cosine similarity and
    computes Recall@K, MRR (unbounded), MAP, nDCG@10.

    key_to_idx is passed in (built once by _load_corpus in main()) rather than
    rebuilt here to avoid constructing a 165K-entry dict on every call.
    """
    # Group test triplets by query — get unique positive refs per query
    test_query_groups = (
        test_df.groupby("query_idx")["pos_ref"]
        .apply(lambda x: list(dict.fromkeys(x)))   # unique, order-preserving
        .reset_index()
    )

    k10 = 10

    accum: Dict[str, List[float]] = {
        **{f"recall@{k}": [] for k in cfg.TOP_K_EVAL},
        "mrr": [],
        "map": [],
        "ndcg@10": [],
    }

    if adapter is not None:
        adapter.eval()

    for _, qrow in test_query_groups.iterrows():
        q_idx = int(qrow["query_idx"])
        q_emb = query_embeddings[q_idx].copy().astype(np.float32)

        # Apply adapter
        if adapter is not None:
            with torch.no_grad():
                q_t   = torch.from_numpy(q_emb).unsqueeze(0).to(device)
                q_emb = adapter(q_t).cpu().numpy().squeeze(0)

        # L2-normalise query
        qn = np.linalg.norm(q_emb)
        if qn > 0:
            q_emb /= qn

        # Cosine similarity vs full corpus (fast dot product — matrix already normalised)
        sims           = corpus_matrix_norm @ q_emb   # (N_corpus,)
        ranked_indices = np.argsort(sims)[::-1]       # descending int array

        # Resolve positives to a set of integer corpus row indices.
        # All metric lookups below use this int set — no 165K tuple list needed.
        pos_indices: set = set()
        for ref_str in qrow["pos_ref"]:
            key = parse_page_ref(ref_str)
            if key and key in key_to_idx:
                pos_indices.add(key_to_idx[key])

        if not pos_indices:
            continue

        # Recall@K — only inspect the top k indices per threshold
        for k in cfg.TOP_K_EVAL:
            top_k = set(ranked_indices[:k].tolist())
            accum[f"recall@{k}"].append(1.0 if pos_indices & top_k else 0.0)

        # MRR (unbounded — iterate ranked_indices lazily as integers)
        first_pos_rank = next(
            (rank for rank, idx in enumerate(ranked_indices, 1) if idx in pos_indices),
            None,
        )
        accum["mrr"].append(1.0 / first_pos_rank if first_pos_rank else 0.0)

        # MAP — iterate with early break once all positives are found
        ap_sum, n_found = 0.0, 0
        for rank, idx in enumerate(ranked_indices, 1):
            if idx in pos_indices:
                n_found += 1
                ap_sum  += n_found / rank
                if n_found == len(pos_indices):
                    break
        accum["map"].append(ap_sum / len(pos_indices))

        # nDCG@10 — only inspect the first k10 indices
        dcg  = sum(
            1.0 / np.log2(rank + 1)
            for rank, idx in enumerate(ranked_indices[:k10], 1)
            if idx in pos_indices
        )
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(pos_indices), k10)))
        accum["ndcg@10"].append(dcg / idcg if idcg > 0 else 0.0)

    return {name: float(np.mean(vals)) for name, vals in accum.items() if vals}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = _get_device()

    print("=" * 60)
    print("Adapter Training")
    print("=" * 60)
    arch_note = f"  r={cfg.LOW_RANK_DIM}" if cfg.ADAPTER_TYPE == "low_rank" else ""
    print(f"  Device       : {device}")
    print(f"  Adapter      : {cfg.ADAPTER_TYPE}{arch_note}")
    print(f"  LR={cfg.LEARNING_RATE}  margin={cfg.MARGIN}  batch={cfg.BATCH_SIZE}  "
          f"epochs={cfg.NUM_EPOCHS}  patience={cfg.EARLY_STOPPING_PATIENCE}")

    model_path = cfg.DATA_DIR / "best_model.pt"
    log_path   = cfg.DATA_DIR / "training_log.jsonl"

    # Start a fresh log for each training run
    if log_path.exists():
        log_path.unlink()

    # ------------------------------------------------------------------
    # [1] Load preprocessed data
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading preprocessed data from {cfg.DATA_DIR}")
    emb_path      = cfg.DATA_DIR / "query_embeddings.npy"
    triplets_path = cfg.DATA_DIR / "triplets.csv"

    if not emb_path.exists() or not triplets_path.exists():
        print("  ERROR: Run adapter/preprocess.py first.")
        sys.exit(1)

    query_embeddings: np.ndarray = np.load(emb_path)   # (N_queries, D)
    triplets_df = pd.read_csv(triplets_path)
    # Empty source_file values are written as blank CSV cells and read back as NaN;
    # normalise to empty string so sorted() and set operations work correctly.
    triplets_df["source_file"] = triplets_df["source_file"].fillna("").astype(str)
    print(f"  Query embeddings : {query_embeddings.shape}")
    print(f"  Triplets         : {len(triplets_df):,}")

    # ------------------------------------------------------------------
    # [2] File-level split
    # ------------------------------------------------------------------
    print(f"\n[2/5] File-level split  ({cfg.TRAIN_RATIO}/{cfg.VAL_RATIO}/{cfg.TEST_RATIO})")
    train_df, val_df, test_df = _file_level_split(triplets_df)
    print(f"  Train : {len(train_df):,} triplets  ({train_df['query_idx'].nunique():,} queries)")
    print(f"  Val   : {len(val_df):,} triplets  ({val_df['query_idx'].nunique():,} queries)")
    print(f"  Test  : {len(test_df):,} triplets  ({test_df['query_idx'].nunique():,} queries)")

    # ------------------------------------------------------------------
    # [3] Load corpus embeddings
    # ------------------------------------------------------------------
    print(f"\n[3/5] Loading corpus embeddings")
    corpus_keys, key_to_idx, corpus_matrix = _load_corpus(cfg.DB_PATH)

    # ------------------------------------------------------------------
    # [4] Build datasets & DataLoaders
    # ------------------------------------------------------------------
    print(f"\n[4/5] Building datasets")
    train_dataset = TripletDataset(train_df, query_embeddings, key_to_idx)
    val_dataset   = TripletDataset(val_df,   query_embeddings, key_to_idx)

    print(f"  Train : {len(train_dataset):,} usable triplets")
    print(f"  Val   : {len(val_dataset):,} usable triplets")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ------------------------------------------------------------------
    # [5] Train
    # ------------------------------------------------------------------
    print(f"\n[5/5] Training")
    adapter  = build_adapter(cfg.ADAPTER_TYPE, cfg.EMB_DIM, cfg.LOW_RANK_DIM)
    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    _train(adapter, train_loader, val_loader, corpus_matrix, device, log_path, model_path)

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"Test Evaluation  (loading best checkpoint)")

    best_adapter = load_adapter(model_path).to(device)

    # Normalise corpus matrix in-place for retrieval (training is complete).
    # Guard against zero-norm and NaN/inf rows from degenerate page embeddings.
    print(f"  Normalising corpus matrix for retrieval...")
    norms = np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
    norms = np.where(np.isfinite(norms) & (norms > 0), norms, 1.0)
    corpus_matrix /= norms
    np.nan_to_num(corpus_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    n_test_queries = test_df["query_idx"].nunique()
    print(f"  Test queries: {n_test_queries:,}")

    print(f"  Running baseline (no adapter)...")
    baseline = _evaluate_retrieval(
        test_df, query_embeddings, corpus_keys, key_to_idx, corpus_matrix,
        adapter=None, device=device,
    )

    print(f"  Running adapted model...")
    adapted = _evaluate_retrieval(
        test_df, query_embeddings, corpus_keys, key_to_idx, corpus_matrix,
        adapter=best_adapter, device=device,
    )

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    metric_order = [f"recall@{k}" for k in cfg.TOP_K_EVAL] + ["mrr", "map", "ndcg@10"]
    print(f"\n{'Metric':<14} {'Baseline':>10} {'Adapted':>10} {'Delta':>10} {'Δ%':>8}")
    print("─" * 56)
    for metric in metric_order:
        if metric not in baseline:
            continue
        b     = baseline[metric]
        a     = adapted.get(metric, 0.0)
        delta = a - b
        pct   = (delta / b * 100) if b > 0 else 0.0
        print(f"{metric:<14} {b:>10.4f} {a:>10.4f} {delta:>+10.4f} {pct:>+7.1f}%")

    # ------------------------------------------------------------------
    # Write final log entry
    # ------------------------------------------------------------------
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "final":            True,
            "n_test_queries":   n_test_queries,
            "baseline_metrics": baseline,
            "adapted_metrics":  adapted,
            "delta":            {k: round(adapted.get(k, 0.0) - baseline[k], 6) for k in baseline},
        }) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Done")
    print(f"  Best model   : {model_path}")
    print(f"  Training log : {log_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
