#!/usr/bin/env python3
"""Preprocessing step for adapter training.

Reads the evaluation dataset CSV, generates triplet metadata
(query_idx, pos_ref, neg_ref, neg_type, source_file) and caches
query embeddings via the OpenAI API.

For each query:
  - N_SOFT_NEGS soft-negative triplets  (random pages from the full ~165K-page
    SQLite corpus, excluding the query's own positive and hard-negative pages)
  - One hard-negative triplet per entry in hard_negatives column
    (cross-product with positives; queries with 0 hard negatives get 0 hard triplets)

Output written to adapter/data/:
    query_embeddings.npy   float32 array (N_queries, EMB_DIM)
    query_texts.json       list of query strings (index-aligned, for debugging)
    triplets.csv           one row per triplet: query_idx, pos_ref, neg_ref, neg_type, source_file

Resume-safe: if interrupted during embedding, a partial checkpoint is saved
after every API batch. Re-running resumes from where it left off.

Usage:
    python adapter/preprocess.py
    python adapter/preprocess.py --recompute   # force re-embedding of queries
"""

import argparse
import json
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — works regardless of working directory
# ---------------------------------------------------------------------------
_ADAPTER_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _ADAPTER_DIR.parent
sys.path.insert(0, str(_ADAPTER_DIR))

import config as cfg
from utils import fmt_page_ref, parse_page_ref

load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_col(val) -> list:
    """Parse a JSON-encoded list column (tolerates already-parsed lists)."""
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.strip():
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return []
    return []


# ---------------------------------------------------------------------------
# Corpus page pool from SQLite
# ---------------------------------------------------------------------------

def _load_corpus_pool(db_path: Path) -> List[Tuple[str, int]]:
    """Return list of (filename, page_number) for all pages that have embeddings."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT filename, page_number FROM pdf_page_store "
        "WHERE emb__text_embedding_3_large IS NOT NULL"
    ).fetchall()
    conn.close()
    pool = [(str(r[0]), int(r[1])) for r in rows]
    print(f"  Corpus pool: {len(pool):,} pages with embeddings")
    return pool


# ---------------------------------------------------------------------------
# Triplet generation
# ---------------------------------------------------------------------------

def _build_triplets(df: pd.DataFrame, corpus_pool: List[Tuple[str, int]], rng: random.Random) -> pd.DataFrame:
    """Generate all triplet records.

    For each query row:
      - Soft-neg triplets: (pos × N_SOFT_NEGS) where negatives are random corpus
        pages excluding the query's own positives and hard negatives.
      - Hard-neg triplets: (pos × hard_neg) cartesian product.

    Returns a DataFrame with columns:
        query_idx, pos_ref, neg_ref, neg_type, source_file
    """
    records = []
    skipped_no_pos = 0

    for idx, row in df.iterrows():
        # Parse positives
        pos_refs = [parse_page_ref(r) for r in _parse_json_col(row.get("source_files_with_pages", ""))]
        pos_refs = [p for p in pos_refs if p is not None]

        if not pos_refs:
            skipped_no_pos += 1
            continue

        # Parse hard negatives
        hard_neg_refs = [parse_page_ref(r) for r in _parse_json_col(row.get("hard_negatives", ""))]
        hard_neg_refs = [p for p in hard_neg_refs if p is not None]

        source_file = str(row.get("source_file", "")) if pd.notna(row.get("source_file")) else ""

        # Base exclusion set shared across all positives: the pos pages + hard negs
        base_excluded = set(pos_refs) | set(hard_neg_refs)

        # Emit triplet records — sample fresh soft negs per positive so that
        # multi-positive (multi-hop) queries get diverse negatives rather than
        # the same N_SOFT_NEGS pages repeated for every positive.
        for pos in pos_refs:
            pos_str = fmt_page_ref(*pos)

            # Hard-negative triplets
            for hn in hard_neg_refs:
                records.append({
                    "query_idx":   int(idx),
                    "pos_ref":     pos_str,
                    "neg_ref":     fmt_page_ref(*hn),
                    "neg_type":    "hard",
                    "source_file": source_file,
                })

            # Sample N_SOFT_NEGS for this specific positive via rejection sampling.
            # Rejection rate is ~0.008% with 165K pool and ~12 exclusions — instant.
            soft_negs: List[Tuple[str, int]] = []
            seen_soft: set = set()
            attempts = 0
            max_attempts = cfg.N_SOFT_NEGS * 200
            while len(soft_negs) < cfg.N_SOFT_NEGS and attempts < max_attempts:
                candidate = rng.choice(corpus_pool)
                if candidate not in base_excluded and candidate not in seen_soft:
                    soft_negs.append(candidate)
                    seen_soft.add(candidate)
                attempts += 1
            if len(soft_negs) < cfg.N_SOFT_NEGS:
                print(
                    f"  WARNING: query_idx={int(idx)} pos={pos_str} — "
                    f"only {len(soft_negs)}/{cfg.N_SOFT_NEGS} soft negs sampled "
                    f"(corpus too small or exclusion set too large)"
                )

            # Soft-negative triplets
            for sn in soft_negs:
                records.append({
                    "query_idx":   int(idx),
                    "pos_ref":     pos_str,
                    "neg_ref":     fmt_page_ref(*sn),
                    "neg_type":    "soft",
                    "source_file": source_file,
                })

    if skipped_no_pos:
        print(f"  WARNING: {skipped_no_pos} queries skipped (no positive page ref resolved)")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------

def _build_embedding_client():
    """Auto-detect Azure vs regular OpenAI and return (client, model_or_deployment)."""
    azure_key      = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

    if azure_key and azure_endpoint:
        from openai import AzureOpenAI
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        deployment  = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", cfg.EMBEDDING_MODEL)
        print(f"  Provider: Azure OpenAI | endpoint={azure_endpoint} | deployment={deployment}")
        client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        return client, deployment

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError(
            "No API credentials found. Set OPENAI_API_KEY (or AZURE_OPENAI_API_KEY + "
            "AZURE_OPENAI_ENDPOINT) in your .env file."
        )
    from openai import OpenAI
    print(f"  Provider: OpenAI | model={cfg.EMBEDDING_MODEL}")
    return OpenAI(api_key=openai_key), cfg.EMBEDDING_MODEL


def _embed_queries(queries: List[str], partial_path: Path) -> np.ndarray:
    """Embed all queries, resuming from a partial checkpoint if present.

    Saves progress after every API batch so a restart continues where it left off.
    Returns float32 array of shape (len(queries), EMB_DIM).
    """
    # Load existing partial checkpoint
    done: dict = {}
    if partial_path.exists():
        with open(partial_path) as f:
            done = json.load(f)
        print(f"  Resuming from checkpoint: {len(done):,} / {len(queries):,} already embedded")

    client, model = _build_embedding_client()

    embeddings: List[Optional[list]] = [None] * len(queries)
    for idx_str, emb in done.items():
        embeddings[int(idx_str)] = emb

    remaining = [(i, q) for i, q in enumerate(queries) if embeddings[i] is None]
    total_remaining = len(remaining)

    if total_remaining == 0:
        print("  All queries already embedded (from checkpoint).")
    else:
        for batch_start in range(0, total_remaining, cfg.EMBED_BATCH_SIZE):
            batch      = remaining[batch_start: batch_start + cfg.EMBED_BATCH_SIZE]
            indices    = [b[0] for b in batch]
            texts      = [b[1] for b in batch]

            response = client.embeddings.create(input=texts, model=model)

            for i, item in zip(indices, response.data):
                emb = item.embedding
                embeddings[i] = emb
                done[str(i)]  = emb

            # Checkpoint after every batch
            with open(partial_path, "w") as f:
                json.dump(done, f)

            completed = min(batch_start + cfg.EMBED_BATCH_SIZE, total_remaining)
            print(f"  Embedded {completed:,} / {total_remaining:,} queries", end="\r")

        print(f"  Embedded {total_remaining:,} / {total_remaining:,} queries — done          ")

    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(recompute: bool = False) -> None:
    print("=" * 60)
    print("Adapter Preprocessing")
    print("=" * 60)

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)

    emb_path      = cfg.DATA_DIR / "query_embeddings.npy"
    texts_path    = cfg.DATA_DIR / "query_texts.json"
    triplets_path = cfg.DATA_DIR / "triplets.csv"
    partial_path  = cfg.DATA_DIR / "embeddings_partial.json"

    # ------------------------------------------------------------------
    # [1] Load dataset
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading dataset")
    print(f"  Path: {cfg.DATASET_PATH}")
    df = pd.read_csv(cfg.DATASET_PATH).reset_index(drop=True)
    print(f"  Rows: {len(df):,}")

    queries = df["user_input"].tolist()

    # ------------------------------------------------------------------
    # [2] Build corpus page pool from SQLite
    # ------------------------------------------------------------------
    print(f"\n[2/4] Building corpus pool from SQLite")
    print(f"  DB: {cfg.DB_PATH}")
    corpus_pool = _load_corpus_pool(cfg.DB_PATH)

    # ------------------------------------------------------------------
    # [3] Build triplets
    # ------------------------------------------------------------------
    print(f"\n[3/4] Generating triplets  (seed={cfg.RANDOM_SEED}, soft_negs={cfg.N_SOFT_NEGS})")
    if triplets_path.exists() and not recompute:
        print(f"  Skipping: triplets already at {triplets_path.name}  (pass --recompute to redo)")
        triplets_df = pd.read_csv(triplets_path)
    else:
        rng         = random.Random(cfg.RANDOM_SEED)
        triplets_df = _build_triplets(df, corpus_pool, rng)
        triplets_df.to_csv(triplets_path, index=False)
        print(f"  Saved: {triplets_path}")

    n_hard = int((triplets_df["neg_type"] == "hard").sum())
    n_soft = int((triplets_df["neg_type"] == "soft").sum())
    n_queries_hard = int(triplets_df[triplets_df["neg_type"] == "hard"]["query_idx"].nunique())
    print(f"  Total triplets : {len(triplets_df):,}")
    print(f"    Hard negs    : {n_hard:,}  ({n_queries_hard:,} queries with ≥1 hard neg)")
    print(f"    Soft negs    : {n_soft:,}  (all {len(df):,} queries)")

    # ------------------------------------------------------------------
    # [4] Embed queries
    # ------------------------------------------------------------------
    print(f"\n[4/4] Embedding {len(queries):,} queries via OpenAI API")
    if emb_path.exists() and not recompute:
        print(f"  Skipping: embeddings already at {emb_path.name}  (pass --recompute to redo)")
        embeddings = np.load(emb_path)
    else:
        embeddings = _embed_queries(queries, partial_path)
        np.save(emb_path, embeddings)
        # Clean up partial checkpoint on successful completion
        if partial_path.exists():
            partial_path.unlink()
        print(f"  Saved: {emb_path}  shape={embeddings.shape}")

    # Save query texts for debugging / verification
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Preprocessing complete")
    print(f"  query_embeddings.npy : {embeddings.shape}  float32")
    print(f"  triplets.csv         : {len(triplets_df):,} triplets")
    print(f"                         {n_hard:,} hard  +  {n_soft:,} soft")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess dataset into triplets and cache query embeddings."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force re-embedding of queries even if query_embeddings.npy already exists.",
    )
    args = parser.parse_args()
    main(recompute=args.recompute)
