#!/usr/bin/env python3
"""Deterministic train/val/test dataset splitter for adapter training.

This is the **canonical source** of the connected-components split used by
adapter/train.py.  Both scripts use the same function with the same seed so
the split is always identical as long as adapter/data/triplets.csv is unchanged.

Split strategy (70 / 15 / 15):
  - Queries that share a positive document are placed into the same "component"
    via union-find, then whole components are assigned to train / val / test.
  - This prevents document leakage: a page that is a positive in train cannot
    also be a positive in val or test.

Outputs (saved to --output-dir, default adapter/data/splits/):
    train.csv   — full dataset rows assigned to the training split
    val.csv     — full dataset rows assigned to the validation split
    test.csv    — full dataset rows assigned to the test split

Only rows that appear in triplets.csv (had at least one valid positive page
reference) are included.  Rows skipped during preprocessing (no positives)
are excluded from all splits.

Usage:
    # Use defaults from config.py
    python adapter/split_dataset.py

    # Custom dataset / output location
    python adapter/split_dataset.py \\
        --dataset path/to/dataset.csv \\
        --output-dir adapter/data/splits
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_ADAPTER_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _ADAPTER_DIR.parent
sys.path.insert(0, str(_ADAPTER_DIR))

import config as cfg


# ---------------------------------------------------------------------------
# Canonical split function — imported by train.py
# ---------------------------------------------------------------------------

def _connected_components_split(
    triplets_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split triplets so that (1) each query is in exactly one split, and (2) no
    document appears as a positive in more than one split.

    Queries that share any positive document (pos_ref) are placed in the same
    "component"; each component is assigned entirely to train, val, or test.
    This avoids document leakage (same doc positive in train and val/test) and
    keeps each query in a single split.

    Deterministic: uses np.random.RandomState(cfg.RANDOM_SEED).
    """
    qp = triplets_df[["query_idx", "pos_ref"]].drop_duplicates()
    doc_to_queries: Dict[str, List[int]] = {}
    for _, row in qp.iterrows():
        doc = str(row["pos_ref"])
        q = int(row["query_idx"])
        doc_to_queries.setdefault(doc, []).append(q)

    all_queries = triplets_df["query_idx"].unique()
    parent: Dict[int, int] = {q: q for q in all_queries}

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for queries in doc_to_queries.values():
        for i in range(1, len(queries)):
            union(queries[0], queries[i])

    components: Dict[int, List[int]] = {}
    for q in all_queries:
        root = find(q)
        components.setdefault(root, []).append(q)
    comp_list = list(components.values())

    total_queries = len(all_queries)
    rng = np.random.RandomState(cfg.RANDOM_SEED)
    order = np.arange(len(comp_list), dtype=np.intp)
    rng.shuffle(order)
    n_train_target = int(total_queries * cfg.TRAIN_RATIO)
    n_val_target   = int(total_queries * cfg.VAL_RATIO)
    train_queries: set = set()
    val_queries: set = set()
    test_queries: set = set()
    count_train, count_val = 0, 0
    for idx in order:
        comp = comp_list[idx]
        n = len(comp)
        if count_train < n_train_target:
            train_queries.update(comp)
            count_train += n
        elif count_val < n_val_target:
            val_queries.update(comp)
            count_val += n
        else:
            test_queries.update(comp)

    query_to_split: Dict[int, str] = {}
    for q in train_queries:
        query_to_split[q] = "train"
    for q in val_queries:
        query_to_split[q] = "val"
    for q in test_queries:
        query_to_split[q] = "test"

    def split_for_row(row) -> str:
        return query_to_split[int(row["query_idx"])]

    triplets_df = triplets_df.copy()
    triplets_df["_split"] = triplets_df.apply(split_for_row, axis=1)
    train_df = triplets_df[triplets_df["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    val_df   = triplets_df[triplets_df["_split"] == "val"].drop(columns=["_split"]).reset_index(drop=True)
    test_df  = triplets_df[triplets_df["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Standalone CLI — saves dataset-row CSVs for each split
# ---------------------------------------------------------------------------

def save_dataset_splits(
    dataset_path: Path,
    triplets_path: Path,
    output_dir: Path,
) -> None:
    """Run the split and save train/val/test CSVs of the full dataset rows.

    Args:
        dataset_path:  Full dataset CSV (e.g. vision_validated_relaxed.csv).
        triplets_path: adapter/data/triplets.csv (built by preprocess.py).
        output_dir:    Directory to write train.csv / val.csv / test.csv.
    """
    print("=" * 60)
    print("Dataset Split  (connected-components, seed={})".format(cfg.RANDOM_SEED))
    print("=" * 60)

    print(f"\n  Dataset  : {dataset_path}")
    print(f"  Triplets : {triplets_path}")
    print(f"  Output   : {output_dir}")

    if not dataset_path.exists():
        print(f"\nERROR: dataset not found: {dataset_path}")
        sys.exit(1)
    if not triplets_path.exists():
        print(f"\nERROR: triplets.csv not found: {triplets_path}")
        print("       Run adapter/preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(dataset_path).reset_index(drop=True)
    print(f"\n  Loaded {len(df):,} dataset rows")

    triplets_df = pd.read_csv(triplets_path)
    triplets_df["source_file"] = triplets_df["source_file"].fillna("").astype(str)
    print(f"  Loaded {len(triplets_df):,} triplets  ({triplets_df['query_idx'].nunique():,} unique queries)")

    # Split
    train_triplets, val_triplets, test_triplets = _connected_components_split(triplets_df)

    train_idx = sorted(train_triplets["query_idx"].unique())
    val_idx   = sorted(val_triplets["query_idx"].unique())
    test_idx  = sorted(test_triplets["query_idx"].unique())

    # Map query_idx → dataset rows (query_idx is the df row index)
    train_df = df.iloc[train_idx].copy()
    val_df   = df.iloc[val_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir  / "val.csv",   index=False)
    test_df.to_csv(output_dir / "test.csv",  index=False)

    total_split = len(train_idx) + len(val_idx) + len(test_idx)
    skipped = triplets_df["query_idx"].nunique() - total_split  # should be 0
    print(f"\n  Split results ({cfg.TRAIN_RATIO}/{cfg.VAL_RATIO}/{cfg.TEST_RATIO}):")
    print(f"    Train : {len(train_df):,} queries  →  {output_dir / 'train.csv'}")
    print(f"    Val   : {len(val_df):,} queries  →  {output_dir / 'val.csv'}")
    print(f"    Test  : {len(test_df):,} queries  →  {output_dir / 'test.csv'}")
    if skipped:
        print(f"    (Note: {skipped} queries in triplets had no split assignment)")
    rows_without_triplets = len(df) - total_split
    if rows_without_triplets > 0:
        print(f"    ({rows_without_triplets} dataset rows excluded — no valid positives in triplets.csv)")
    print(f"\n  Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split the dataset into train/val/test CSVs using the same "
            "deterministic connected-components algorithm as adapter/train.py."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(cfg.DATASET_PATH),
        help=f"Path to the full dataset CSV (default: {cfg.DATASET_PATH})",
    )
    parser.add_argument(
        "--triplets",
        type=str,
        default=str(cfg.DATA_DIR / "triplets.csv"),
        help=f"Path to triplets.csv built by preprocess.py (default: adapter/data/triplets.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_ADAPTER_DIR / "data" / "splits"),
        help="Directory to write train.csv / val.csv / test.csv (default: adapter/data/splits/)",
    )
    args = parser.parse_args()

    save_dataset_splits(
        dataset_path=Path(args.dataset),
        triplets_path=Path(args.triplets),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
