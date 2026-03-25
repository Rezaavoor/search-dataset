#!/usr/bin/env python3
"""
Statistical significance testing for retrieval evaluation results.

Compares two eval JSON files (produced by evaluate_search.py) using:
  1. Paired permutation test on per-query Recall@5
  2. Bootstrap 95% confidence interval for the mean Recall@5 difference

No API calls, no re-computation — operates entirely on the per-query
scores already stored in the eval JSONs.

Usage:
    python statistical_test.py output/eval_baseline.json output/eval_experiment.json
    python statistical_test.py --metric recall@10 output/eval_A.json output/eval_B.json
    python statistical_test.py --all-pairs output/eval_*.json
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


def load_per_query_metric(path: str, metric: str) -> tuple[list[str], list[float]]:
    """Load per-query scores for a given metric from an eval JSON file."""
    with open(path) as f:
        data = json.load(f)

    queries = []
    scores = []
    for entry in data["per_query"]:
        if "error" in entry:
            continue
        val = entry.get(metric)
        if val is None:
            continue
        queries.append(entry["query"])
        scores.append(float(val))

    return queries, scores


def paired_permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired permutation test.

    Returns p-value: fraction of random sign-flips that produce a mean
    difference at least as extreme as the observed one.
    """
    rng = np.random.default_rng(seed)
    diffs = scores_a - scores_b
    observed = np.abs(np.mean(diffs))
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        if np.abs(np.mean(signs * diffs)) >= observed:
            count += 1
    return count / n_permutations


def bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean difference (A - B)."""
    rng = np.random.default_rng(seed)
    diffs = scores_a - scores_b
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means[i] = np.mean(sample)
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def compare_pair(
    path_a: str,
    path_b: str,
    metric: str,
    n_permutations: int,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> dict:
    """Run both tests for one pair of eval files."""
    queries_a, scores_a_list = load_per_query_metric(path_a, metric)
    queries_b, scores_b_list = load_per_query_metric(path_b, metric)

    # Match queries by position (evaluate_search.py processes the same
    # dataset CSV in order, so per_query arrays are aligned).
    if len(scores_a_list) != len(scores_b_list):
        # Fall back to query-text matching if lengths differ
        query_to_b = {q: s for q, s in zip(queries_b, scores_b_list)}
        matched_a, matched_b = [], []
        for q, s in zip(queries_a, scores_a_list):
            if q in query_to_b:
                matched_a.append(s)
                matched_b.append(query_to_b[q])
        scores_a_arr = np.array(matched_a)
        scores_b_arr = np.array(matched_b)
        n_matched = len(matched_a)
    else:
        scores_a_arr = np.array(scores_a_list)
        scores_b_arr = np.array(scores_b_list)
        n_matched = len(scores_a_list)

    if n_matched == 0:
        return {"error": "No matching queries found between the two files."}

    mean_a = float(scores_a_arr.mean())
    mean_b = float(scores_b_arr.mean())
    delta = mean_a - mean_b

    p_value = paired_permutation_test(
        scores_a_arr, scores_b_arr, n_permutations, seed
    )
    ci_lower, ci_upper = bootstrap_ci(
        scores_a_arr, scores_b_arr, n_bootstrap, alpha, seed
    )

    return {
        "file_a": Path(path_a).name,
        "file_b": Path(path_b).name,
        "metric": metric,
        "n_queries": n_matched,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": delta,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_value < alpha,
    }


def format_result(r: dict) -> str:
    """Pretty-print one comparison result."""
    if "error" in r:
        return f"  ERROR: {r['error']}"

    sig_marker = "***" if r["significant"] else "(n.s.)"
    lines = [
        f"  {r['file_a']}  vs  {r['file_b']}",
        f"  Metric: {r['metric']}   |   Queries: {r['n_queries']}",
        f"  Mean A: {r['mean_a']:.4f}   Mean B: {r['mean_b']:.4f}   Δ: {r['delta']:+.4f}",
        f"  Permutation p-value: {r['p_value']:.4f}  {sig_marker}",
        f"  95% Bootstrap CI:    [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance tests for retrieval evaluation results."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Two or more eval JSON files produced by evaluate_search.py",
    )
    parser.add_argument(
        "--metric",
        default="recall@5",
        help="Per-query metric to compare (default: recall@5)",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Compare every pair of files (instead of just file1 vs file2)",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10_000,
        help="Number of permutation iterations (default: 10000)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    if args.all_pairs:
        if len(args.files) < 2:
            parser.error("--all-pairs requires at least 2 files")
        pairs = list(combinations(args.files, 2))
    else:
        if len(args.files) != 2:
            parser.error("Provide exactly 2 files, or use --all-pairs with 2+")
        pairs = [(args.files[0], args.files[1])]

    print(f"Metric: {args.metric}  |  α={args.alpha}  |  "
          f"permutations={args.permutations}  |  bootstrap={args.bootstrap}  |  seed={args.seed}")
    print("=" * 80)

    for path_a, path_b in pairs:
        result = compare_pair(
            path_a, path_b, args.metric,
            args.permutations, args.bootstrap, args.alpha, args.seed,
        )
        print()
        print(format_result(result))

    print()
    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
