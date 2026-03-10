# TODO: Add Statistical Significance Testing to evaluate_search.py

## What's missing

`evaluate_search.py` currently computes only **mean** aggregate metrics (lines 622-633).
The thesis (§3.6.2) commits to paired permutation tests and bootstrap confidence intervals.
This must be implemented before running final experiments.

## What to implement

### 1. Paired permutation test (Recall@5, the primary metric)

For each pair of systems (e.g., C1 vs C2c), compare their **per-query Recall@5** scores:

```python
import numpy as np

def paired_permutation_test(scores_a, scores_b, n_permutations=10000, seed=42):
    """Two-sided paired permutation test.

    Returns p-value: probability of observing a mean difference
    this large or larger under the null (systems are equal).
    """
    rng = np.random.default_rng(seed)
    diffs = np.array(scores_a) - np.array(scores_b)
    observed = np.abs(np.mean(diffs))
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        if np.abs(np.mean(signs * diffs)) >= observed:
            count += 1
    return count / n_permutations
```

### 2. Bootstrap confidence interval (Recall@5 difference)

```python
def bootstrap_ci(scores_a, scores_b, n_resamples=10000, alpha=0.05, seed=42):
    """95% bootstrap CI for mean(scores_a - scores_b)."""
    rng = np.random.default_rng(seed)
    diffs = np.array(scores_a) - np.array(scores_b)
    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)
```

### 3. Integration plan

- The per-query results are already saved in the JSON output (`per_query` field).
- Option A: Add a `--compare` mode to `evaluate_search.py` that loads two result JSONs
  and runs the permutation test + bootstrap CI on Recall@20.
- Option B: Create a separate `compare_systems.py` script that takes two eval JSONs.
- Either way, extract per-query `recall@5` arrays from each JSON and run the two
  functions above.

### 4. Parameters (matching thesis §3.6.2)

| Parameter              | Value       |
|------------------------|-------------|
| Primary metric         | Recall@5    |
| Significance level (α) | 0.05        |
| Permutation iterations | 10,000      |
| Bootstrap resamples    | 10,000      |
| CI level               | 95%         |
| Random seed            | 42          |

All other metrics (Recall@1/5/10, MRR, MAP, nDCG@K) are reported as exploratory
(no multiple comparison correction needed since there is only one primary metric).

### 5. Output format

For each system pair, report:
- Observed Recall@5 difference (Δ)
- p-value from permutation test
- 95% bootstrap CI for Δ
- Whether p < 0.05 (significant yes/no)
