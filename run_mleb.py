#!/usr/bin/env python3
"""
Run the Massive Legal Embedding Benchmark (MLEB) locally.

Evaluates embedding models on 10 expert-annotated legal retrieval datasets
using the official MTEB evaluation framework, producing nDCG@10 scores
directly comparable to the MLEB leaderboard (https://isaacus.com/mleb).

Usage:
    # Run both models (default):
    python run_mleb.py

    # Run a specific model:
    python run_mleb.py --models openai
    python run_mleb.py --models bedrock-cohere

    # Run on a subset of datasets:
    python run_mleb.py --datasets bar-exam-qa contractual-clause-retrieval

Environment Variables:
    OPENAI_API_KEY          - Required for OpenAI models
    AWS_ACCESS_KEY_ID       - Required for Bedrock Cohere
    AWS_SECRET_ACCESS_KEY   - Required for Bedrock Cohere
    AWS_BEDROCK_REGION      - Bedrock region (default: eu-central-1)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv

from modules.utils import iter_batched

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"


# ============================================================================
# MLEB dataset configurations (from isaacus-dev/mleb)
# ============================================================================
MLEB_DATASETS = [
    {
        "name": "bar-exam-qa",
        "id": "isaacus/mteb-barexam-qa",
        "revision": "dd157bbfa479359488c656981e3999da6f42e4e9",
        "category": "caselaw",
    },
    {
        "name": "scalr",
        "id": "isaacus/mleb-scalr",
        "revision": "319b6cc4b012d733f126a943a8a66bdf9df5dc40",
        "category": "caselaw",
    },
    {
        "name": "singaporean-judicial-keywords",
        "id": "isaacus/singaporean-judicial-keywords",
        "revision": "427e2ae4b22cd9ad990ef8dd4647c16d79c89198",
        "category": "caselaw",
    },
    {
        "name": "gdpr-holdings-retrieval",
        "id": "isaacus/gdpr-holdings-retrieval",
        "revision": "8d41f3d22bb73685b6f42b62ad95940ea3dfbf84",
        "category": "caselaw",
    },
    {
        "name": "uk-legislative-long-titles",
        "id": "isaacus/uk-legislative-long-titles",
        "revision": "436d6a79d06cac556799e9e0be54a6fb90bf7182",
        "category": "regulation",
    },
    {
        "name": "australian-tax-guidance-retrieval",
        "id": "isaacus/australian-tax-guidance-retrieval",
        "revision": "c64c3baac6bfd5f934d2df6e5d42dcb7d87c8ba8",
        "category": "regulation",
    },
    {
        "name": "irish-legislative-summaries",
        "id": "isaacus/irish-legislative-summaries",
        "revision": "bbf8b2d84b7de5970b2ba4ea843c791285fdb1df",
        "category": "regulation",
    },
    {
        "name": "contractual-clause-retrieval",
        "id": "isaacus/contractual-clause-retrieval",
        "revision": "48ed7bcb1f50896a0f71461a04b2df0ca84329d9",
        "category": "contracts",
    },
    {
        "name": "license-tldr-retrieval",
        "id": "isaacus/license-tldr-retrieval",
        "revision": "ec00129f88e9476e582131dc3a5db9220dfefa48",
        "category": "contracts",
    },
    {
        "name": "consumer-contracts-qa",
        "id": "isaacus/mleb-consumer-contracts-qa",
        "revision": "2095f248902963b4480ac96b774ba64b2104cbee",
        "category": "contracts",
    },
]


# ============================================================================
# MTEB-compatible model wrappers
# ============================================================================
def _extract_texts(inputs) -> List[str]:
    """Extract plain text strings from MTEB DataLoader[BatchedInput]."""
    texts: List[str] = []
    for batch in inputs:
        if isinstance(batch, dict) and "text" in batch:
            # Standard BatchedInput: {"text": ["...", "..."]}
            texts.extend(batch["text"])
        elif isinstance(batch, (list, tuple)):
            texts.extend([str(t) for t in batch])
        else:
            texts.append(str(batch))
    return texts


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> torch.Tensor:
    """Compute cosine similarity matrix between two sets of embeddings."""
    a_t = torch.tensor(a, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32)
    a_n = torch.nn.functional.normalize(a_t, p=2, dim=1)
    b_n = torch.nn.functional.normalize(b_t, p=2, dim=1)
    return a_n @ b_n.T


class OpenAIEmbedder:
    """MTEB-compatible wrapper for OpenAI / Azure OpenAI embedding models."""

    def __init__(self, model_id: str = "text-embedding-3-large", batch_size: int = 16):
        from mteb.models import ModelMeta

        self.batch_size = batch_size
        self.model_id = model_id

        # Auto-detect Azure vs direct OpenAI
        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if azure_key and azure_endpoint:
            from langchain_openai import AzureOpenAIEmbeddings
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            # Use the embedding deployment name if set, otherwise use model_id
            deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", model_id)
            self.client = AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                api_version=api_version,
                azure_deployment=deployment,
            )
            print(f"  Using Azure OpenAI (deployment: {deployment})")
        else:
            from langchain_openai import OpenAIEmbeddings
            self.client = OpenAIEmbeddings(model=model_id)
            print(f"  Using OpenAI API (model: {model_id})")

        self.mteb_model_meta = ModelMeta(
            name=f"openai/{model_id}",
            revision="1",
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=8191,
            embed_dim=3072,
            license=None,
            open_weights=False,
            public_training_data=None,
            public_training_code=None,
            framework=["API"],
            reference=None,
            similarity_fn_name="cosine",
            use_instructions=None,
            training_datasets=None,
            adapted_from=None,
            superseded_by=None,
            loader=None,
        )

    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, **kwargs) -> np.ndarray:
        from mteb.types import PromptType

        sentences = _extract_texts(inputs)

        if prompt_type == PromptType.query:
            embeddings = [self.client.embed_query(s) for s in sentences]
        else:
            embeddings = [
                emb
                for batch in iter_batched(sentences, batch_size=self.batch_size)
                for emb in self.client.embed_documents(list(batch))
            ]

        return np.array(embeddings, dtype=np.float32)

    def similarity(self, e1: np.ndarray, e2: np.ndarray) -> torch.Tensor:
        return _cosine_similarity(e1, e2)

    def similarity_pairwise(self, e1: np.ndarray, e2: np.ndarray) -> torch.Tensor:
        return torch.diag(_cosine_similarity(e1, e2))


class BedrockCohereEmbedder:
    """MTEB-compatible wrapper for Cohere Embed v4 via AWS Bedrock.

    Properly handles the asymmetric input_type parameter:
      - PromptType.query    -> input_type="search_query"
      - PromptType.document -> input_type="search_document"
    """

    def __init__(
        self,
        model_id: str = "eu.cohere.embed-v4:0",
        region: Optional[str] = None,
        batch_size: int = 16,
    ):
        import boto3
        from mteb.models import ModelMeta

        self.model_id = model_id
        self.batch_size = batch_size
        self.region = region or os.environ.get("AWS_BEDROCK_REGION", "eu-central-1")
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

        self.mteb_model_meta = ModelMeta(
            name=f"cohere/{model_id}",
            revision="1",
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=128000,
            embed_dim=1536,
            license=None,
            open_weights=False,
            public_training_data=None,
            public_training_code=None,
            framework=["API"],
            reference=None,
            similarity_fn_name="cosine",
            use_instructions=None,
            training_datasets=None,
            adapted_from=None,
            superseded_by=None,
            loader=None,
        )

    def _embed_batch(
        self, texts: List[str], input_type: str
    ) -> List[List[float]]:
        """Call Bedrock Cohere embed API with explicit input_type."""
        import json as _json

        body = _json.dumps({
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        })

        response = self.client.invoke_model(
            body=body,
            modelId=self.model_id,
            accept="*/*",
            contentType="application/json",
        )

        response_body = _json.loads(response["body"].read())

        # Cohere v4 returns {"embeddings": {"float": [[...], ...]}}
        embeddings = response_body.get("embeddings", {})
        if isinstance(embeddings, dict) and "float" in embeddings:
            return embeddings["float"]
        # Cohere v3 fallback: {"embeddings": [[...], ...]}
        if isinstance(embeddings, list):
            return embeddings

        raise ValueError(f"Unexpected Bedrock Cohere response format: {list(response_body.keys())}")

    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, **kwargs) -> np.ndarray:
        from mteb.types import PromptType

        sentences = _extract_texts(inputs)

        # Map MTEB prompt type to Cohere input_type
        if prompt_type == PromptType.query:
            input_type = "search_query"
        else:
            input_type = "search_document"

        all_embeddings: List[List[float]] = []
        for batch in iter_batched(sentences, batch_size=self.batch_size):
            batch_list = list(batch)
            retries = 0
            max_retries = 8
            while retries < max_retries:
                try:
                    vecs = self._embed_batch(batch_list, input_type)
                    all_embeddings.extend(vecs)
                    break
                except Exception as e:
                    err = str(e).lower()
                    is_retriable = (
                        "throttl" in err or "rate" in err or "too many" in err
                        or "serviceunavailable" in err or "unable to process" in err
                        or "timeout" in err or "timed out" in err
                    )
                    if is_retriable and retries < max_retries - 1:
                        delay = min(5 * (2 ** retries), 120)
                        print(f"  Retryable error, waiting {delay}s (attempt {retries + 2}/{max_retries})...")
                        time.sleep(delay)
                        retries += 1
                    else:
                        raise

        return np.array(all_embeddings, dtype=np.float32)

    def similarity(self, e1: np.ndarray, e2: np.ndarray) -> torch.Tensor:
        return _cosine_similarity(e1, e2)

    def similarity_pairwise(self, e1: np.ndarray, e2: np.ndarray) -> torch.Tensor:
        return torch.diag(_cosine_similarity(e1, e2))


# ============================================================================
# MTEB task resolution (borrowed from MLEB's pattern)
# ============================================================================
_TASK_CACHE: Dict[str, Any] = {}


def get_mteb_task(dataset_config: Dict[str, str]):
    """Create an MTEB retrieval task for an MLEB dataset."""
    from mteb import TaskMetadata
    from mteb.abstasks import AbsTaskRetrieval

    name = dataset_config["name"]

    if name in _TASK_CACHE:
        return _TASK_CACHE[name]

    class MLEBTask(AbsTaskRetrieval):
        metadata = TaskMetadata(
            name=name,
            dataset={
                "path": dataset_config["id"],
                "revision": dataset_config["revision"],
            },
            main_score="ndcg_at_10",
            description="An MLEB evaluation dataset.",
            type="Retrieval",
            category="t2t",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            date=("2021-06-06", "2025-07-28"),
            domains=["Legal"],
            task_subtypes=[],
            license="cc-by-4.0",
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="found",
        )

    task = MLEBTask()
    _TASK_CACHE[name] = task
    return task


# ============================================================================
# Evaluation runner
# ============================================================================
def evaluate_model(
    model,
    model_name: str,
    dataset_configs: List[Dict[str, str]],
    output_dir: str = "mleb_results",
) -> Dict[str, Dict[str, float]]:
    """Run MTEB evaluation on MLEB datasets for a single model."""
    import mteb

    results = {}
    results_dir = OUTPUT_DIR / output_dir

    for i, ds_config in enumerate(dataset_configs, 1):
        ds_name = ds_config["name"]
        print(f"  [{i}/{len(dataset_configs)}] {ds_name}...")

        try:
            task = get_mteb_task(ds_config)

            eval_results = mteb.MTEB(tasks=[task]).run(
                model, output_folder=str(results_dir), verbosity=0, progress_bar=False
            )

            # Extract scores from the first result's first split
            task_result = eval_results[0]
            all_scores = list(task_result.scores.values())
            if all_scores and all_scores[0]:
                raw_scores = all_scores[0][0]
            else:
                raw_scores = {}

            scores = {
                k: float(v)
                for k, v in raw_scores.items()
                if isinstance(v, (int, float, np.integer, np.floating))
            }

            results[ds_name] = scores
            ndcg10 = scores.get("main_score", scores.get("ndcg_at_10", 0))
            print(f"    nDCG@10: {ndcg10:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[ds_name] = {"main_score": 0.0, "error": str(e)}

    return results


# ============================================================================
# Main
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MLEB (Massive Legal Embedding Benchmark) locally"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["openai", "bedrock-cohere", "all"],
        default=["all"],
        help="Which models to evaluate (default: all)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific MLEB dataset names to run (default: all 10)",
    )
    parser.add_argument(
        "--cohere-model-id",
        type=str,
        default="eu.cohere.embed-v4:0",
        help="Bedrock Cohere model ID (default: eu.cohere.embed-v4:0)",
    )
    parser.add_argument(
        "--openai-model-id",
        type=str,
        default="text-embedding-3-large",
        help="OpenAI model ID (default: text-embedding-3-large)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: output/mleb_results.json)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("MLEB - Massive Legal Embedding Benchmark")
    print("=" * 60)

    # Resolve which models to run
    run_models = set(args.models)
    if "all" in run_models:
        run_models = {"openai", "bedrock-cohere"}

    # Resolve which datasets to run
    if args.datasets:
        dataset_configs = [
            d for d in MLEB_DATASETS if d["name"] in args.datasets
        ]
        if not dataset_configs:
            print(f"\nError: No matching datasets. Available: {[d['name'] for d in MLEB_DATASETS]}")
            return 1
    else:
        dataset_configs = MLEB_DATASETS

    print(f"  Models: {sorted(run_models)}")
    print(f"  Datasets: {len(dataset_configs)}/10")
    print()

    all_results: Dict[str, Any] = {
        "benchmark": "MLEB (Massive Legal Embedding Benchmark)",
        "num_datasets": len(dataset_configs),
        "datasets": [d["name"] for d in dataset_configs],
        "models": {},
    }

    # --- OpenAI ---
    if "openai" in run_models:
        print(f"{'=' * 60}")
        print(f"Evaluating: OpenAI {args.openai_model_id}")
        print(f"{'=' * 60}")

        model = OpenAIEmbedder(model_id=args.openai_model_id)
        results = evaluate_model(model, args.openai_model_id, dataset_configs)

        avg_score = np.mean([s.get("main_score", 0) for s in results.values()])
        print(f"\n  Overall nDCG@10: {avg_score:.4f}")

        # Category averages
        categories: Dict[str, List[float]] = {}
        for ds_config in dataset_configs:
            cat = ds_config.get("category", "other")
            score = results.get(ds_config["name"], {}).get("main_score", 0)
            categories.setdefault(cat, []).append(score)

        cat_avgs = {cat: float(np.mean(scores)) for cat, scores in categories.items()}
        for cat, avg in sorted(cat_avgs.items()):
            print(f"  {cat}: {avg:.4f}")

        all_results["models"][args.openai_model_id] = {
            "provider": "openai",
            "overall_ndcg10": float(avg_score),
            "category_averages": cat_avgs,
            "per_dataset": {
                name: {"ndcg_at_10": scores.get("main_score", 0), **scores}
                for name, scores in results.items()
            },
        }
        print()

    # --- Bedrock Cohere ---
    if "bedrock-cohere" in run_models:
        print(f"{'=' * 60}")
        print(f"Evaluating: Bedrock Cohere {args.cohere_model_id}")
        print(f"  (with proper input_type: search_query / search_document)")
        print(f"{'=' * 60}")

        model = BedrockCohereEmbedder(model_id=args.cohere_model_id, batch_size=1)
        results = evaluate_model(model, args.cohere_model_id, dataset_configs)

        avg_score = np.mean([s.get("main_score", 0) for s in results.values()])
        print(f"\n  Overall nDCG@10: {avg_score:.4f}")

        categories: Dict[str, List[float]] = {}
        for ds_config in dataset_configs:
            cat = ds_config.get("category", "other")
            score = results.get(ds_config["name"], {}).get("main_score", 0)
            categories.setdefault(cat, []).append(score)

        cat_avgs = {cat: float(np.mean(scores)) for cat, scores in categories.items()}
        for cat, avg in sorted(cat_avgs.items()):
            print(f"  {cat}: {avg:.4f}")

        all_results["models"][args.cohere_model_id] = {
            "provider": "bedrock",
            "overall_ndcg10": float(avg_score),
            "category_averages": cat_avgs,
            "per_dataset": {
                name: {"ndcg_at_10": scores.get("main_score", 0), **scores}
                for name, scores in results.items()
            },
        }
        print()

    # --- Comparison ---
    model_keys = list(all_results["models"].keys())
    if len(model_keys) >= 2:
        print(f"{'=' * 60}")
        print("Comparison")
        print(f"{'=' * 60}")

        m1, m2 = model_keys[0], model_keys[1]
        r1, r2 = all_results["models"][m1], all_results["models"][m2]

        print(f"\n  {'Dataset':<40} {m1[:20]:>12} {m2[:20]:>12} {'Delta':>8}")
        print(f"  {'-' * 76}")

        for ds_config in dataset_configs:
            ds = ds_config["name"]
            s1 = r1["per_dataset"].get(ds, {}).get("ndcg_at_10", 0)
            s2 = r2["per_dataset"].get(ds, {}).get("ndcg_at_10", 0)
            delta = s2 - s1
            winner = "<<<" if abs(delta) > 0.02 else ""
            print(f"  {ds:<40} {s1:>12.4f} {s2:>12.4f} {delta:>+8.4f} {winner}")

        print(f"  {'-' * 76}")
        o1, o2 = r1["overall_ndcg10"], r2["overall_ndcg10"]
        print(f"  {'OVERALL':<40} {o1:>12.4f} {o2:>12.4f} {o2 - o1:>+8.4f}")

        # Category comparison
        print(f"\n  By category:")
        all_cats = sorted(set(list(r1.get("category_averages", {}).keys()) + list(r2.get("category_averages", {}).keys())))
        for cat in all_cats:
            c1 = r1.get("category_averages", {}).get(cat, 0)
            c2 = r2.get("category_averages", {}).get(cat, 0)
            print(f"    {cat:<20} {c1:.4f} vs {c2:.4f} (delta: {c2 - c1:+.4f})")

    # --- Save ---
    if args.output:
        output_path = Path(args.output)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "mleb_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {output_path}")

    print(f"\n{'=' * 60}")
    print("MLEB evaluation complete")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
