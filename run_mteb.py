#!/usr/bin/env python3
"""
External benchmark: evaluate models on two public MTEB retrieval tasks.

Runs the project's embedding models against ARCChallenge (general-domain,
science QA) and BarExamQA (legal-domain, US bar exam) to verify that the
trained query adapter does not degrade general retrieval while maintaining
its in-domain advantage.

  ARCChallenge  - ~9,350 corpus docs, ~1,172 queries  (general science)
  BarExamQA     - ~116 corpus docs,   ~117 queries    (US legal provisions)

Datasets are downloaded from HuggingFace on first run and cached locally.
MTEB caches per-model results so re-runs are free (no API calls).

Usage:
    # Both tasks, all 5 models:
    python run_mteb.py

    # Both tasks, specific models:
    python run_mteb.py --models openai openai+adapter

    # Single task:
    python run_mteb.py --tasks ARCChallenge --models openai openai+adapter

Environment Variables:
    OPENAI_API_KEY / AZURE_OPENAI_*  - Required for openai / openai+adapter
    VOYAGE_API_KEY                   - Required for voyage-* models
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
DEFAULT_ADAPTER_PATH = SCRIPT_DIR / "adapter" / "data" / "best_full_rank.pt"

VOYAGE_OUTPUT_DIM = 2048

BENCHMARK_TASKS = ["ARCChallenge", "BarExamQA"]
MODEL_CHOICES = ["openai", "openai+adapter", "voyage-4-large", "voyage-4", "voyage-4-lite"]


# ============================================================================
# Shared helpers
# ============================================================================
def _extract_texts(inputs) -> List[str]:
    """Extract plain text strings from MTEB DataLoader[BatchedInput]."""
    texts: List[str] = []
    for batch in inputs:
        if isinstance(batch, dict) and "text" in batch:
            texts.extend(batch["text"])
        elif isinstance(batch, (list, tuple)):
            texts.extend([str(t) for t in batch])
        else:
            texts.append(str(batch))
    return texts


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> torch.Tensor:
    """Cosine similarity matrix between two embedding sets."""
    a_t = torch.tensor(a, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32)
    a_n = torch.nn.functional.normalize(a_t, p=2, dim=1)
    b_n = torch.nn.functional.normalize(b_t, p=2, dim=1)
    return a_n @ b_n.T


def _load_adapter(adapter_path: Path):
    """Load a trained query adapter checkpoint from disk."""
    sys.path.insert(0, str(SCRIPT_DIR / "adapter"))
    from model import load_adapter as _la
    adapter = _la(adapter_path)
    adapter.eval()
    return adapter


# ============================================================================
# MTEB-compatible model wrappers
# ============================================================================
class OpenAIEmbedder:
    """MTEB-compatible wrapper for text-embedding-3-large.

    Auto-detects Azure OpenAI vs direct OpenAI from environment variables.
    """

    def __init__(self, model_id: str = "text-embedding-3-large", batch_size: int = 16):
        from mteb.models import ModelMeta

        self.batch_size = batch_size
        self.model_id = model_id

        azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if azure_key and azure_endpoint:
            from langchain_openai import AzureOpenAIEmbeddings
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", model_id)
            self.client = AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=azure_key,
                api_version=api_version,
                azure_deployment=deployment,
            )
            print(f"  Provider: Azure OpenAI (deployment: {deployment})")
        else:
            from langchain_openai import OpenAIEmbeddings
            self.client = OpenAIEmbeddings(model=model_id)
            print(f"  Provider: OpenAI (model: {model_id})")

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


class OpenAIEmbedderWithAdapter(OpenAIEmbedder):
    """text-embedding-3-large with the trained full-rank query adapter.

    Document embeddings are unchanged; query embeddings pass through the
    linear adapter, mirroring how evaluate_search.py applies --adapter.
    """

    def __init__(
        self,
        model_id: str = "text-embedding-3-large",
        adapter_path: Path = DEFAULT_ADAPTER_PATH,
        batch_size: int = 16,
    ):
        super().__init__(model_id=model_id, batch_size=batch_size)

        if not Path(adapter_path).exists():
            raise FileNotFoundError(
                f"Adapter checkpoint not found: {adapter_path}\n"
                f"Train it first with: python adapter/train.py"
            )
        self.adapter = _load_adapter(Path(adapter_path))

        from mteb.models import ModelMeta
        self.mteb_model_meta = ModelMeta(
            name=f"openai/{model_id}+adapter",
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
        print(f"  Adapter: {adapter_path}")

    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, **kwargs) -> np.ndarray:
        from mteb.types import PromptType

        embeddings = super().encode(
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )

        if prompt_type == PromptType.query:
            with torch.no_grad():
                t = torch.from_numpy(embeddings)
                embeddings = self.adapter(t).numpy()

        return embeddings.astype(np.float32)


class VoyageEmbedder:
    """MTEB-compatible wrapper for the Voyage AI embedding family.

    Corpus is always embedded with voyage-4-large. Queries can use a lighter
    model (voyage-4 or voyage-4-lite) for asymmetric evaluation.
    """

    def __init__(
        self,
        corpus_model_id: str = "voyage-4-large",
        query_model_id: Optional[str] = None,
        batch_size: int = 64,
    ):
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "voyageai is required for Voyage models. "
                "Install with: pip install voyageai"
            ) from exc

        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("VOYAGE_API_KEY is not set.")

        self.client = voyageai.Client(api_key=api_key)
        self.corpus_model_id = corpus_model_id
        self.query_model_id = query_model_id or corpus_model_id
        self.batch_size = batch_size

        label = corpus_model_id
        if self.query_model_id != corpus_model_id:
            label = f"{corpus_model_id} (queries: {self.query_model_id})"

        from mteb.models import ModelMeta
        self.mteb_model_meta = ModelMeta(
            name=f"voyage/{label}",
            revision="1",
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=32000,
            embed_dim=VOYAGE_OUTPUT_DIM,
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

        q_label = self.query_model_id if self.query_model_id != corpus_model_id else "same"
        print(f"  Provider: Voyage AI (corpus: {corpus_model_id}, queries: {q_label}, dim: {VOYAGE_OUTPUT_DIM})")

    def _embed_batch(self, texts: List[str], model_id: str, input_type: str) -> List[List[float]]:
        max_retries = 8
        for attempt in range(max_retries):
            try:
                result = self.client.embed(
                    texts,
                    model=model_id,
                    input_type=input_type,
                    output_dimension=VOYAGE_OUTPUT_DIM,
                )
                return result.embeddings
            except Exception as e:
                err = str(e).lower()
                is_retriable = (
                    "throttl" in err or "rate" in err or "too many" in err
                    or "429" in err or "timeout" in err or "timed out" in err
                )
                if is_retriable and attempt < max_retries - 1:
                    delay = min(5 * (2 ** attempt), 120)
                    print(f"  Rate limited, retrying in {delay}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(delay)
                else:
                    raise

    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, **kwargs) -> np.ndarray:
        from mteb.types import PromptType

        sentences = _extract_texts(inputs)

        if prompt_type == PromptType.query:
            model_id = self.query_model_id
            input_type = "query"
        else:
            model_id = self.corpus_model_id
            input_type = "document"

        all_embeddings: List[List[float]] = []
        for batch in iter_batched(sentences, batch_size=self.batch_size):
            vecs = self._embed_batch(list(batch), model_id=model_id, input_type=input_type)
            all_embeddings.extend(vecs)

        return np.array(all_embeddings, dtype=np.float32)

    def similarity(self, e1: np.ndarray, e2: np.ndarray) -> torch.Tensor:
        return _cosine_similarity(e1, e2)

    def similarity_pairwise(self, e1: np.ndarray, e2: np.ndarray) -> torch.Tensor:
        return torch.diag(_cosine_similarity(e1, e2))


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_model(
    model,
    task_name: str,
    output_dir: Path,
) -> Dict[str, float]:
    """Run MTEB evaluation for a single model on the given task."""
    import mteb
    from mteb import ResultCache

    tasks = mteb.get_tasks(tasks=[task_name])
    if not tasks:
        raise ValueError(f"MTEB task not found: {task_name!r}")

    print(f"  Running {task_name}...")
    model_result = mteb.evaluate(
        model=model,
        tasks=[tasks[0]],
        cache=ResultCache(cache_path=output_dir),
        overwrite_strategy="only-missing",
        show_progress_bar=False,
        raise_error=False,
    )

    if model_result.exceptions:
        raise RuntimeError(str(model_result.exceptions[0]))
    if not model_result.task_results:
        raise RuntimeError("No task results returned.")

    task_result = model_result.task_results[0]
    all_scores = list(task_result.scores.values())
    raw_scores = all_scores[0][0] if all_scores and all_scores[0] else {}

    scores = {
        k: float(v)
        for k, v in raw_scores.items()
        if isinstance(v, (int, float, np.integer, np.floating))
    }

    ndcg10 = scores.get("main_score", scores.get("ndcg_at_10", 0.0))
    print(f"  nDCG@10: {ndcg10:.4f}")
    return scores


# ============================================================================
# Model factory
# ============================================================================
def build_model(name: str, args):
    """Instantiate the model wrapper for the given model name."""
    if name == "openai":
        return OpenAIEmbedder(model_id=args.openai_model_id)
    if name == "openai+adapter":
        return OpenAIEmbedderWithAdapter(
            model_id=args.openai_model_id,
            adapter_path=Path(args.adapter),
        )
    if name == "voyage-4-large":
        return VoyageEmbedder(corpus_model_id="voyage-4-large", query_model_id="voyage-4-large")
    if name == "voyage-4":
        return VoyageEmbedder(corpus_model_id="voyage-4-large", query_model_id="voyage-4")
    if name == "voyage-4-lite":
        return VoyageEmbedder(corpus_model_id="voyage-4-large", query_model_id="voyage-4-lite")
    raise ValueError(f"Unknown model: {name}")


# ============================================================================
# CLI + main
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="External benchmark: evaluate models on ARCChallenge and BarExamQA"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=BENCHMARK_TASKS,
        default=BENCHMARK_TASKS,
        help=f"Tasks to run (default: {' '.join(BENCHMARK_TASKS)})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES + ["all"],
        default=["all"],
        help=f"Models to evaluate (default: all). Choices: {', '.join(MODEL_CHOICES)}, all",
    )
    parser.add_argument(
        "--openai-model-id",
        type=str,
        default="text-embedding-3-large",
        help="OpenAI model ID (default: text-embedding-3-large)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=str(DEFAULT_ADAPTER_PATH),
        help=f"Adapter checkpoint path (default: {DEFAULT_ADAPTER_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: output/mteb_results.json)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_tasks = args.tasks
    run_models: List[str] = (
        MODEL_CHOICES if "all" in args.models else list(dict.fromkeys(args.models))
    )

    print("=" * 64)
    print("MTEB External Benchmark")
    print("=" * 64)
    print(f"  Tasks   : {', '.join(run_tasks)}")
    print(f"  Models  : {', '.join(run_models)}")
    print("  Metric  : nDCG@10")
    print()

    # {task: {model: {scores}}}
    all_results: Dict[str, Any] = {
        "benchmark": "MTEB External Benchmark",
        "tasks": run_tasks,
        "metric": "ndcg_at_10",
        "results": {},
    }

    for task_name in run_tasks:
        safe_task = task_name.lower().replace(" ", "_")
        cache_dir = OUTPUT_DIR / f"mteb_{safe_task}_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        all_results["results"][task_name] = {}

        for model_name in run_models:
            print(f"{'=' * 64}")
            print(f"{task_name} / {model_name}")
            print(f"{'=' * 64}")

            try:
                model = build_model(model_name, args)
                scores = evaluate_model(model, task_name=task_name, output_dir=cache_dir)
                ndcg10 = scores.get("main_score", scores.get("ndcg_at_10", 0.0))
                all_results["results"][task_name][model_name] = {
                    "ndcg_at_10": ndcg10,
                    "all_scores": scores,
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results["results"][task_name][model_name] = {
                    "ndcg_at_10": None,
                    "error": str(e),
                }

            print()

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print("=" * 64)
    print("Results - nDCG@10")
    print("=" * 64)

    header_tasks = [t for t in run_tasks if t in all_results["results"]]
    print(f"  {'Model':<30}", end="")
    for t in header_tasks:
        print(f" {t:>15}", end="")
    print()
    print(f"  {'-' * (30 + 16 * len(header_tasks))}")

    for model_name in run_models:
        print(f"  {model_name:<30}", end="")
        for t in header_tasks:
            info = all_results["results"][t].get(model_name, {})
            score = info.get("ndcg_at_10")
            if score is not None:
                print(f" {score:>15.4f}", end="")
            else:
                print(f" {'ERROR':>15}", end="")
        print()

    # Adapter delta
    if "openai" in run_models and "openai+adapter" in run_models:
        print()
        for t in header_tasks:
            base = (all_results["results"][t].get("openai") or {}).get("ndcg_at_10")
            adapted = (all_results["results"][t].get("openai+adapter") or {}).get("ndcg_at_10")
            if base is not None and adapted is not None:
                delta = adapted - base
                sign = "+" if delta >= 0 else ""
                print(f"  Adapter delta on {t}: {sign}{delta:.4f} ({sign}{delta * 100:.2f}pp)")

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    if args.output:
        output_path = Path(args.output)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "mteb_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved to: {output_path}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
