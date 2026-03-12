"""Central configuration for the adapter training pipeline.

All hyperparameters are defined here. Change this file to run different
experiments — no other file needs to be modified.
"""

from pathlib import Path

# Project root: adapter/config.py -> adapter/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATASET_PATH = PROJECT_ROOT / "output/verified/run 3/vision_validated_relaxed.csv"
DB_PATH      = PROJECT_ROOT / "processed/pdf_page_store.sqlite"
DATA_DIR     = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-large"
EMB_DIM         = 3072

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
N_SOFT_NEGS     = 5    # Random soft negatives sampled per query (from full corpus)
RANDOM_SEED     = 42
EMBED_BATCH_SIZE = 100  # OpenAI API batch size for query embedding

# ---------------------------------------------------------------------------
# Train / val / test split (file-level — all queries from the same source
# file stay in one split to prevent information leakage)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # must sum to 1.0 with TRAIN_RATIO + VAL_RATIO

# ---------------------------------------------------------------------------
# Adapter architecture
# ---------------------------------------------------------------------------
ADAPTER_TYPE = "full_rank"   # "low_rank" | "full_rank"
LOW_RANK_DIM = 256          # r for low_rank; ignored for full_rank
#                            # Parameter counts at EMB_DIM=3072:
#                            #   full_rank           -> ~9.4M params
#                            #   low_rank  r=256     -> ~1.57M params
#                            #   low_rank  r=128     -> ~789K  params  (default)
#                            #   low_rank  r=64      -> ~397K  params

# ---------------------------------------------------------------------------
# Training hyperparameters  (from Chroma Research + thesis Notion guide)
# ---------------------------------------------------------------------------
LEARNING_RATE           = 3e-6   # Chroma's most frequently best LR; sweep: {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
MARGIN                  = 0.3    # Triplet loss margin alpha; sweep: {0.3, 1.0, 3.0}
BATCH_SIZE              = 256
NUM_EPOCHS              = 80
WEIGHT_DECAY            = 1e-4
EARLY_STOPPING_PATIENCE = 3      # Stop after this many epochs without val loss improvement
GRAD_CLIP_NORM          = 1.0    # Max gradient norm; set to None to disable

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
TOP_K_EVAL = [1, 5, 10, 20]   # Recall@K values to compute during test evaluation

# ---------------------------------------------------------------------------
# Device  ("auto" -> CUDA > MPS > CPU)
# ---------------------------------------------------------------------------
DEVICE = "auto"

# ---------------------------------------------------------------------------
# Leaderboard
# The metric used to decide whether a new training run is "best" for its
# (adapter_type, low_rank_dim) configuration.
# ---------------------------------------------------------------------------
BEST_METRIC      = "ndcg@10"   # primary comparison metric; any key from TOP_K_EVAL or mrr/map/ndcg@10
LEADERBOARD_PATH = Path(__file__).resolve().parent / "leaderboard.json"
