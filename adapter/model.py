"""Adapter model definitions.

FullRankAdapter : nn.Linear(d, d) with identity init. ~9.4M params at d=3072.
LowRankAdapter  : residual low-rank correction  q + up(down(q)).
                  up.weight initialised to zero so the adapter starts as an exact
                  identity and learns a low-rank perturbation (same principle as LoRA).
                  ~789K params at d=3072, r=128.

Both forward() methods return the adapted (un-normalised) embedding.
Normalisation is applied in the training loop before computing triplet loss.

Utilities:
    build_adapter(adapter_type, emb_dim, low_rank_dim)  -> nn.Module
    save_adapter(adapter, path, config_dict)             saves self-describing checkpoint
    load_adapter(path)                                   -> nn.Module (ready to use)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Adapter classes
# ---------------------------------------------------------------------------

class FullRankAdapter(nn.Module):
    """Full d×d linear adapter.

    W is initialised to the identity matrix and bias to zeros so the
    adapter begins as an exact pass-through and learns a dense correction.
    ~9.4M trainable parameters at d=3072.
    """

    def __init__(self, d: int = 3072) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.linear(q)


class LowRankAdapter(nn.Module):
    """Residual low-rank adapter:  q_out = q + up(down(q))

    down : d → r   (no bias, Kaiming uniform init)
    up   : r → d   (with bias; weight initialised to zeros)

    Because up.weight starts at 0 the adapter is an exact identity at
    initialisation and learns a rank-r correction — same principle as LoRA.
    ~789K trainable parameters at d=3072, r=128.
    """

    def __init__(self, d: int = 3072, r: int = 128) -> None:
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.up   = nn.Linear(r, d, bias=True)
        nn.init.kaiming_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return q + self.up(self.down(q))


# ---------------------------------------------------------------------------
# Factory & checkpoint utilities
# ---------------------------------------------------------------------------

def build_adapter(adapter_type: str, emb_dim: int, low_rank_dim: int) -> nn.Module:
    """Create an adapter from config values."""
    if adapter_type == "full_rank":
        return FullRankAdapter(d=emb_dim)
    if adapter_type == "low_rank":
        return LowRankAdapter(d=emb_dim, r=low_rank_dim)
    raise ValueError(
        f"Unknown adapter_type {adapter_type!r}. Choose 'full_rank' or 'low_rank'."
    )


def save_adapter(adapter: nn.Module, path, config_dict: dict) -> None:
    """Save adapter weights + config as a self-describing checkpoint.

    The saved file contains everything needed to reconstruct the adapter
    without access to the original config:
        {"state_dict": ..., "adapter_type": ..., "low_rank_dim": ..., "emb_dim": ...}
    """
    torch.save({"state_dict": adapter.state_dict(), **config_dict}, path)


def load_adapter(path) -> nn.Module:
    """Load a self-describing adapter checkpoint.

    Returns the adapter in eval mode, on CPU. Move to device with .to(device).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    adapter = build_adapter(
        adapter_type=ckpt["adapter_type"],
        emb_dim=ckpt["emb_dim"],
        low_rank_dim=ckpt.get("low_rank_dim", 128),
    )
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    return adapter
