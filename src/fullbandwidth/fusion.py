"""Gated fusion from Wang et al. 2026, Eq. (4).

    e ⊗ h = W_U h  ⊙  σ(W_G e)

The hidden state is the value pathway; the token embedding is only a gate.
If the model drops h, the input itself vanishes — there is no additive shortcut
back to a plain token embedding.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LatentFeedbackFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w_u = nn.Linear(dim, dim, bias=False)
        self.w_g = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return self.w_u(hidden) * torch.sigmoid(self.w_g(embedding))


def shift_right(hidden: torch.Tensor) -> torch.Tensor:
    """Pad a zero state at t=0 and drop the last position: h_{t-1} aligned with e_t."""
    zeros = torch.zeros_like(hidden[:, :1])
    return torch.cat([zeros, hidden[:, :-1]], dim=1)


def fuse_sequence(
    fusion: LatentFeedbackFusion,
    hidden: torch.Tensor,
    embedding: torch.Tensor,
) -> torch.Tensor:
    """Build inputs e_1, e_2⊗h_1, e_3⊗h_2, … as in Eq. (8). Position 0 stays a plain embed."""
    fused = fusion(shift_right(hidden), embedding)
    fused = fused.clone()
    fused[:, 0] = embedding[:, 0]
    return fused


def prefix_mixin(
    fused: torch.Tensor,
    embedding: torch.Tensor,
    cut: int | None = None,
) -> torch.Tensor:
    """Keep a random prefix as plain embeddings so training matches prompt-then-generate."""
    batch, length, _ = fused.shape
    if cut is None:
        cut = int(torch.randint(0, length, (1,), device=fused.device).item())
    cut = max(0, min(cut, length))
    if cut == 0:
        return fused
    mixed = fused.clone()
    mixed[:, :cut] = embedding[:, :cut]
    return mixed
