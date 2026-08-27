"""A small decoder-only stack used as f_θ in the paper.

The blocks are ordinary GPT-style. The architecture change is that their
inputs can be latent-feedback fusions instead of token embeddings.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .fusion import LatentFeedbackFusion, fuse_sequence, prefix_mixin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        if dim % n_head != 0:
            raise ValueError("dim must divide n_head")
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        out = attn.transpose(1, 2).contiguous().view(batch, length, dim)
        return self.dropout(self.proj(out))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = hidden_mult * dim
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head, dropout)
        self.n2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class FullBandwidthTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        n_layer: int = 4,
        n_head: int = 4,
        max_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(dim, n_head, dropout) for _ in range(n_layer))
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.fusion = LatentFeedbackFusion(dim)
        self.head.weight = self.tok.weight

    def token_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tok(tokens)

    def with_pos(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return self.drop(x + self.pos(positions))

    def stack(self, x: torch.Tensor) -> torch.Tensor:
        x = self.with_pos(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token embeddings only (no positions). Used as e in the paper's glu_cross."""
        return self.token_embed(tokens)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)

    def forward_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.stack(self.embed(tokens))

    def ntp_loss(self, hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.logits(hidden[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))

    def multipass_loss(
        self,
        tokens: torch.Tensor,
        passes: int = 2,
        mix: bool = True,
        lambda_feedback: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Eq. (12) / Listing 1: pass 1 is ordinary NTP; later passes fuse shifted states."""
        if passes < 1:
            raise ValueError("passes must be >= 1")
        embedding = self.embed(tokens)
        hidden = self.stack(embedding)
        loss = self.ntp_loss(hidden, tokens)
        stats = {"loss_pass1": float(loss.detach())}
        if passes == 1:
            stats["loss"] = stats["loss_pass1"]
            return loss, stats

        feedback = []
        for step in range(passes - 1):
            fused = fuse_sequence(self.fusion, hidden, embedding)
            if mix:
                fused = prefix_mixin(fused, embedding)
            hidden = self.stack(fused)
            step_loss = self.ntp_loss(hidden, tokens)
            feedback.append(step_loss)
            stats[f"loss_pass{step + 2}"] = float(step_loss.detach())

        loss = loss + lambda_feedback * (sum(feedback) / len(feedback))
        stats["loss"] = float(loss.detach())
        return loss, stats

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new: int = 64,
        temperature: float = 1.0,
        latent_feedback: bool = True,
    ) -> torch.Tensor:
        """Listing 2. Prefill on plain embeddings, then fused (or standard) decode.

        Tiny reference: recomputes the full sequence each step instead of a KV cache.
        """
        self.eval()
        tokens = prompt.clone()
        inputs = self.embed(tokens)
        hidden = self.stack(inputs)
        for _ in range(max_new):
            if tokens.size(1) >= self.max_len:
                break
            logits = self.logits(hidden[:, -1])
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                nxt = torch.multinomial(torch.softmax(logits / max(temperature, 1e-6), dim=-1), 1).squeeze(-1)
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
            emb = self.embed(nxt.unsqueeze(1))
            if latent_feedback:
                fused = self.fusion(hidden[:, -1], emb.squeeze(1)).unsqueeze(1)
            else:
                fused = emb
            inputs = torch.cat([inputs, fused], dim=1)
            hidden = self.stack(inputs)
        return tokens
