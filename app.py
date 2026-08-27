"""Vercel / local demo. No PyTorch — fusion is Eq. (4) in plain Python."""

from __future__ import annotations

import math
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Full-bandwidth Transformer", version="0.1.0")
PAGE = Path(__file__).with_name("index.html")


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def glu_cross(hidden: list[float], embedding: list[float], w_u: list[list[float]], w_g: list[list[float]]) -> list[float]:
    """e ⊗ h = (W_U h) ⊙ σ(W_G e)  — Wang et al. 2026, Eq. (4)."""
    if len(hidden) != len(embedding):
        raise ValueError("hidden and embedding must have the same width")
    dim = len(hidden)
    value = [sum(w_u[i][j] * hidden[j] for j in range(dim)) for i in range(dim)]
    gate = [sigmoid(sum(w_g[i][j] * embedding[j] for j in range(dim))) for i in range(dim)]
    return [v * g for v, g in zip(value, gate)]


class FuseRequest(BaseModel):
    hidden: list[float] = Field(..., min_length=1, max_length=32)
    embedding: list[float] = Field(..., min_length=1, max_length=32)
    w_u: list[list[float]]
    w_g: list[list[float]]


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE.read_text(encoding="utf-8")


@app.post("/api/fuse")
def fuse(req: FuseRequest) -> dict:
    if len(req.hidden) != len(req.embedding):
        return {"error": "widths must match"}
    dim = len(req.hidden)
    if len(req.w_u) != dim or len(req.w_g) != dim:
        return {"error": "weight rows must match width"}
    out = glu_cross(req.hidden, req.embedding, req.w_u, req.w_g)
    zero_h = glu_cross([0.0] * dim, req.embedding, req.w_u, req.w_g)
    return {
        "fused": out,
        "zero_hidden_is_zero": all(abs(x) < 1e-12 for x in zero_h),
        "equation": "e ⊗ h = W_U h  ⊙  σ(W_G e)",
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}
