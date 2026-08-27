"""Byte-level language model training with the paper's scheduled multi-pass loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import FullBandwidthTransformer

TINY = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub.
"""


def sample_batch(data: torch.Tensor, batch: int, length: int, device: torch.device) -> torch.Tensor:
    starts = torch.randint(0, data.numel() - length - 1, (batch,))
    return torch.stack([data[s : s + length] for s in starts]).to(device)


def choose_passes(step: int, warmup: int, three_pass_prob: float) -> int:
    """Progressive schedule: k=1, then k=2, with a small mix of k=3 (paper Fig. 3)."""
    if step < warmup:
        return 1
    if torch.rand(1).item() < three_pass_prob:
        return 3
    return 2


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a tiny Full-bandwidth Transformer")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--three-pass-prob", type=float, default=0.03)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/fbt.pt"))
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(list(TINY.encode("utf-8")), dtype=torch.long)
    model = FullBandwidthTransformer(
        vocab_size=256,
        dim=args.dim,
        n_layer=args.layers,
        n_head=args.heads,
        max_len=args.length,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(args.steps):
        tokens = sample_batch(data, args.batch, args.length, device)
        passes = choose_passes(step, args.warmup, args.three_pass_prob)
        loss, stats = model.multipass_loss(tokens, passes=passes)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step:04d}  k={passes}  " + "  ".join(f"{k}={v:.3f}" for k, v in stats.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(args)}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
