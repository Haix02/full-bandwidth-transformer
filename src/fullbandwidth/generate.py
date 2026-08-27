"""Generate text with latent feedback (paper Listing 2) or standard decoding."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import FullBandwidthTransformer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/fbt.pt"))
    parser.add_argument("--prompt", type=str, default="To be")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--standard", action="store_true", help="decode without latent feedback")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = bundle["config"]
    model = FullBandwidthTransformer(
        vocab_size=256,
        dim=cfg["dim"],
        n_layer=cfg["layers"],
        n_head=cfg["heads"],
        max_len=max(cfg["length"], args.tokens + 16),
    ).to(device)
    model.load_state_dict(bundle["model"])

    prompt = torch.tensor([list(args.prompt.encode("utf-8"))], dtype=torch.long, device=device)
    out = model.generate(
        prompt,
        max_new=args.tokens,
        temperature=args.temperature,
        latent_feedback=not args.standard,
    )
    text = bytes(out[0].tolist()).decode("utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    main()
