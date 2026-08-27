# Full-bandwidth Transformer — open implementation

Wang, Cai, Zhan, Dong, Fan, de Rosa, Pearce, Langford. **Full-bandwidth transformer.** arXiv:2608.08888, 2026.

Microsoft / Johns Hopkins / Princeton. **The paper did not release code.** This repo is a from-scratch reference of the architecture they describe: latent feedback through a gated linear unit, parallel multi-pass training, and fused decoding.

## Why this exists

A standard transformer’s only vertical feedback between decoding steps is the sampled token. The top-layer hidden state is thrown away. The paper widens that channel to the full residual-stream width:

\[
e_t \otimes h_{t-1} = W_U h_{t-1} \odot \sigma(W_G e_t)
\]

The hidden state is the **value**. The token is only a **gate**. There is no additive shortcut back to a plain embedding — if the model ignores \(h\), the input itself is zero.

Training stays parallel: pass 1 is ordinary next-token prediction; later passes shift the previous hidden states one step right, fuse, and re-run the stack (Listing 1). A few percent of 3-pass batches keep the feedback map a contraction at long horizon (Fig. 3).

This is a **tiny** model (CPU minutes) so the mechanism can be audited. It is not a 1B-parameter reproduction of their 400B-token runs.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
```

## Train and generate

```bash
pytest
python -m fullbandwidth.train --steps 80
python -m fullbandwidth.generate --prompt "To be"
python -m fullbandwidth.generate --prompt "To be" --standard   # no latent feedback
```

## Web demo (Vercel)

The site is a FastAPI app in `app.py`. It does **not** install PyTorch. Eq. (4) is evaluated in plain Python so the function stays under Vercel’s bundle limit.

```bash
pip install -e .
uvicorn app:app --reload
```

## Mapping to the paper

| Paper | Code |
|---|---|
| Eq. (4) gated fusion | `LatentFeedbackFusion` |
| Eq. (8) first position plain, later fused | `fuse_sequence` |
| Listing 1 multi-pass train | `FullBandwidthTransformer.multipass_loss` |
| Prefix mixin | `prefix_mixin` |
| Listing 2 fused decode | `FullBandwidthTransformer.generate(..., latent_feedback=True)` |
| 3% three-pass mix | `--three-pass-prob 0.03` |

## What this is not

- Not an official Microsoft artifact.
- Not a claim that a 128-d toy matches their 1B numbers.
- Not a license to ignore the paper’s citation. Please cite arXiv:2608.08888.

## License

MIT for this implementation. The paper remains the authors’ work.
