import torch

from fullbandwidth.fusion import LatentFeedbackFusion, fuse_sequence, prefix_mixin, shift_right
from fullbandwidth.model import FullBandwidthTransformer


def test_fusion_has_no_additive_shortcut():
    fusion = LatentFeedbackFusion(16)
    e = torch.randn(2, 8, 16)
    zeros = torch.zeros_like(e)
    out = fusion(zeros, e)
    assert torch.allclose(out, torch.zeros_like(out))


def test_shift_right_aligns_previous_state():
    h = torch.arange(12, dtype=torch.float).view(1, 4, 3)
    shifted = shift_right(h)
    assert shifted.shape == h.shape
    assert torch.equal(shifted[:, 0], torch.zeros(1, 3))
    assert torch.equal(shifted[:, 1], h[:, 0])
    assert torch.equal(shifted[:, 3], h[:, 2])


def test_fuse_sequence_keeps_first_position_plain():
    fusion = LatentFeedbackFusion(8)
    e = torch.randn(1, 5, 8)
    h = torch.randn(1, 5, 8)
    fused = fuse_sequence(fusion, h, e)
    assert torch.equal(fused[:, 0], e[:, 0])
    assert fused.shape == e.shape


def test_prefix_mixin_cut():
    fused = torch.ones(1, 6, 4)
    embedding = torch.zeros(1, 6, 4)
    mixed = prefix_mixin(fused, embedding, cut=2)
    assert torch.equal(mixed[:, :2], embedding[:, :2])
    assert torch.equal(mixed[:, 2:], fused[:, 2:])


def test_multipass_loss_backprops():
    torch.manual_seed(0)
    model = FullBandwidthTransformer(vocab_size=32, dim=32, n_layer=2, n_head=4, max_len=16)
    tokens = torch.randint(0, 32, (2, 12))
    loss, stats = model.multipass_loss(tokens, passes=3, mix=True)
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert loss.ndim == 0 and loss.item() > 0
    assert "loss_pass1" in stats and "loss_pass3" in stats
    assert sum(grads) > 0


def test_generate_grows_and_feedback_changes_path():
    torch.manual_seed(1)
    model = FullBandwidthTransformer(vocab_size=32, dim=32, n_layer=2, n_head=4, max_len=24)
    prompt = torch.randint(0, 32, (1, 4))
    a = model.generate(prompt, max_new=6, temperature=0.0, latent_feedback=True)
    torch.manual_seed(1)
    b = model.generate(prompt, max_new=6, temperature=0.0, latent_feedback=False)
    assert a.shape == (1, 10)
    assert b.shape == (1, 10)
    assert not torch.equal(a, b)
