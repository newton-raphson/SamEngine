import math
import pytest
import torch
import torch.nn.functional as F
from distributedtraining.model.gpt import GPTConfig, rope_cache, apply_rope, RMSNorm, Attention, MLP, Block, GPT, param_count

TINY = GPTConfig(vocab_size=97, d_model=64, n_layers=3, n_heads=8, n_kv_heads=2, max_seq_len=32)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def ref_rope(x, base=10000.0):
    B, H, T, d = x.shape
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    ang = torch.arange(T, dtype=torch.float64)[:, None] * theta[None]
    z = torch.view_as_complex(x.double().reshape(B, H, T, d // 2, 2).contiguous())
    return torch.view_as_real(z * torch.polar(torch.ones_like(ang), ang)).reshape(B, H, T, d).to(x.dtype)


def ref_attn(m, x, H, Hkv):
    B, T, D = x.shape
    dh = D // H
    q = ref_rope(m.wq(x).view(B, T, H, dh).transpose(1, 2))
    k = ref_rope(m.wk(x).view(B, T, Hkv, dh).transpose(1, 2))
    v = m.wv(x).view(B, T, Hkv, dh).transpose(1, 2)
    k, v = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return m.wo(o.transpose(1, 2).reshape(B, T, D))


def test_rope_cache_shapes():
    cos, sin = rope_cache(10, 16)
    assert cos.shape == sin.shape == (10, 8)
    assert torch.allclose(cos[0], torch.ones(8)) and torch.allclose(sin[0], torch.zeros(8))


def test_rope_matches_complex_reference():
    x = torch.randn(2, 3, 12, 16)
    cos, sin = rope_cache(12, 16)
    torch.testing.assert_close(apply_rope(x, cos, sin), ref_rope(x), atol=1e-5, rtol=1e-5)


def test_rope_score_depends_only_on_offset():
    d, T = 16, 20
    cos, sin = rope_cache(T, d)
    q = torch.randn(1, 1, 1, d).expand(1, 1, T, d).clone()
    k = torch.randn(1, 1, 1, d).expand(1, 1, T, d).clone()
    rq, rk = apply_rope(q, cos, sin)[0, 0], apply_rope(k, cos, sin)[0, 0]
    torch.testing.assert_close((rq[3] * rk[1]).sum(), (rq[15] * rk[13]).sum(), atol=1e-4, rtol=1e-4)


def test_rmsnorm():
    n = RMSNorm(32)
    with torch.no_grad():
        n.weight.copy_(torch.randn(32))
    x = torch.randn(4, 7, 32)
    torch.testing.assert_close(n(x), F.rms_norm(x, (32,), n.weight, 1e-6), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("n_kv_heads", [8, 2])
def test_attention(n_kv_heads):
    m = Attention(GPTConfig(d_model=64, n_heads=8, n_kv_heads=n_kv_heads))
    assert m.wk.out_features == n_kv_heads * 8
    x = torch.randn(2, 10, 64)
    cos, sin = rope_cache(10, 8)
    torch.testing.assert_close(m(x, cos, sin), ref_attn(m, x, 8, n_kv_heads), atol=1e-5, rtol=1e-5)


def test_mlp():
    m = MLP(GPTConfig(d_model=32))
    x = torch.randn(3, 5, 32)
    torch.testing.assert_close(m(x), m.fc2(F.gelu(m.fc1(x))))
    assert m.fc1.out_features == 128 and m.fc1.bias is None


def test_block_residuals():
    b = Block(GPTConfig(d_model=64, n_heads=8, n_kv_heads=2))
    with torch.no_grad():
        for n in (b.norm1, b.norm2):
            n.weight.copy_(torch.rand(64) + 0.5)
    x = torch.randn(2, 10, 64)
    cos, sin = rope_cache(10, 8)
    h = x + b.attn(b.norm1(x), cos, sin)
    torch.testing.assert_close(b(x, cos, sin), h + b.mlp(b.norm2(h)), atol=1e-5, rtol=1e-5)


def test_gpt_output_shape():
    assert GPT(TINY)(torch.randint(0, 97, (2, 16))).shape == (2, 16, 97)


def test_gpt_is_causal():
    m = GPT(TINY).eval()
    idx = torch.randint(0, 97, (1, 16))
    idx2 = idx.clone()
    idx2[0, 10:] = (idx2[0, 10:] + 1) % 97
    with torch.no_grad():
        torch.testing.assert_close(m(idx)[:, :10], m(idx2)[:, :10])


@pytest.mark.parametrize("cfg", [
    TINY,
    GPTConfig(),
    GPTConfig(vocab_size=50257, d_model=768, n_layers=12, n_heads=12, n_kv_heads=12, max_seq_len=1024),
])
def test_param_count(cfg):
    assert param_count(cfg) == sum(p.numel() for p in GPT(cfg).parameters())


def test_gpt2_init():
    cfg = GPTConfig(vocab_size=4096, d_model=256, n_layers=8, n_heads=8, n_kv_heads=8)
    m = GPT(cfg)
    std = lambda w: w.detach().std().item()
    assert abs(std(m.tok_emb.weight) - 0.02) < 0.002
    assert abs(std(m.blocks[0].attn.wq.weight) - 0.02) < 0.002
    want = 0.02 / math.sqrt(2 * cfg.n_layers)
    for w in (m.blocks[0].attn.wo.weight, m.blocks[0].mlp.fc2.weight):
        assert abs(std(w) - want) < 0.15 * want
    assert torch.all(m.norm_f.weight == 1)


def test_overfits_one_batch():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = GPT(TINY).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    idx = torch.randint(0, 97, (4, 33), device=dev)
    for _ in range(200):
        loss = F.cross_entropy(m(idx[:, :-1]).reshape(-1, 97), idx[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
