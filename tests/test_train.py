import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from distributedtraining.model.gpt import GPT, GPTConfig
from distributedtraining.train import (
    TrainConfig, get_batch, lr_at, configure_optimizer, flops_per_token,
    accumulate_grads, train_step, evaluate, train,
)

TINY = GPTConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_get_batch():
    data = (np.arange(1000) % 251).astype(np.uint8)
    x, y = get_batch(data, 8, 16, "cpu", np.random.default_rng(0))
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == y.dtype == torch.int64
    assert torch.equal(y, (x + 1) % 251), "y must be x shifted by one"
    x2, _ = get_batch(data, 8, 16, "cpu", np.random.default_rng(1))
    assert not torch.equal(x, x2)
    small = np.arange(17, dtype=np.uint8)
    for s in range(20):
        x, _ = get_batch(small, 4, 16, "cpu", np.random.default_rng(s))
        assert torch.equal(x[0], torch.arange(16)), "window runs past the end of data"


@pytest.mark.parametrize("step,want", [(0, 0.1), (4, 0.5), (9, 1.0), (10, 1.0), (55, 0.55), (100, 0.1), (150, 0.1)])
def test_lr_schedule(step, want):
    tc = TrainConfig(warmup_steps=10, max_steps=100, max_lr=1.0, min_lr=0.1)
    assert lr_at(step, tc) == pytest.approx(want, abs=1e-9)


def test_optimizer_groups():
    m = GPT(TINY)
    opt = configure_optimizer(m, TrainConfig(weight_decay=0.1, max_lr=3e-4), "cpu")
    assert isinstance(opt, torch.optim.AdamW)
    wd = {}
    for g in opt.param_groups:
        assert g["betas"] == (0.9, 0.95) and g["lr"] == 3e-4
        for p in g["params"]:
            assert id(p) not in wd, "parameter in two groups"
            wd[id(p)] = g["weight_decay"]
    for name, p in m.named_parameters():
        assert wd.get(id(p)) == (0.1 if p.dim() >= 2 else 0.0), name


@pytest.mark.parametrize("cfg,T", [
    (TINY, 32),
    (GPTConfig(vocab_size=50257, d_model=768, n_layers=12, n_heads=12, n_kv_heads=4), 1024),
])
def test_flops_per_token(cfg, T):
    m = GPT(cfg)
    n = sum(p.numel() for name, p in m.named_parameters() if p.dim() >= 2 and not name.startswith("tok_emb"))
    assert flops_per_token(cfg, T) == 6 * n + 12 * cfg.n_layers * cfg.d_model * T


def test_grad_accumulation_matches_big_batch():
    m = GPT(TINY)
    x, y = torch.randint(0, 256, (8, 32)), torch.randint(0, 256, (8, 32))
    grads = lambda: {n: p.grad.clone() for n, p in m.named_parameters()}

    m.zero_grad(set_to_none=True)
    full = accumulate_grads(m, [(x, y)], bf16=False)
    g_full = grads()
    m.zero_grad(set_to_none=True)
    acc = accumulate_grads(m, [(x[i:i + 2], y[i:i + 2]) for i in range(0, 8, 2)], bf16=False)
    g_acc = grads()

    assert isinstance(acc, float)
    assert acc == pytest.approx(full, abs=1e-5)
    assert full == pytest.approx(F.cross_entropy(m(x).reshape(-1, 256), y.reshape(-1)).item(), abs=1e-5)
    for n in g_full:
        torch.testing.assert_close(g_acc[n], g_full[n], atol=1e-5, rtol=1e-4)


def test_train_step():
    m = GPT(TINY)
    tc = TrainConfig(warmup_steps=10, max_steps=100, max_lr=1e-3, min_lr=1e-4, grad_clip=1e-3, bf16=False)
    opt = configure_optimizer(m, tc, "cpu")
    before = [p.detach().clone() for p in m.parameters()]
    mb = [(torch.randint(0, 256, (2, 32)), torch.randint(0, 256, (2, 32))) for _ in range(2)]

    loss, gn, lr = train_step(m, opt, mb, tc, step=4)

    assert all(isinstance(v, float) for v in (loss, gn, lr))
    assert lr == pytest.approx(lr_at(4, tc)) and all(g["lr"] == lr for g in opt.param_groups)
    assert gn > tc.grad_clip, "return the pre-clip norm"
    post = torch.norm(torch.stack([p.grad.norm() for p in m.parameters()])).item()
    assert post <= tc.grad_clip * 1.01, "grads not clipped"
    assert any(not torch.equal(a, p) for a, p in zip(before, m.parameters()))


def test_evaluate():
    m = GPT(TINY)
    data = np.random.default_rng(0).integers(0, 256, 5000).astype(np.uint8)
    v = evaluate(m, data, TrainConfig(batch_size=4, seq_len=32, eval_batches=3, bf16=False), "cpu",
                 np.random.default_rng(0))
    assert isinstance(v, float) and abs(v - math.log(256)) < 0.5
    assert m.training
    assert all(p.grad is None for p in m.parameters())


def test_train_end_to_end(tmp_path):
    text = np.frombuffer(b"the quick brown fox jumps over the lazy dog. " * 400, dtype=np.uint8)
    text.tofile(tmp_path / "train.bin")
    text[:2000].tofile(tmp_path / "val.bin")
    tc = TrainConfig(data_dir=str(tmp_path), batch_size=8, seq_len=32, grad_accum=2, max_steps=60,
                     warmup_steps=5, max_lr=3e-3, min_lr=3e-4, log_every=10, eval_every=30,
                     eval_batches=2, out=str(tmp_path / "ckpt.pt"))
    hist = train(TINY, tc)

    assert {"step", "loss", "lr", "grad_norm", "tok_per_s", "mfu", "peak_mem_gb"} <= set(hist[0])
    assert hist[-1]["step"] == 59
    assert hist[-1]["loss"] < 0.5 * hist[0]["loss"]
    assert all(h["tok_per_s"] > 0 and 0 <= h["mfu"] < 1 for h in hist)
    ck = torch.load(tc.out, weights_only=False)
    assert {"model", "optimizer", "step", "model_cfg", "train_cfg"} <= set(ck)
    GPT(TINY).load_state_dict(ck["model"])
