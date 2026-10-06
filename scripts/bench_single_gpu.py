"""
Runs train() for one TrainConfig variant and writes throughput, memory and loss to runs/{name}.json.

    python scripts/bench_single_gpu.py compile_bf16 compile=true
    python scripts/bench_single_gpu.py eager_fp32 bf16=false max_steps=200
"""
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, fields
import numpy as np
import torch
from distributedtraining.model.gpt import GPT, param_count
from distributedtraining.train import MODEL, TrainConfig, evaluate, load_data, train

WARMUP_STEPS = 50   # excluded from throughput medians: CUDA warmup and torch.compile


def parse_overrides(args):
    types = {f.name: type(f.default) for f in fields(TrainConfig)}
    kw = {}
    for a in args:
        k, v = a.split("=", 1)
        kw[k] = v.lower() in ("1", "true", "yes") if types[k] is bool else types[k](v)
    return kw


def main():
    name, kw = sys.argv[1], parse_overrides(sys.argv[2:])
    os.makedirs("runs", exist_ok=True)
    tc = TrainConfig(out=f"runs/{name}.pt", **kw)

    t0 = time.perf_counter()
    hist = train(MODEL, tc)
    wall = time.perf_counter() - t0

    # final val loss on a fixed rng, so runs are compared on the same batches
    model = GPT(MODEL).cuda()
    model.load_state_dict(torch.load(tc.out, weights_only=False)["model"])
    val = evaluate(model, load_data(tc.data_dir, "val"), tc, "cuda", np.random.default_rng(123))

    steady = [h for h in hist if h["step"] >= WARMUP_STEPS]
    result = {
        "name": name,
        "gpu": torch.cuda.get_device_name(),
        "params": param_count(MODEL),
        "tokens_per_step": tc.batch_size * tc.seq_len * tc.grad_accum,
        "train_cfg": asdict(tc),
        "wall_s": wall,
        "tok_per_s_median": statistics.median(h["tok_per_s"] for h in steady),
        "mfu_median": statistics.median(h["mfu"] for h in steady),
        "peak_mem_gb": max(h["peak_mem_gb"] for h in hist),
        "train_loss_last": hist[-1]["loss"],
        "val_loss": val,
        "history": hist,
    }
    with open(f"runs/{name}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"{name}: {result['tok_per_s_median']:,.0f} tok/s, MFU {result['mfu_median']:.1%}, "
          f"peak {result['peak_mem_gb']:.2f} GB, train {result['train_loss_last']:.3f}, val {val:.3f}, "
          f"{wall:.0f} s")


if __name__ == "__main__":
    main()
