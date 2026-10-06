# SamEngine

LLM training built from first principles in PyTorch, scaling from a single GPU
to 3D parallelism (FSDP + tensor + pipeline), with every parallelism strategy
benchmarked against the single-GPU baseline. A GPT model serves as the
baseline; DeepSeek-V3 (multi-head latent attention, mixture of experts) is the
target architecture.

> **Status: work in progress.** The GPT baseline model and the single-GPU
> training loop are done and benchmarked. Parallelism follows.

## Components

| Component | Status |
|---|---|
| GPT baseline: RoPE, grouped-query attention, RMSNorm (pre-norm), GELU MLP, GPT-2 init | ✅ done, tested |
| Single-GPU training: bf16, grad accumulation, clipping, warmup + cosine LR, MFU / memory logging | ✅ done, tested, benchmarked |
| Pipeline parallelism: GPipe and 1F1B schedules | ⏳ planned |
| Collectives from scratch: all-reduce, all-gather, reduce-scatter | ⏳ planned |
| Data parallelism: DDP and FSDP | ⏳ planned |
| Tensor parallelism: column / row parallel linear layers | ⏳ planned |
| Combined DP × TP × PP training run | ⏳ planned |
| DeepSeek-V3: MLA, MoE, YaRN (config in `model/model_args.py`) | ⏳ planned |

## Layout

```
src/distributedtraining/
  model/
    gpt.py          GPT baseline and analytic parameter count
    model_args.py   DeepSeek-V3 configuration
  train.py          single-GPU training loop
scripts/
  prepare_shakespeare.py
  bench_single_gpu.py   one benchmark run -> runs/{name}.json
  bench_single_gpu.sh   single-GPU baseline: eager vs compile, bf16 vs fp32
tests/
  test_model.py     checks against independent references (complex-number RoPE, F.rms_norm, SDPA)
  test_train.py
```

## Quickstart

```bash
uv sync
uv run python scripts/prepare_shakespeare.py      # byte-level Tiny Shakespeare -> data/
uv run pytest
uv run python -m distributedtraining.train --max_steps 1000
./scripts/bench_single_gpu.sh                      # single-GPU baseline numbers below
```

## Results

Every row records hardware, model size, global batch size, tokens/s, MFU, and
peak memory per GPU. Parallelism rows will be added as each strategy lands.

Hardware: 2× NVIDIA RTX 6000 Ada (48 GB). MFU is against 364 TFLOPS dense bf16.

### Single-GPU baseline

GPT, 10.8M params (d_model 384, 6 layers, 6 heads, MHA), byte-level Tiny
Shakespeare, seq_len 256, 32 × 4 micro-batches = 32,768 tokens/step,
AdamW, LR 1e-3 → 1e-4 (100 warmup steps). Tokens/s and MFU are medians over
steps ≥ 50. Reproduce with `./scripts/bench_single_gpu.sh`.

| Config | GPUs | Steps | Tokens/s | MFU | Peak memory / GPU | Train loss | Val loss |
|---|---|---|---|---|---|---|---|
| eager, fp32 | 1 | 200 | 241k | 4.7% | 1.65 GB | 1.59 | 1.74 |
| eager, bf16 | 1 | 1000 | 349k | 6.8% | 1.22 GB | 0.52 | 2.28 |
| `torch.compile`, bf16 | 1 | 1000 | 549k | 10.8% | 1.09 GB | 0.52 | 2.29 |

- **bf16** is 1.45× the fp32 throughput with 26% less memory.
- **`torch.compile`** adds 1.57× on top. RoPE uses complex ops, which Inductor
  cannot generate code for, so that part runs unfused.
- **MFU is low** because the model is small: d_model 384 matmuls cannot fill
  the tensor cores, and launch overhead is a large share of a ~60 ms step.
- **The 1000-step runs overfit**: 1000 × 32k tokens is ~33 epochs of the 1M
  training tokens. The 200-step run has the better val loss.

## References

- Umar Jamil, *Building a distributed training framework from first principles*
  ([video](https://www.youtube.com/watch?v=XoGvCBRnwLs), [torchfeather](https://github.com/hkproj/torchfeather))
