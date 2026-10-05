# SamEngine

LLM training built from first principles in PyTorch, scaling from a single GPU
to 3D parallelism (FSDP + tensor + pipeline), with every parallelism strategy
benchmarked against the single-GPU baseline. A GPT model serves as the
baseline; DeepSeek-V3 (multi-head latent attention, mixture of experts) is the
target architecture.

> **Status: work in progress.** The GPT baseline model is done; the single-GPU
> training loop is being built. Parallelism and benchmarks follow.

## Components

| Component | Status |
|---|---|
| GPT baseline: RoPE, grouped-query attention, RMSNorm (pre-norm), GELU MLP, GPT-2 init | ✅ done, tested |
| Single-GPU training: bf16, grad accumulation, clipping, warmup + cosine LR, MFU / memory logging | 🚧 in progress |
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
```

## Results

Benchmarks will be added as each parallelism strategy lands. Every row will
record hardware, model size, global batch size, tokens/s, MFU, and peak memory per GPU.

Hardware: 2× NVIDIA RTX 6000 Ada (48 GB).

| Config | GPUs | Tokens/s | MFU | Peak memory / GPU |
|---|---|---|---|---|
| Single-GPU baseline | 1 | — | — | — |

## References

- Umar Jamil, *Building a distributed training framework from first principles*
  ([video](https://www.youtube.com/watch?v=XoGvCBRnwLs), [torchfeather](https://github.com/hkproj/torchfeather))
