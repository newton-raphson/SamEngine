#!/usr/bin/env bash
# Single-GPU baseline: eager vs torch.compile in bf16, plus a short fp32 run for the precision comparison.
# The bf16 runs go on separate GPUs in parallel. Results land in runs/*.json.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

( CUDA_VISIBLE_DEVICES=0 python scripts/bench_single_gpu.py eager_bf16
  CUDA_VISIBLE_DEVICES=0 python scripts/bench_single_gpu.py eager_fp32 bf16=false max_steps=200 ) &
CUDA_VISIBLE_DEVICES=1 python scripts/bench_single_gpu.py compile_bf16 compile=true &
wait
