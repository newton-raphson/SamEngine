"""
Single-GPU training loop: bf16 autocast, gradient accumulation, clipping,
warmup + cosine LR, and per-step throughput / MFU / memory logging.

    python -m distributedtraining.train --max_steps 1000 --compile true
"""
import argparse
import math
import os
import time
from dataclasses import dataclass, asdict, fields
import numpy as np
import torch
import torch.nn.functional as F
from distributedtraining.model.gpt import GPT, GPTConfig,param_count


# RTX 6000 Ada, dense bf16 tensor-core peak (spec sheet). MFU is reported against this.
PEAK_FLOPS = 364e12


@dataclass
class TrainConfig:
    data_dir: str = "data"
    batch_size: int = 32        # micro-batch, in sequences
    seq_len: int = 256
    grad_accum: int = 4         # micro-batches per optimizer step
    max_steps: int = 1000
    warmup_steps: int = 100
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    bf16: bool = True
    compile: bool = False
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 20
    seed: int = 0
    out: str = "ckpt.pt"


MODEL = GPTConfig(vocab_size=256, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6, max_seq_len=256)


def load_data(data_dir, split):
    """uint8 token ids for split 'train' or 'val', as a read-only memmap."""
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint8, mode="r")


def get_batch(data, batch_size, seq_len, device, rng):
    """
    Samples batch_size windows at random starts i with i + seq_len + 1 <= len(data).
    Returns x = data[i : i+seq_len], y = data[i+1 : i+seq_len+1] as int64 (batch_size, seq_len) on device.
    """
    idx_ = rng.integers(low=0,high=len(data)-seq_len,size=batch_size)
    x_list = np.array([data[i:i+seq_len] for i in idx_])
    y_list = np.array([data[i+1:i+seq_len+1] for i in idx_])
    # 
    return torch.tensor(x_list,dtype=torch.int64,device=device),torch.tensor(y_list,dtype=torch.int64,device=device)



def lr_at(step, tc: TrainConfig):
    """
    step < warmup_steps:  max_lr * (step + 1) / warmup_steps
    step >= max_steps:    min_lr
    otherwise:            r = (step - warmup_steps) / (max_steps - warmup_steps)
                          min_lr + 0.5 * (1 + cos(pi * r)) * (max_lr - min_lr)
    """
    if step<tc.warmup_steps:
        return tc.max_lr *(step+1)/tc.warmup_steps
    elif step>=tc.max_steps:
        return tc.min_lr
    else:
        r = (step - tc.warmup_steps)/(tc.max_steps-tc.warmup_steps)
        return tc.min_lr + 0.5 *(1+np.cos(np.pi*r))*(tc.max_lr-tc.min_lr)


def configure_optimizer(model, tc: TrainConfig, device):
    """AdamW(betas=(0.9, 0.95), lr=max_lr); weight decay on params with dim >= 2 only; fused on cuda."""
    fused = True if "cuda" in str(device) else False



    groups = [{"params":[p for p in model.parameters() if p.dim()>1],"weight_decay":tc.weight_decay},
              {"params":[p for p in model.parameters() if p.dim()<=1],"weight_decay":0.0}]
    
    optim = torch.optim.AdamW(groups, lr=tc.max_lr,betas=(0.9,0.95),fused=fused)
    return optim


def flops_per_token(cfg: GPTConfig, seq_len):
    """
    Training FLOPs per token: 6 * N_matmul + 12 * n_layers * d_model * seq_len,
    where N_matmul excludes tok_emb and norm weights, and the second term is attention (QK^T, PV).
    """
    # class GPTConfig:
    # vocab_size: int = 1024
    # d_model: int = 128
    # n_layers: int = 4
    # n_heads: int = 8
    # n_kv_heads: int = 8        # == n_heads -> MHA; < n_heads -> GQA
    # max_seq_len: int = 256
    # rope_base: float = 10000.0

    d, L, V = cfg.d_model, cfg.n_layers, cfg.vocab_size
    d_head = d // cfg.n_heads
    N_matmul = L * (10 * d ** 2 + 2 * d * d_head * cfg.n_kv_heads ) +  V * d 
    return 6*N_matmul + 12*L*d*seq_len



def accumulate_grads(model, micro_batches, bf16):
    """
    Forward/backward over micro_batches [(x, y), ...] with loss scaled by 1 / len(micro_batches).
    Does not zero grads or step. Returns the mean loss as a float.
    """
    total = 0
    for x, y in micro_batches:
        with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=bf16):
            logits = model(x)
            logits = logits.flatten(0,-2)
            loss = F.cross_entropy(logits, y.flatten())
        (loss / len(micro_batches)).backward()
        total += loss.item()
    return total / len(micro_batches)



def train_step(model, opt, micro_batches, tc: TrainConfig, step):
    """zero_grad -> accumulate -> clip to tc.grad_clip -> set lr_at(step) -> step. Returns (loss, pre-clip grad norm, lr)."""
    opt.zero_grad()
    losses = accumulate_grads(model,micro_batches,tc.bf16)
    pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tc.grad_clip)
    lr = lr_at(step=step,tc=tc)
    for g in opt.param_groups:
        g["lr"] = lr
    opt.step()
    return (losses,pre_clip_norm.item(),lr)


@torch.no_grad()
def evaluate(model, data, tc: TrainConfig, device, rng):
    """Mean cross-entropy over tc.eval_batches batches in eval mode; restores train mode."""
    model.eval()
    total = 0.0
    for i in range(tc.eval_batches):
        batch_x,batch_y = get_batch(data=data,batch_size=tc.batch_size,seq_len=tc.seq_len,rng=rng,device=device)
        with torch.no_grad():
            pred_y = model(batch_x)
            logits = pred_y.flatten(0,-2)
            loss = F.cross_entropy(logits, batch_y.flatten())
            total+=loss.item()
    model.train()
    return total / tc.eval_batches



def train(model_cfg: GPTConfig, tc: TrainConfig):
    """
    Runs tc.max_steps optimizer steps. Every tc.log_every steps (and the last) appends
    {step, loss, lr, grad_norm, tok_per_s, mfu, peak_mem_gb} to the returned history;
    every tc.eval_every steps reports val loss. Step time is measured between
    torch.cuda.synchronize() calls. Saves {model, optimizer, step, model_cfg, train_cfg} to tc.out.
    """
    torch.manual_seed(tc.seed)
    data_ = load_data(data_dir=tc.data_dir,split="train")
    data_eval = load_data(data_dir=tc.data_dir,split="val")
    model = GPT(cfg=model_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    rng = np.random.default_rng(tc.seed)
    model.to(device)
    raw_model = model
    if tc.compile:
        model = torch.compile(model)
    optim = configure_optimizer(model=model,tc=tc,device=device)
    logs_=[]
    for i in range(tc.max_steps):
        # microbatches
        microbatch=[get_batch(data=data_, batch_size=tc.batch_size, seq_len=tc.seq_len, device=device, rng=rng) for _ in range(tc.grad_accum)]
        if device.type == "cuda":
            torch.cuda.synchronize() # Clear any remaining background GPU tasks
        start_time = time.perf_counter()
        loss,norm,lr = train_step(model=model,opt=optim,micro_batches=microbatch,tc=tc,step=i)
        if device.type == "cuda":
            torch.cuda.synchronize() # Wait until the GPU finishes the block
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        tokenps = tc.batch_size*tc.seq_len*tc.grad_accum/elapsed_time

        mfu = tokenps*flops_per_token(cfg=model_cfg,seq_len=tc.seq_len)/PEAK_FLOPS
        peak_mem_gb = 0
        if device.type == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        if(i%tc.log_every==0 or i == tc.max_steps-1):
            logs_.append({"step": i, "loss": loss, "lr": lr, "grad_norm": norm, "tok_per_s": tokenps, "mfu": mfu, "peak_mem_gb": peak_mem_gb})


        if(i%tc.eval_every==0):
            eval_loss = evaluate(model=model,data=data_eval,tc=tc,device=device,rng=rng)
            print(f"Step{i}: Eval Loss = {eval_loss}")

    checkpoint = {"model":raw_model.state_dict(),
        "optimizer":optim.state_dict(),
        "step":i, 
        "model_cfg":asdict(model_cfg),
        "train_cfg":asdict(tc)}
    torch.save(checkpoint, tc.out)
    return logs_

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for f in fields(TrainConfig):
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if f.type in (bool, "bool") else type(f.default)
        p.add_argument(f"--{f.name}", type=kind, default=f.default)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    train(MODEL, _parse_args())
