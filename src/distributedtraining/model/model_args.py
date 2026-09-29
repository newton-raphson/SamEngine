from dataclasses import field,dataclass

from torch import nn
from loguru import logger

from ... import MoEArgs
@dataclass
class DeepSeekV3ModelArgs:
    max_seq_len: int = 4096*4
    vocab_size: int = 102400
    dim: int = 2048
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    norm_eps: float =1E-5

    #MOE
    moe_args: MoEArgs = field(default_factory=MoEArgs)


    #Multi-Head Latent Attention
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_dim: int =128
    qk_rope_dim: int = 64
    v_head_dim: int =128
    #yarn 
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor:float = 40
    beta_fast:int = 32
    beta_slow:int = 1
    mscale:float = 1.0