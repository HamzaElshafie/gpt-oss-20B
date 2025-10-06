import torch
import torch.nn as nn
import torch.functional as F
from dataclasses import dataclass
import math
import os


@dataclass
class ModelArgs:
    num_hidden_layers: int = 24
    num_experts: int = 32
    experts_per_token: int = 4
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    vocab_size: int = 201088
    hidden_size: int = 2880 # Also refered to as d_model
    intermediate_size: int = 28800
    swiglu_limit: float = 7.0
    sliding_window: int = 128
    initial_context_length: int = 4096
    norm_eps: float = 1e-05
    rope_theta: float = 150000.0 # This is the "base" during RoPE in the llama-2 implementation
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
