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
    rope_theta: float = 150000.0 # This is the "base" during RoPE
    rope_scaling_factor: float = 32.0 # s = L_new / L_orig
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float, device: torch.device | None = None):
            """See RMSNorm paper https://arxiv.org/pdf/1910.07467
            
            Formula: 
                    RMSNorm(a) = (a / RMS(a)) * scale 
                    where RMS(a) = sqrt(mean(x^2) + eps)
                    mean is across the model `hidden_size` dimension
            """
            super().__init__()
            self.eps = eps
            self.hidden_size = hidden_size
            self.scale = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Shape (Batch, Seq_len, hidden_size)
            assert x.shape[-1] == self.hidden_size
            # Cast to FP32 for numerical stability
            t, dtype = x.float(), x.dtype
            # Mathematically, x/sqrt(v) can be written as x * 1/sqrt(v)
            # Keepdim=True makes shape (Batch, Seq_len, 1) --> which will be broadcasted later back to (Batch, Seq_len, hidden_size)
            t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps) # Keepdim=True makes shape
            # Shape: (Batch, Seq_len, hidden_size)
            return (t * self.scale).to(dtype)
        
    class RotaryEmbedding(nn.Module):
        def __init__(
            self, 
            head_dim: int, # Must be even 
            base: int, # Base for the geometric progression of frequencies
            dtype: torch.dtype,
            initial_context_length: int = 4096, # The training context L
            max_content_length: int = 131072,
            scaling_factor: float = 1.0, # s = L_new / L_orig --> 131072 / 4096 = 32
            ntk_alpha: float = 1.0, # Low frequencies below α follow original NTK-aware behaviour
            ntk_beta: float = 32.0, # High frequencies beyond β follow linear interpolation
            device: torch.device | None = None # Where to allocate the cos/sin tensors
        ) -> None:
            """See YaRN paper https://arxiv.org/pdf/2309.00071 and README.md for theory"""
            super().__init__()
            self.head_dim = head_dim
            self.base = base
            self.dtype = dtype
            self.initial_context_length = initial_context_length
            self.max_content_length = max_content_length
            self.scaling_factor = scaling_factor
            self.ntk_alpha = ntk_alpha
            self.ntk_beta = ntk_beta
            self.device = device
            # Each of shape (max_context_length, head_dim // 2)
            self.cos, self.sin = self._compute_cos_sin(0, self.max_content_length)

        def _compute_cos_sin(self, start: int, num_tokens: int):
            pass