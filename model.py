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
    hidden_size: int = 2880 # Model dimension
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
            # Shape: (Batch, Seq_len, hidden_size)
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
            # Each of shape: (max_context_length, head_dim // 2)
            self.cos, self.sin = self._compute_cos_sin(0, self.max_content_length)

        
        def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
            # Calculate the θ (theta) pair indices [0, 2, 4, ..., head_dim-2]
            # Shape: (head_dim / 2)
            pair_indices = torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            # Calculate freqs --> Formula: θ_i = base^-2(i-1)/head_dim, where i = [1, 2, ..., head_dim/2] (We don't inverse yet, later!)
            # Shape: (head_dim / 2)
            freqs = self.base ** (pair_indices / self.head_dim)

            if self.scaling_factor > 1.0: # Do YaRN otherwise, do original RoPE
                # Original formula: t = √(1/s)·ln(s) + 1, tho for numerical stability OpenAI opted for a fixed coefficient for numerical 
                # stability. It appears this was found emperically
                concentration = 0.1 * math.log(self.scaling_factor) + 1.0 # YaRN concentration

                d_half = self.head_dim / 2 # Ex. 32
                # ============== NTK-by-parts ==============
                # Compute the cutpoints i.e i_β and i_α. Recall formula from documentation of the ration r(i) = L/λ_d 
                # where λ_i = 2π / θ_i. We can write formula as r(i) = L*θ_i / 2π. We know the formula for θ_i already
                # the "inverse freqs". Since "inverse freqs" is a decrease function of the dimension index (i), that
                # as i increases, θ_i decreases, so r(i) = L*θ_i / 2π also decreases. Thats why we want to find cutpoints to know 
                # which indices fall in which region
                
                # Index space:
                # 0                    i_β                i_α                d/2
                # ├─────────────────────┼──────────────────┼─────────────────┤
                # │   i < i_β           │  i_β ≤ i ≤ i_α   │    i > i_α      │
                # │   r(i) > β          │  α ≤ r(i) ≤ β    │   r(i) < α      │
                # │   FAST CLOCKS       │    MID RANGE     │   SLOW CLOCKS   │
                # │   (many cycles)     │                  │   (few cycles)  │
                # └─────────────────────┴──────────────────┴─────────────────┘

                low = (
                    d_half 
                    * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) 
                    / math.log(self.base)
                ) # i_β

                high = (
                    d_half 
                    * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) 
                    / math.log(self.base)
                ) # i_α    

                assert 0 < low < high < d_half - 1, "low and high cutoffs should match: 0 < low < high < d_half - 1"

                # Specify interpolation strategies (we inverse the freqs here!)
                # Note: Theory uses position interpolation for fast clocks and NTK aware base change for slow clocks, with a blend between.
                # This implementation keeps fast clocks original and applies position interpolation to slow clocks, then blends.
                interpolation = 1.0 / (self.scaling_factor * freqs)
                extrapolation = 1.0 / freqs # Standard RoPE

                # Shape: (d_half)
                # ramp < 0 = fast clocks (i < low)
                # 0 ≤ ramp ≤ 1 = transition zone (low ≤ i ≤ high)
                # ramp > 1 = slow clocks (i > high)
                ramp = (
                    torch.arange(d_half, dtype=torch.float, device=freqs.device) - low
                ) / (high - low)

                # Follows ramp function definition (see section 2.4.2 in README)
                # fast clocks after inversing clamps becomes = 1 --> original
                # slow clocks = 0 --> position interpolation
                mask = 1 - ramp.clamp(0, 1)

                inv_freqs = interpolation * (1-mask) + extrapolation * mask
            else:
                concentration = 1.0
                inv_freqs = 1.0 / freqs # Original RoPE

            return concentration, inv_freqs




        def _compute_cos_sin(self, start: int, num_tokens: int):
            pass