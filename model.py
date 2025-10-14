import torch
import torch.nn as nn
import torch.functional as F
from torch.profiler import record_function
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
    swiglu_alpha: float = 1.702
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
        # Calculate base frequencies: freq = base^(2i/d)
        # Later we'll invert to get θ_i = base^(-2i/d)
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
        concentration, inv_freqs = self._compute_concentration_and_inv_freq()
        # Shape: (max_context_length)
        t = torch.arange(start, start + num_tokens - 1, dtype=torch.float32, device=self.device)
        # Compute outer product
        # Shape: (max_context_length) ⊗ (head_dim / 2) --> (max_context_length, head_dim / 2)
        freqs = torch.einsum("i,j->ij", t, inv_freqs)
        # Turn into rotation coefficients cos(tθ_i) and sin(tθ_i)
        # Multiply by YaRN concentration to apply the attention temperature softening via length scaling trick
        # Shapes: (max_context_length, head_dim / 2)
        cos = freqs.cos() * concentration 
        sin = freqs.sin() * concentration
        return cos, sin
    
    @record_function("rotate")
    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # Query or Key tensors to rotate
        # Shape: (max_context_length, head_dim / 2) --> (1, max_context_length, 1, head_dim / 2). 1's for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(2).to(x.dtype)
        sin = sin.unsqueeze(0).unsqueeze(2).to(x.dtype)
        # x's Shape: (Batch_size, Seq_len, n_heads, head_dim) --> Shape: (Batch_size, Seq_len, n_heads, head_dim / 2)
        # Assume Batch_size, Seq_len and n_heads = 1 for simplicity and head_dim = 8
        # x = [x1, x2, x3, x4, x5, x6, x7, x8]
        # x1 = [x1, x2, x3, x4]
        # x2 = [x5, x6, x7, x8]
        x1, x2 = torch.chunk(x, 2, dim=-1)
        # Shape: (Batch_size, Seq_len, n_heads, head_dim / 2)
        o1 = x1 * cos - x2 * sin
        # Shape: (Batch_size, Seq_len, n_heads, head_dim / 2)
        o2 = x2 * cos + x1 * sin
        # Shape: (Batch_size, Seq_len, n_heads, head_dim)
        return torch.cat((o1, o2), dim=-1)
    
    @record_function("rope")
    def forward(self, query: torch.Tensor, key: torch.Tensor, offset: torch.LongTensor):
        batch_size, num_tokens, num_heads, head_dim = query.shape
        batch_size, num_tokens, num_key_value_heads, head_dim = key.shape
        # Shape: (num_tokens)
        idx = torch.arange(num_tokens) + offset
        idx = idx % self.max_content_length
        # Shapes: (max_context_length, head_dim / 2) --> 0 below being the dim index
        cos = self.cos.index_select(0, idx)
        sin = self.sin.index_select(0, idx)

        query = self._rotate(query, cos, sin)
        key = self._rotate(key, cos, sin)
        return query, key
    
class Cache:
    pass

class AttentionBlock(nn.Module):
    pass

def swiglu(x: torch.Tensor, alpha: float, limit: float):
    pass

class MLPBlock(nn.Module):
    def __init__(self, args: ModelArgs, device: torch.device | None = None):
        super().__init__()
        self.num_experts = args.num_experts
        self.experts_per_token = args.experts_per_token
        self.swiglu_limit = args.swiglu_limit
        self.swiglu_alpha = args.swiglu_alpha
        # We apply normalisation "before" MLP block
        self.norm = RMSNorm(args.hidden_size, args.norm_eps, device=device)
        # Shape: (hidden_size, num_experts)
        # This means lets say we are doing a decoding step with a single token, the input
        # will be of shape (Batch_size, 1, hidden_size) @ (Batch_size, hidden_size, num_experts)
        # = (Batch_size, 1, num_experts)
        self.gate = nn.Linear(args.hidden_size, self.num_experts)
        # This is the first weight matrix which will get us to the expanded dimensions.
        # Usually in the case with a normal MLP as with llama2 for ex we could do this with nn.Linear
        # but since we are doing MoE, we will stack `num_experts` number tensors each 
        # of shape: (intermediate_size * 2, hidden_size)
        # The 2 here is for W and V tensors from the SwiGLU paper. The formula for the output of a
        # MLP with SwiGLU module is FFN_SwiGLU = (Swish(xW) * (xV))W2. Here we just pack W and V together
        self.mlp1_weight = nn.Parameter(
            torch.empty(
                (
                    args.num_experts,
                    args.intermediate_size * 2,
                    args.hidden_size,
                ),
                device=device,
                dtype=torch.bfloat16
            )
        )

        # Shape per expert tensor: (intermediate_size * 2)
        self.mlp1_bias = nn.Parameter(
            torch.empty(
                (
                    args.num_experts,
                    args.intermediate_size * 2
                )
            )
        )

        # This gets us back to the model's dimension `hidden_size`
        # Shape per expert tensor: (hidden_size, intermediate_size)
        self.mlp2_weight = nn.Parameter(
            torch.empty(
                (
                    args.num_experts,
                    args.hidden_size,
                    args.intermediate_size,
                ),
                device=device,
                dtype=torch.bfloat16
            )
        )

        # This the W2 tensor from the paper
        # Shape per expert tensor: (hidden_size)
        self.mlp2_bias = nn.Parameter(
            torch.empty(
                (
                    args.num_experts,
                    args.hidden_size
                )
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Recall from the paper: "applying root mean square normalization
        # [6] on the activations before each attention and MoE block. Similar to 
        # GPT-2 we use Pre-LN placement [7][8]."
        # (Batch_size, Seq_len, hidden_size) --> (Batch_size, Seq_len, hidden_size)
        t = self.norm(x)
        # Apply gating mechanism
        # (Batch_size, Seq_len, hidden_size) @ (Batch_size, hidden_size, num_experts)
        # = (Batch_size, Seq_len, num_experts)
        g = self.gate(t)
        # Shape: (Batch_size, Seq_len, experts_per_token)
        # A tuple is returned of (values, indices)
        experts = torch.topk(g, self.experts_per_token, dim=-1, sorted=True)
        expert_weights = F.softmax(experts.values) # how much each token contributes
        expert_indices = experts.indices

        # We now want to select the experts weights only from mlp1
        # Shape before: (Batch_size, Seq_len, experts_per_token)
        # Shape after: (Batch_size, Seq_len, experts_per_token, intermediate_size * 2, hidden_size)
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        
        # Apply first projection
        #  t: (Batch_size, Seq_len, hidden_size)
        #  mlp1_weight: (Batch_size, Seq_len, experts_per_token, intermediate_size * 2, hidden_size)

        # Each token’s hidden vector (last dim = hidden_size)
        # is multiplied by each expert’s weight matrix (intermediate_size * 2, hidden_size)
        # summing over the shared 'hidden_size' dimension.
        # Resulting shape:
        # t: (Batch_size, Seq_len, experts_per_token, intermediate_size * 2)
        t = torch.einsum("bth,btkih->btki", t, mlp1_weight) + mlp1_bias
        t = swiglu(t, alpha=self.swiglu_alpha, limit=self.swiglu_limit)

        # Now we perform the second linear projection (compression back to model dim)
        # Select the corresponding expert parameters again:
        # (Batch_size, Seq_len, experts_per_token, hidden_size, intermediate_size)
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        t = torch.einsum("btki,btkhi->btkh", t, mlp2_weight) + mlp2_bias
        t = swiglu(t, alpha=self.swiglu_alpha, limit=self.swiglu_limit)
        
        # Weighted sum of experts
        # (Batch_size, Seq_len, experts_per_token, hidden_size) * (Batch_size, Seq_len, experts_per_token)
        # = (Batch_size, Seq_len, hidden_size) 
        t = torch.einsum("btkh,btk->bth", t, expert_weights)
        return x + t

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx, device: torch.device | None = None):
        super().__init__()
        self.args = ModelArgs
        # We pass layer_idx to each block because from the paper: "Following GPT-3, attention blocks 
        # alternate between banded window and fully dense patterns [10][11], where the bandwidth is 128 tokens."
        self.layer_idx = layer_idx
        self.device = device

        self.attn = AttentionBlock(args, layer_idx, device)
        self.mlp = MLPBlock(args, layer_idx, device)

    def forward(self, x: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        x = self.attn(x, cache=cache)
        x = self.mlp(x)
        return x