import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function
from dataclasses import dataclass
import math
import os
import json


@dataclass
class ModelConfigs:
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
        t = torch.arange(start, start + num_tokens, dtype=torch.float32, device=self.device)
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
    # Input shape: (Batch_size, Seq_len, experts_per_token, 2 * intermediate_size)
    # The formula for the output of a SwiGLU MLP is:
    # FFN_SwiGLU = (Swish(xW) * (xV))W2
    # The 2 in 2 * intermediate_size stores both W and V in one tensor we split them
    # Note also the last projection with W2 is made with mlp2_weights inside the MLPBlock
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # From paper "Our SwiGLU implementation is unconventional, including clamping and a residual connection."
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Add an extra bias to linear layer
    return out_glu * (x_linear + 1)

class MLPBlock(nn.Module):
    def __init__(self, configs: ModelConfigs, layer_idx: int = 0, device: torch.device | None = None):
        super().__init__()
        self.num_experts = configs.num_experts
        self.experts_per_token = configs.experts_per_token
        self.swiglu_limit = configs.swiglu_limit
        self.swiglu_alpha = configs.swiglu_alpha

        # We apply normalisation "before" the MLP block (Pre-LN placement)
        self.norm = RMSNorm(configs.hidden_size, configs.norm_eps, device=device)

        # Shape: (hidden_size, num_experts)
        # This means, for example, during a decoding step with a single token,
        # the input will be of shape (Batch_size, 1, hidden_size) @ (hidden_size, num_experts)
        # = (Batch_size, 1, num_experts)
        self.gate = nn.Linear(configs.hidden_size, self.num_experts)

        # This is the first weight matrix which expands the model dimension
        # to the intermediate dimension. Usually, in a normal MLP (like in LLaMA),
        # this would just be a single nn.Linear. But since we’re doing MoE,
        # we stack `num_experts` tensors, each of shape (hidden_size → intermediate_size * 2).
        #
        # The 2 here is for the W and V tensors from the SwiGLU paper.
        # The formula for the output of a SwiGLU MLP is:
        # FFN_SwiGLU = (Swish(xW) * (xV))W₂
        # Here, we just pack W and V together into one matrix.
        self.mlp1_weight = nn.Parameter(
            torch.empty(
                (
                    configs.num_experts,
                    configs.intermediate_size * 2,
                    configs.hidden_size,
                ),
                device=device,
                dtype=torch.bfloat16
            )
        )

        # Bias per expert for the first projection
        # Shape: (2 * intermediate_size)
        self.mlp1_bias = nn.Parameter(
            torch.empty(
                (
                    configs.num_experts,
                    configs.intermediate_size * 2
                ),
                device=device,
                dtype=torch.bfloat16
            )
        )

        # This is the second weight matrix which projects back down
        # from the intermediate dimension to the model dimension (`hidden_size`)
        # Shape per expert: (intermediate_size → hidden_size)
        self.mlp2_weight = nn.Parameter(
            torch.empty(
                (
                    configs.num_experts,
                    configs.hidden_size,
                    configs.intermediate_size,
                ),
                device=device,
                dtype=torch.bfloat16
            )
        )

        # This is the W₂ bias term from the SwiGLU formulation
        # Shape per expert: (hidden_size)
        self.mlp2_bias = nn.Parameter(
            torch.empty(
                (
                    configs.num_experts,
                    configs.hidden_size
                ),
                device=device,
                dtype=torch.bfloat16
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # As mentioned in the paper: "applying root mean square normalisation
        # on the activations before each attention and MoE block".
        # This is also similar to GPT-2 which uses a Pre-LN setup.
        # (Batch_size, Seq_len, hidden_size) --> (Batch_size, Seq_len, hidden_size)
        t = self.norm(x)

        # Apply the gating mechanism to determine expert routing
        # (Batch_size, Seq_len, hidden_size) @ (hidden_size, num_experts)
        # = (Batch_size, Seq_len, num_experts)
        g = self.gate(t)

        # Pick the top-k experts for each token
        # Shape: (Batch_size, Seq_len, experts_per_token)
        # torch.topk returns (values, indices)
        experts = torch.topk(g, self.experts_per_token, dim=-1, sorted=True)
        expert_weights = F.softmax(experts.values, dim=-1)  # how much each token contributes
        expert_indices = experts.indices

        # Select the corresponding experts’ weights and biases for MLP1
        # Before selection: (num_experts, hidden_size → 2 * intermediate_size)
        # After selection:  (Batch_size, Seq_len, experts_per_token, hidden_size → 2 * intermediate_size)
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]

        # Apply first projection
        # t: (Batch_size, Seq_len, hidden_size)
        # mlp1_weight: (Batch_size, Seq_len, experts_per_token, 2 * intermediate_size, hidden_size)
        #
        # Each token’s hidden vector (dim = hidden_size)
        # is multiplied by each expert’s projection (hidden_size → 2 * intermediate_size)
        # summing over the shared 'hidden_size' dimension.
        # Resulting shape:
        # (Batch_size, Seq_len, experts_per_token, 2 * intermediate_size)
        t = torch.einsum("bth,btkih->btki", t, mlp1_weight) + mlp1_bias
        t = swiglu(t, alpha=self.swiglu_alpha, limit=self.swiglu_limit)

        # Now perform the second projection which compresses back to model dim
        # Select the expert parameters again:
        # Before selection: (num_experts, intermediate_size → hidden_size)
        # After selection:  (Batch_size, Seq_len, experts_per_token, intermediate_size → hidden_size)
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]

        # Apply second projection
        # t: (Batch_size, Seq_len, experts_per_token, intermediate_size)
        # mlp2_weight: (Batch_size, Seq_len, experts_per_token, hidden_size, intermediate_size)
        # Einsum: "btki,btkhi->btkh"
        # Output: (Batch_size, Seq_len, experts_per_token, hidden_size)
        t = torch.einsum("btki,btkhi->btkh", t, mlp2_weight) + mlp2_bias
    
        # Weighted sum of expert outputs
        # (Batch_size, Seq_len, experts_per_token, hidden_size)
        # weighted by (Batch_size, Seq_len, experts_per_token)
        # Einsum: "btkh,btk->bth"
        # Result: (Batch_size, Seq_len, hidden_size)
        t = torch.einsum("btkh,btk->bth", t, expert_weights)

        # Add residual connection
        return x + t
    
class TransformerBlock(nn.Module):
    def __init__(self, configs: ModelConfigs, layer_idx, device: torch.device | None = None):
        super().__init__()
        self.configs = configs
        # We pass layer_idx to each block because from the paper: "Following GPT-3, attention blocks 
        # alternate between banded window and fully dense patterns [10][11], where the bandwidth is 128 tokens."
        self.layer_idx = layer_idx
        self.device = device

        self.attn = AttentionBlock(configs, layer_idx, device)
        self.mlp = MLPBlock(configs, layer_idx, device)

    def forward(self, x: torch.Tensor, cache: Cache | None = None) -> torch.Tensor:
        x = self.attn(x, cache=cache)
        x = self.mlp(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, configs: ModelConfigs, device: torch.device | None = None):
        super().__init__()
        self.configs = configs
        # Define the embedding "lookup" table. We want all tokens in the vocab to have an embedding vector
        # of hidden_size dimension holding its semantic meaning (no position info here!)
        self.embedding = nn.Embedding(
            configs.vocab_size, configs.hidden_size, device=device, dtype=torch.bfloat16
        )

        self.block = nn.ModuleList() 
        for layer_idx in range(configs.num_hidden_layers):
            self.block.append(TransformerBlock(configs, layer_idx, device))
        
        # The final RMSNorm before output linear
        self.norm = RMSNorm(configs.hidden_size, configs.norm_eps, device=device)
        self.unembedding = nn.Linear(
            configs.hidden_size,
            configs.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16
        )

    def forward(self, x: torch.Tensor, caches: list[Cache] | None = None) -> torch.Tensor:
        # KV caches
        # If not caches are provided we will have: caches = [None, None, ..., None]  (24 Nones)
        # If provided: caches = [cache_0, cache_1, cache_2, ..., cache_23]
        caches=caches or [None] * len(self.block)
        with record_function("embdedding"):
            # (B, Seq_len) --> (B, Seq_len, hidden_size)
            x = self.embedding(x)
        # Consecutively apply all the layers
        for block, cache in zip(self.block, caches):
            with record_function("block"):
                # (B, Seq_len, hidden_size) -> (B, Seq_len, hidden_size)
                x = block(x, cache=cache)
        with record_function("norm_f"):
            # (B, Seq_len, hidden_size) -> (B, Seq_len, hidden_size)
            x = self.norm(x)
            with record_function("unembedding"):
                # (B, Seq_len, hidden_size) -> (B, Seq_len, vocab_size)
                x = self.unembedding(x)
        return x.float()