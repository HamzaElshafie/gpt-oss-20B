# gpt-oss-20B
A PyTorch + Triton implementation of the GPT-OSS-20B architecture focused on efficient inference. All components are coded from scratch: RoPE with YaRN, RMSNorm, SwiGLU with clamping and residual connection, Mixture-of-Experts (MoE), plus a Triton FlashAttention V2 algorithm with learned sinks, banded attention, and GQA, and KV-cache.
