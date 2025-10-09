# gpt-oss-20B
A PyTorch + Triton implementation of the GPT-OSS-20B architecture focused on efficient inference. All components are coded from scratch: RoPE with YaRN, RMSNorm, SwiGLU with clamping and residual connection, Mixture-of-Experts (MoE), plus a Triton FlashAttention V2 algorithm with learned sinks, banded attention, and GQA, and KV-cache.

## Table of Contents

1. [Setup Instructions](#1-setup-instructions)  
2. [Original Rotary Position Embedding (RoPE)](#2-original-rotary-position-embedding-rope)

## 1. Setup Instructions

### Remote Setup

- Rent a GPU instance from a provider such as Vast.ai.  
- Start your instance, add your SSH public key, and copy one of the SSH connection commands provided.  
- Connect via SSH using your preferred code editor (e.g., VS Code or Cursor):  
  - In VS Code or Cursor, use the option to **Connect to Host...** and paste the SSH command.  
- Once connected to the remote instance, follow the setup steps below as you would on a local machine.

### Install Prerequisites

#### Install Miniconda  
[Official installation instructions](https://www.anaconda.com/docs/getting-started/miniconda/install)

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

Close and reopen terminal, then run:

```bash
source ~/miniconda3/bin/activate
conda init --all
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Download the Original Checkpoint

```bash
huggingface-cli download openai/gpt-oss-20b \
  --include "original/*" \
  --local-dir gpt-oss-20b/
```

### Download Tokenizer Files

```bash
huggingface-cli download openai/gpt-oss-20b \
  --include "tokenizer.json,tokenizer_config.json,special_tokens_map.json" \
  --local-dir gpt-oss-20b/
```

## 2. Original Rotary Position Embedding (RoPE)

### Core Idea

Attention scores use dot products. We want the score between token $$m$$ and token $$n$$ to depend on the distance $$(n - m)$$ rather than on absolute $$m$$ and $$n$$. RoPE achieves this by rotating each two dimensional slice of the query and key vectors by angles that grow linearly with position.

### Mathematical Definition

For a head with dimension size $$d$$, the final axis is divided into $$\frac{d}{2}$$ pairs. Each pair is indexed by $$i \in [0,\, \frac{d}{2}-1]$$. The base value, as originally defined in the paper, is $$10000.0$$. In the code for this repository it is $$150000.0$$, following OpenAI’s official implementation. The dimension $$d$$ must be even; otherwise, one component would remain unpaired.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ff44262-8e66-47e0-90ce-6a4852d99ee9" alt="Image 2" width="650">
</p>

The angles for each pair are defined as $$\theta_i = \text{base}^{-\frac{2i}{d}}$$, and the angle at position $$m$$ is $$m\theta_i$$. Equivalently, we can express this as $$m / \mathrm{inv\_freq}_i$$, where $$\mathrm{inv\_freq}_i = \text{base}^{\frac{2i}{d}}$$. Both forms describe the same geometric progression of frequencies.

In the two dimensional case, the rotation applied to each vector at position $$m$$ is  

$$
x'_m = R(m\theta_i)x_m, \quad \text{where} \quad
R(m\theta_i) =
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i)\\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix},
\quad x_m \in \mathbb{R}^2.
$$

For a full $$d$$-dimensional head, the complete transformation is a block-diagonal matrix composed of all such $$2\times2$$ rotations:

$$
R(m\Theta_d) = \mathrm{diag}\big(R(m\theta_0),\,R(m\theta_1),\,\dots,\,R(m\theta_{d/2-1})\big).
$$

Each block corresponds to one frequency pair, giving every subspace its own independent rotation rate.

### Intuitive Explanation

This formula defines a geometric progression of frequencies across all pairs. Small values of $$i$$ give large $$\theta_i$$, resulting in fast rotating clocks with very short wavelengths that capture fine local variations. Large $$i$$ values give small $$\theta_i$$, producing slow rotating clocks with long wavelengths that encode long range structure.

The wavelength of pair $$i$$, measured in tokens, is $$\lambda_i = \frac{2\pi}{\theta_i}$$. This quantity describes how many tokens it takes for that rotational “hand” to complete one full revolution. In essence, each pair corresponds to one time scale of the position encoding spectrum.

### Dimensional Trade-offs

Increasing $$d$$ increases the number of pairs $$i$$, and thus the number of clocks available to encode positional information. It also reduces the gaps between adjacent frequencies, providing finer and smoother coverage of scales. However, this comes at the cost of higher memory use, more parameters, and additional floating point operations per token. Smaller $$d$$ values are cheaper but less expressive, making them suitable for edge or lightweight models where efficiency outweighs positional granularity.

### Visual Example

Let us take $$d = 64$$, the fastest pair $$i = 0$$, a sequence length of 6, and the original base value $$10000.0$$. The wavelength of this pair is $$\lambda_0 = \frac{2\pi}{\theta_0} = \frac{2\pi}{1} \approx 6.28$$ tokens. This means the pair completes one full rotation roughly every 6.28 tokens.

| Position (m) | Angle (mθ₀) | Degrees (°) |
|:-------------:|:-----------:|:------------:|
| 0 | 0.000 rad | 0° |
| 1 | 1.000 rad | 57.3° |
| 2 | 2.000 rad | 114.6° |
| 3 | 3.000 rad | 171.9° |
| 4 | 4.000 rad | 229.2° |
| 5 | 5.000 rad | 286.5° |


![Image 1](https://github.com/user-attachments/assets/bf815024-b442-4c50-baa2-167f91f5e605)

Now consider a slower pair $$i = 7$$. For $$d = 64$$ and $$\text{base} = 10000$$, this gives $$\lambda_7 \approx 47$$ tokens, meaning it takes about 47 tokens for this pair to complete a full rotation.

| Token | Position (m) | θ (rad) | θ (°) |
|:------|:-------------:|:--------:|:------:|
| hello | 0 | 0.000 | 0.0° |
| my | 1 | 0.133 | 7.6° |
| name | 2 | 0.267 | 15.3° |
| is | 3 | 0.400 | 22.9° |
| Shubham | 4 | 0.533 | 30.5° |
| Anand | 5 | 0.667 | 38.2° |

![Image 2](https://github.com/user-attachments/assets/d1c0b004-3b90-410e-a7ca-9274c1c54dfe)

![Image 3](https://github.com/user-attachments/assets/24fa9c22-f581-4533-9c20-7929bbb404a7)

For much slower pairs, such as $$i = 20$$, the wavelength grows to around 2000 tokens. Each pair $$i$$ therefore represents a distinct time scale, ranging from very fine to extremely broad. Fast pairs capture local relationships between nearby tokens, while slow pairs encode long range dependencies. The transformer learns how to combine these scales effectively within attention.
