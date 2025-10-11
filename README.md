# GPT-OSS-20B
A PyTorch + Triton implementation of the GPT-OSS-20B architecture focused on efficient inference. All components are coded from scratch: RoPE with YaRN, RMSNorm, SwiGLU with clamping and residual connection, Mixture-of-Experts (MoE), plus a Triton FlashAttention V2 algorithm with learned sinks, banded attention, and GQA, and KV-cache.

## Contents

## Contents

1. [Setup Instructions](#1-setup-instructions)  
2. [Rotary Position Embedding (RoPE)](#2-rotary-position-embedding-rope)  
&nbsp;&nbsp;&nbsp;&nbsp;2.1 [Original RoPE](#21-original-rope)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1.1 [Mathematical Definition](#211-mathematical-definition)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1.2 [Intuitive Explanation](#212-intuitive-explanation)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1.3 [Dimensional Trade-offs](#213-dimensional-trade-offs)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1.4 [Visual Example](#214-visual-example)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1.5 [Long-term Decay](#215-long-term-decay)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.1.6 [Why RoPE shapes model weights and fails at extrapolation](#216-why-rope-shapes-model-weights-and-fails-at-extrapolation)  
&nbsp;&nbsp;&nbsp;&nbsp;2.2 [Position Interpolation](#22-position-interpolation)  
&nbsp;&nbsp;&nbsp;&nbsp;2.3 [The NTK-Aware Approach](#23-the-ntk-aware-approach)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3.1 [The Core Problem](#231-the-core-problem)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3.2 [The NTK-Aware Solution: Changing the Base](#232-the-ntk-aware-solution-changing-the-base)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.3.3 [Numerical Example: Selective Scaling](#233-numerical-example-selective-scaling)  
&nbsp;&nbsp;&nbsp;&nbsp;2.4 [The "NTK-by-parts" Interpolation](#24-the-ntk-by-parts-interpolation)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.4.1 [The Core Mechanism: The Ratio r(d)](#241-the-core-mechanism-the-ratio-rd)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.4.2 [Alpha and Beta: Defining the Three Scaling Zones](#242-alpha-and-beta-defining-the-three-scaling-zones)

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

## 2. Rotary Position Embedding (RoPE)
Rotary Position Embedding (RoPE) is an effective position-encoding technique which was first introduced in [Su et al. 2021](https://arxiv.org/pdf/2104.09864). Due to its simplicity and effictivness has since become the de facto for modern LLMs including Llama 2, 3 [Grattafiori, Dubey, et al. 2024](https://arxiv.org/pdf/2407.21783), Mistral, Gemma-2 and other open source models. While the original method proved to be effective, models failed faced a crucial limitation of not being able to maintain quaility while processing sequences longer than their trained context. Other methods have been proposed which I am going to go through in this section until we reach the [YaRN](https://arxiv.org/pdf/2309.00071) extenstion which I use in this repo following OpenAI's original implementation 

Other great in-depth resources (Most of the visuals in this documentation is taken from these resources so credits to all authors
Sources: 
- [How LLMs Scaled from 512 to 2M context: A Technical Deep Dive](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html#roformer-enhanced-transformer-with-rotary-position-embedding-rope)
- [Inside RoPE: Rotary Magic into Position Embeddings](https://learnopencv.com/rope-position-embeddings/)
- [Extending the RoPE](https://blog.eleuther.ai/yarn/#rotary-position-embedding)
- [Extending Context is Hard](https://kaiokendev.github.io/context#a-bigger-problem)


## 2.1 Original RoPE

### Core Idea

Attention scores use dot products. We want the score between token $$m$$ and token $$n$$ to depend on the distance $$(n - m)$$ rather than on absolute $$m$$ and $$n$$. RoPE achieves this by rotating each two-dimensional slice of the query and key vectors by angles that grow linearly with position.

### 2.1.1 Mathematical Definition

We require the attention score to depend only on relative distance:  

$$
f_q(x_m, m)^{\top} f_k(x_n, n) = g(x_m, x_n, m-n)
$$

A uniform construction that satisfies this is:  

$$
f_W(x_m, m, \theta_d) =
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0\\
\sin m\theta_1 & \ \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0\\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0\\
0 & 0 & \sin m\theta_2 & \ \cos m\theta_2 & \cdots & 0 & 0\\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots\\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{\ell} & -\sin m\theta_{\ell}\\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{\ell} & \ \cos m\theta_{\ell}
\end{pmatrix} W_q x_m,
\qquad f_q = f_{W_q},\ f_k = f_{W_k}.
$$

Here the per-pair angles follow the RoPE schedule $$\theta_i = b^{-2i/d}$$ with $$b = 10000$$ and $$i=0,\dots,\frac{d}{2}-1$$ across a head of dimension $$d$$.  
In this repo we set $$b=150000$$ (matching OpenAI’s implementation). The head dimension $$d$$ must be even so every pair can form a $$2\times2$$ rotation block. Later, extensions will modify RoPE by changing $$f$$ into $$f’$$ via simple functions $$g$$ and $$h$$:

$$
f’_W(x_m, m, \theta_d) = f_W\big(x_m,\ g(m),\ h(\theta_d)\big)
$$

### 2.1.2 Intuitive Explanation

The schedule $$\theta_i=b^{-2i/d}$$ creates a geometric progression of frequencies across the $$\ell=d/2$$ pairs. Small $$i$$ gives large $$\theta_i$$ (fast “clocks”) with short wavelengths for very local detail; large $$i$$ gives small $$\theta_i$$ (slow “clocks”) with long wavelengths for long-range structure. The wavelength in tokens for pair $$i$$ is $$\lambda_i = \frac{2\pi}{\theta_i}$$, i.e., how many tokens it takes that pair’s “clock hand” to complete one full revolution.

### 2.1.3 Dimensional Trade-offs

Increasing $$d$$ gives more pairs (more clocks) and finer coverage—the gaps between adjacent frequencies shrink—at the cost of more memory, parameters, and FLOPs per token. Smaller $$d$$ is cheaper but less expressive.

### 2.1.4 Visual Example

Let $$d=64$$, the fastest pair $$i=0$$, sequence length $$6$$, and base $$b=10000$$. Then $$\lambda_0=\frac{2\pi}{\theta_0}=\frac{2\pi}{1}\approx 6.28$$ tokens, so the clock completes a full lap roughly every $$6.28$$ tokens.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bf815024-b442-4c50-baa2-167f91f5e605" alt="Image 1" width="40%">
</p>

Now a slower pair $$i=7$$. For $$d=64$$ and $$b=10000$$, $$\lambda_7\approx 47$$ tokens, so it takes about $$47$$ tokens to complete a lap.


<p align="center">
  <img src="https://github.com/user-attachments/assets/d1c0b004-3b90-410e-a7ca-9274c1c54dfe" alt="Image 2" width="40%">
  <img src="https://github.com/user-attachments/assets/24fa9c22-f581-4533-9c20-7929bbb404a7" alt="Image 3" width="40%">
</p>

Much slower pairs (e.g., $$i=20$$) have wavelengths in the thousands of tokens, acting like very long-scale channels. The model learns to mix fast (local) and slow (global) clocks inside attention.

### 2.1.5 Long-term Decay

Following Vaswani et al. (2017), we set $$\theta_i = 10000^{-\frac{2i}{d}}$$. One can prove this setting provides a long-term decay property (see §3.4.3), meaning the inner product decays as the relative distance increases, aligning with the intuition that tokens far apart should connect more weakly.

<p align="center">
  <img src="https://github.com/user-attachments/assets/18b88b79-503b-41d4-9207-1045c8959b4c" alt="Image 4" width="45%">
</p>

### 2.1.6 Why RoPE shapes model weights and fails at extrapolation  

RoPE defines position by rotating each two-dimensional subvector at its own fixed frequency, so that every token’s representation becomes a composite phase pattern - a multi-frequency fingerprint across all pairs of dimensions. During training, the projection weights $$W_Q$$ and $$W_K$$ learn not just token semantics, but how those semantics behave after rotation: they implicitly encode how to interpret those fingerprints so that relative rotations yield meaningful attention. Because those weights have only ever seen fingerprint patterns arising from the training positional range, they internalise mappings tailored to that phase space. If you extend to much larger positions, the rotations generate fingerprint patterns that lie in phase regions never encountered in training, and the learned projections no longer know how to map them consistently. leading attention to misalign, explode, or degrade. This is explained more in next section.

## 2.2 Position Interpolation

During pre-training, sequences are chunked to a fixed context length $$L$$. After training, raw models tend to degrade on inputs much longer than $$L$$. Instead of fully retraining on a larger window $$L' > L$$, [kaiokendev](https://kaiokendev.github.io/context#a-bigger-problem) and later researchers from Meta [Chen et al. 2023b](https://arxiv.org/pdf/2306.15595) discovered we can exploit RoPE’s relative nature and compress positions at inference.

kaiokendev’s breakthrough: don’t force the model to extrapolate past what it learned,interpolate instead. Scale positions down by a constant $$s<1$$: effectively use $$m’ = sm$$. For example, setting $$s=\tfrac{1}{4}$$ makes position $$8192$$ look like $$2048$$ to a model trained to $$2048$$ keeping the RoPE angles in-distribution.  

Formally, rewrite the RoPE mapping as  

$$
f’_W(x_m, m, \theta_d) = f_W\big(x_m,\ g(m),\ h(\theta_d)\big)
$$

with $$g(m)=m/s$$ or $$g(m)=sm$$ depending on whether you scale the position fed to the angles or the inverse frequency. The common “compress positions” view is $$m’ = sm$$ with $$s=\tfrac{L}{L’}<1$$, and $$h(\theta_d)=\theta_d$$. This is called **Position Interpolation (PI)**: keep the frequency schedule, shrink the effective positions so long inputs fall back into the range the model already mastered.

**Intuition:** the model was trained up to $$L$$ (say 2048). Beyond that, raw RoPE angles enter a regime it never learned. By scaling, position $$8192$$ maps to an effective $$2048$$, so attention continues to operate in the familiar range without breaking long-range reasoning.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d8658608-63a2-4d51-9b7e-d9a55ecefee7" alt="Image 5" width="65%">
</p>

Below is a figure from the paper that clearly illustrates why **extrapolation fails** while **interpolation succeeds**.

1. **Left panel:** The red curve represents the fitted attention score function $$a(s)$$, trained on positional differences $$s \in [0, 2048]$$.  
   The blue dots correspond to training samples (random input points).  
   Within this range, the attention scores remain smooth and well-behaved, typically bounded around $$[-1, 1]$$.

2. **Middle panel:** When evaluated beyond the training range ($$s > L_{\text{train}}$$), the function rapidly diverges, with values exceeding $$8000$$.  
   This uncontrolled growth leads to catastrophic failures in attention computation, as softmax weights collapse or explode.

3. **Right panel:** Under **Position Interpolation**, positions are compressed so that effective distances stay within the trained interval.  
   As a result, the function remains smooth, stable, and well-behaved—preserving consistent attention patterns even for much longer sequences.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1da3ea7c-4865-47a2-bf8e-ca4e88a4a696" alt="Image 5" width="75%">
</p>

## 2.3 The NTK-Aware Approach

The primary limitation of simple Position Interpolation (PI) is that it uniformly compresses all of the model's learned positional frequencies, destroying the critical high-frequency information responsible for local token relationships.

The "NTK-Aware" approach, first proposed in a [reddit post](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), solves this by modifying the rotational base of RoPE. This change is calculated to selectively apply interpolation pressure, ensuring that high frequencies are scaled less (or not at all), while low frequencies are scaled the most.

#### 2.3.1 The Core Problem

Recall that RoPE encodes position using a set of paired dimensions, each associated with a unique frequency $\theta_i$.

The frequency for dimension pair $i$ (where $i=0$ is the fastest pair) is:
$$\theta_i = \mathbf{b}^{-2i/d}$$

| Dimension Index i | θᵢ (Frequency) | Wavelength (λᵢ) | Positional Information Encoded |
| :---: | :---: | :---: | :---: |
| **Small i (e.g., i=0)** | **Highest** | **Shortest** (≈6 tokens) | **Local, fine-grained relationships** |
| **Large i (e.g., i=d/2−1)** | **Lowest** | **Longest** (up to ≈b tokens) | **Global, long-range relationships** |

Simple linear interpolation crushes all these frequencies equally, causing the high-frequency clocks to spin so slowly that adjacent tokens become positionally indistinguishable.

#### 2.3.2 The NTK-Aware Solution: Changing the Base

The NTK-Aware method addresses this by calculating a new base ($\mathbf{b}_{\text{new}}$) designed to achieve two goals:

1.  **Preserve the Highest Frequency:** The $i=0$ (local) dimension must remain unchanged ($\theta_{0, \text{new}} \approx \theta_{0, \text{orig}}$).
2.  **Interpolate the Lowest Frequency:** The final dimension ($i=d/2-1$) must be compressed by the context extension factor $\alpha$.

The required adjustment to the base $\mathbf{b}$ to accomplish this dimension-dependent scaling is:

$$\mathbf{b}_{\text{new}} = \mathbf{b}_{\text{original}} \times \alpha^{d/(d-2)}$$

As the original post stated:

> Instead of the simple linear interpolation scheme, I've tried to design a nonlinear interpolation scheme using tools from NTK literature. Basically this interpolation scheme changes the base of the RoPE instead of the scale, which intuitively changes the "spinning" speed which each of the RoPE's dimension vectors compared to the next. Because it does not scale the fourier features directly, all the positions are perfectly distinguishable from each other...

By applying the new, larger base $\mathbf{b}_{\text{new}}$, the interpolation pressure is naturally distributed: the scaling factor is near $1.0$ for the highest frequencies and gradually increases towards $1/\alpha$ for the lowest frequencies.

#### 2.3.3 Numerical Example: Selective Scaling

Let's see this in action for a toy model where $\mathbf{d=8}$, $\mathbf{b_{\text{orig}} = 10000}$, and we want to extend the context by a factor of $\mathbf{\alpha=4}$ (e.g., from 2K to 8K).

The new base is calculated as:
$$\mathbf{b}_{\text{new}} = 10000 \times 4^{8/(8-2)} \approx \mathbf{63496}$$

We compare the scaling (compression) effect on the wavelengths ($\lambda = 2\pi/\theta$) across the dimensions:

| Pair Index i | Frequency Type | Original Wavelength λ | NTK-Aware Wavelength λₙₜₖ | Scaling Factor |
| :---: | :---: | :---: | :---: | :---: |
| **0** | **Highest (Local)** | ≈ 6.28 | **≈ 6.28** | **1.0× (Protected !)** |
| 1 | High-Mid | ≈ 62.8 | ≈ 69.0 | 1.1× (Minor change) |
| 2 | Low-Mid | ≈ 628 | ≈ 755 | 1.2× |
| **3** | **Lowest (Global)** | **≈ 6283** | **≈ 8243** | **1.31× (Max Compression)** |

This table clearly demonstrates the core success of the NTK-Aware approach:

* The **Fastest (Local) clock** is completely protected (scaled by $1.0\times$) so the model retains its ability to discern local relationships.
* The **Slowest (Global) clock** absorbs most of the required context extension, ensuring the full length (8K tokens) is now mapped within the model's original trained frequency space.

By shifting the base, we **smoothly spread the pressure** to the frequencies that can handle it (the long-range ones), while preserving the high-frequency/local fidelity the model needs to function.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f7db85d4-c8f2-45e7-bd5d-ffb8b4404f92" alt="Image 7" width="65%">
</p>

The figure from post shows the perplexity comparison of the different context extension methods we have been exploring on Llama 7B. The gray line presents the baseline (scale=1), blue corresponds to linear interpolation with scale=4 and then green line corresponds to the NTK-aware scaling with alpha=8. As seen the NTK-aware scaling maintains much lower perplexity across extended content lengths without any fine-tuning

The figure above from the original post compares the perplexity of different RoPE context extension methods on LLaMA 7B. The gray line shows the baseline model with the original RoPE configuration (scale=1), limited to a 2k context. The blue dashed line represents linear position interpolation with a scale of 4, which does extend the context but increases in perplexity as the sequence grows longer. Finally, the green line corresponds to the NTK-aware scaling method with $$\alpha = 8$$, which maintains much lower perplexity across extended context lengths and notably, this is achieved without any fine-tuning.

> To my surprise, this method works extremely well, so much so that you don't even need to fine tune the LLaMA 7B model for 4096 context size! The perplexity degradation is minimal.


### 2.4 The "NTK-by-parts" Interpolation

The core issue that necessitated the "NTK-by-parts" method was the realization that the initial NTK-Aware method, while excellent for extrapolation **without fine-tuning**, introduced a catastrophic instability when the model was trained on long-context data.

This happened because NTK-Aware, in its effort to preserve the high frequencies (fast clocks), allowed their rotation angles to exceed the model's training domain, forcing them to perform **out-of-distribution extrapolation** on the most critical, local patterns.

The solution is to create a frequency-aware interpolation that guarantees **interpolation** (stability) for the fast clocks and allows **NTK-aware scaling** (maximum context extension) for the slow clocks.

### 2.4.1 The Core Mechanism: The Ratio $r(d)$

To distinguish between the fast and slow clocks, the "NTK-by-parts" method uses a variable $\mathbf{r(d)}$ defined as the ratio of the original context length ($L$) to the wavelength ($\lambda_d$) of the current dimension $d$:

$$r(d) = \frac{L}{\lambda_d}$$

Recall the definition from before: The wavelength $\lambda_d$ is the number of tokens it takes for the clock hand to complete one full revolution ($\lambda_d = 2\pi/\theta_d$).

The ratio $r(d)$ gives us a measure of frequency:
* **Large $r(d)$ (e.g., $r(d)>32$):** The wavelength ($\lambda_d$) is very small, meaning the wave completes **many cycles** within $L$. This is a **High-Frequency (Fast) Clock**, crucial for local relationships.
* **Small $r(d)$ (e.g., $r(d)<1$):** The wavelength ($\lambda_d$) is large (even greater than $L$), meaning the wave completes **less than one cycle** within $L$. This is a **Low-Frequency (Slow) Clock**, crucial for global relationships.

### 2.4.2 $\alpha$ and $\beta$: Defining the Three Scaling Zones

The hyperparameters $\mathbf{\alpha}$ and $\mathbf{\beta}$ are the tunable boundary markers on this frequency ratio $r(d)$. They define three frequency zones that dictate the scaling strategy. For LLaMA, the values are $\mathbf{\alpha=1}$ and $\mathbf{\beta=32}$.

The piecewise function $\gamma(r)$, which blends the PI-like and NTK-Aware scaling formulas, relies on these boundaries:

$$
\gamma(r) =
\begin{cases}
0, & \text{if } r < \alpha \\
1, & \text{if } r > \beta \\
\dfrac{r - \alpha}{\beta - \alpha}, & \text{otherwise}
\end{cases}
$$

| Condition | r(d) Range | γ(r) Value | Frequency Zone | Scaling Strategy |
| :---: | :---: | :---: | :---: | :---: |
| **r(d) > β** | r(d) > 32 | **1** | **Highest Frequencies (Fastest Clocks)** | **Simple PI / Linear Interpolation** |
| **α ≤ r(d) ≤ β** | 1 ≤ r(d) ≤ 32 | Ramp | **Mid Frequencies** | **Smooth Blend** |
| **r(d) < α** | r(d) < 1 | **0** | **Lowest Frequencies (Slowest Clocks)** | **NTK-Aware Scaling** |

By separating the frequency spectrum into parts, "NTK-by-parts" effectively solves the trade-off: it ensures the fast, local clocks are always kept stable and in-distribution (using PI), while the slow, global clocks are aggressively scaled for long context (using NTK-Aware). This results in a stable and high-performing model even after fine-tuning.



