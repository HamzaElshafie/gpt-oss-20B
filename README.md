# GPT-OSS-20B
A PyTorch implementation of the [GPT-OSS-20B](https://arxiv.org/pdf/2508.10925) architecture focused on efficient inference. All components are coded from scratch: RoPE with YaRN, RMSNorm, SwiGLU with clamping and residual connection, Mixture-of-Experts (MoE), Self-Attention with learned sinks, banded attention, and GQA, and KV-cache.

## Contents

1. [Setup Instructions](#1-setup-instructions)  
2. [Model Architecture](#2-model-architecture)  
&nbsp;&nbsp;&nbsp;&nbsp;2.1 [Attention](#21-attention)  
&nbsp;&nbsp;&nbsp;&nbsp;2.2 [Mixture-of-Experts (MoE)](#22-mixture-of-experts-moe)  
3. [Rotary Position Embedding (RoPE)](#3-rotary-position-embedding-rope)  
&nbsp;&nbsp;&nbsp;&nbsp;3.1 [Original RoPE](#31-original-rope)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.1 [Mathematical Definition](#311-mathematical-definition)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.2 [Intuitive Explanation](#312-intuitive-explanation)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.3 [Dimensional Trade-offs](#313-dimensional-trade-offs)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.4 [Visual Example](#314-visual-example)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.5 [Long-term Decay](#315-long-term-decay)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.6 [Why RoPE Shapes Model Weights and Fails at Extrapolation](#316-why-rope-shapes-model-weights-and-fails-at-extrapolation)  
&nbsp;&nbsp;&nbsp;&nbsp;3.2 [Position Interpolation](#32-position-interpolation)  
&nbsp;&nbsp;&nbsp;&nbsp;3.3 [The NTK-Aware Approach](#33-the-ntk-aware-approach)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3.1 [The Core Problem](#331-the-core-problem)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3.2 [The NTK-Aware Solution: Changing the Base](#332-the-ntk-aware-solution-changing-the-base)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3.3 [Numerical Example: Selective Scaling](#333-numerical-example-selective-scaling)  
&nbsp;&nbsp;&nbsp;&nbsp;3.4 [The "NTK-by-parts" Interpolation](#34-the-ntk-by-parts-interpolation)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.4.1 [The Core Mechanism: The Ratio r(d)](#341-the-core-mechanism-the-ratio-rd)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.4.2 [Alpha and Beta: Defining the Three Scaling Zones](#342-alpha-and-beta-defining-the-three-scaling-zones)  
&nbsp;&nbsp;&nbsp;&nbsp;3.5 [YaRN: Yet Another RoPE Extension](#35-yarn-yet-another-rope-extension)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.5.1 [The Problem with Pure Interpolation: Softmax Sharpening](#351-the-problem-with-pure-interpolation-softmax-sharpening)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.5.2 [Attention Temperature Scaling](#352-attention-temperature-scaling)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.5.3 [The "Length Scaling" Trick](#353-the-length-scaling-trick)  
4. [Mixture-of-Experts (MoE)](#4-mixture-of-experts-moe)  
&nbsp;&nbsp;&nbsp;&nbsp;4.1 [Experts](#41-experts)  
&nbsp;&nbsp;&nbsp;&nbsp;4.2 [Gating Mechanism](#42-gating-mechanism)  
5. [Self-Attention](#5-self-attention)  
&nbsp;&nbsp;&nbsp;&nbsp;5.1 [Scaled Dot-Product Attention](#51-scaled-dot-product-attention)  
&nbsp;&nbsp;&nbsp;&nbsp;5.2 [Multi-Head Attention (MHA)](#52-multi-head-attention-mha)  
&nbsp;&nbsp;&nbsp;&nbsp;5.3 [Key-Value (KV) Caching](#53-key-value-kv-caching)  
&nbsp;&nbsp;&nbsp;&nbsp;5.4 [Grouped Query Attention (GQA)](#54-grouped-query-attention-gqa)  
&nbsp;&nbsp;&nbsp;&nbsp;5.5 [Banded (Sliding Window) Attention](#55-banded-sliding-window-attention)  
&nbsp;&nbsp;&nbsp;&nbsp;5.6 [Attention Sinks](#56-attention-sinks)

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

## 2. Model Architecture

OpenAI's **GPT-OSS** represents a hugely anticipated family of [open-weights models](https://huggingface.co/blog/welcome-openai-gpt-oss), marking the company's first public release of open-weights models since GPT-2. The family comprises two variants: a large model with **117 billion parameters (gpt-oss-120b)** and a smaller one with **21 billion parameters (gpt-oss-20b)**. Both models utilise a **Mixture-of-Experts (MoE)** architecture and a **4-bit quantisation scheme (MXFP4)**. This combination is crucial for enabling fast inference (due to fewer active parameters) while maintaining low resource consumption.

**Note:** *For simplicity, the quantisation scheme is disregarded in this repository, although the official models do utilise MXFP4 for optimised inference.*

### Components:

### 2.1 Attention
The self-attention mechanism is highly optimised for efficiency and context length:
* Mechanism: Attention layers alternate between a **full content attention** mechanism and a **sliding 128-token window attention** mechanism.
* Heads: Each layer features 64 query heads of dimension 64.
* **Grouped-Query Attention (GQA)**: The model employs GQA with 8 key-value (KV) heads for optimised memory bandwidth and fast inference.
* Positional Encoding: **Rotary Positional Embedding (RoPE)** is used, augmented with the **YaRN extension** (Yet another RoPE extensioN) to support an extended context length of **131,072 tokens**.
* Attention Bias: Each attention head includes a learned bias in the denominator of the softmax, similar to concepts like Attention Sinks. This feature allows the attention mechanism to selectively **pay no attention** to certain tokens, providing an additional learned control signal.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e7d20729-328b-444d-9e1a-5cb2d6997995" alt="Image 5" width="50%">
  <img src="https://github.com/user-attachments/assets/f6b75744-5bb2-4d03-a946-b0abcb985750" alt="Image 5" width="42%">
</p>

### 2.2 Mixture-of-Experts (MoE)
The standard feed-forward network (FFN) is replaced with a Mixture-of-Experts block. This allows only a subset of experts to be engaged for each token generation step, significantly reducing computational load:
* Experts: The gpt-oss-120b model uses 128 experts, while the gpt-oss-20b model uses 32 experts.
* Routing: A standard linear router projection maps residual activations to scores for each expert.
* Selection: For both models, the **top-4 experts** are selected per token, and their outputs are weighted by the softmax of the router projection, calculated only over the selected experts.
* Activation: The MoE blocks utilise the **gated SwiGLU** activation function. *(More details on the MoE mechanism will follow in a later section!)*

<p align="center">
  <img src="https://github.com/user-attachments/assets/33b534e9-ad4f-4b07-9095-a6be6c19096e" alt="Image 5" width="70%">
</p>

As illustrated (figures from ["The Illustrated GPT-OSS"](https://newsletter.languagemodels.co/p/the-illustrated-gpt-oss), the architecture incorporates several state-of-the-art components, aligning closely with current high-performance LLMs while featuring key innovations:

## 3. Rotary Position Embedding (RoPE)
Rotary Position Embedding (RoPE) is an effective position-encoding technique which was first introduced in [Su et al. 2021](https://arxiv.org/pdf/2104.09864). Due to its simplicity and effictivness has since become the de facto for modern LLMs including Llama 2, 3 [Grattafiori, Dubey, et al. 2024](https://arxiv.org/pdf/2407.21783), Mistral, Gemma-2 and other open source models. While the original method proved to be effective, models failed faced a crucial limitation of not being able to maintain quaility while processing sequences longer than their trained context. Other methods have been proposed which I am going to go through in this section until we reach the [YaRN](https://arxiv.org/pdf/2309.00071) extenstion which I use in this repo following OpenAI's original implementation 

Other great in-depth resources (Most of the visuals in this documentation is taken from these resources so credits to all authors
Sources: 
- [How LLMs Scaled from 512 to 2M context: A Technical Deep Dive](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html#roformer-enhanced-transformer-with-rotary-position-embedding-rope)
- [Inside RoPE: Rotary Magic into Position Embeddings](https://learnopencv.com/rope-position-embeddings/)
- [Extending the RoPE](https://blog.eleuther.ai/yarn/#rotary-position-embedding)
- [Extending Context is Hard](https://kaiokendev.github.io/context#a-bigger-problem)


## 3.1 Original RoPE

### Core Idea

Attention scores use dot products. We want the score between token $$m$$ and token $$n$$ to depend on the distance $$(n - m)$$ rather than on absolute $$m$$ and $$n$$. RoPE achieves this by rotating each two-dimensional slice of the query and key vectors by angles that grow linearly with position.

### 3.1.1 Mathematical Definition

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

### 3.1.2 Intuitive Explanation

The schedule $$\theta_i=b^{-2i/d}$$ creates a geometric progression of frequencies across the $$\ell=d/2$$ pairs. Small $$i$$ gives large $$\theta_i$$ (fast “clocks”) with short wavelengths for very local detail; large $$i$$ gives small $$\theta_i$$ (slow “clocks”) with long wavelengths for long-range structure. The wavelength in tokens for pair $$i$$ is $$\lambda_i = \frac{2\pi}{\theta_i}$$, i.e., how many tokens it takes that pair’s “clock hand” to complete one full revolution.

### 3.1.3 Dimensional Trade-offs

Increasing $$d$$ gives more pairs (more clocks) and finer coverage—the gaps between adjacent frequencies shrink—at the cost of more memory, parameters, and FLOPs per token. Smaller $$d$$ is cheaper but less expressive.

### 3.1.4 Visual Example

Let $$d=64$$, the fastest pair $$i=0$$, sequence length $$6$$, and base $$b=10000$$. Then $$\lambda_0=\frac{2\pi}{\theta_0}=\frac{2\pi}{1}\approx 6.28$$ tokens, so the clock completes a full lap roughly every $$6.28$$ tokens.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bf815024-b442-4c50-baa2-167f91f5e605" alt="Image 1" width="45%">
</p>

Now a slower pair $$i=7$$. For $$d=64$$ and $$b=10000$$, $$\lambda_7\approx 47$$ tokens, so it takes about $$47$$ tokens to complete a lap.


<p align="center">
  <img src="https://github.com/user-attachments/assets/d1c0b004-3b90-410e-a7ca-9274c1c54dfe" alt="Image 2" width="40%">
  <img src="https://github.com/user-attachments/assets/24fa9c22-f581-4533-9c20-7929bbb404a7" alt="Image 3" width="40%">
</p>

Much slower pairs (e.g., $$i=20$$) have wavelengths in the thousands of tokens, acting like very long-scale channels. The model learns to mix fast (local) and slow (global) clocks inside attention.

### 3.1.5 Long-term Decay

Following Vaswani et al. (2017), we set $$\theta_i = 10000^{-\frac{2i}{d}}$$. One can prove this setting provides a long-term decay property (see §3.4.3), meaning the inner product decays as the relative distance increases, aligning with the intuition that tokens far apart should connect more weakly.

<p align="center">
  <img src="https://github.com/user-attachments/assets/18b88b79-503b-41d4-9207-1045c8959b4c" alt="Image 4" width="60%">
</p>

### 3.1.6 Why RoPE shapes model weights and fails at extrapolation  

RoPE defines position by rotating each two-dimensional subvector at its own fixed frequency, so that every token’s representation becomes a composite phase pattern - a multi-frequency fingerprint across all pairs of dimensions. During training, the projection weights $$W_Q$$ and $$W_K$$ learn not just token semantics, but how those semantics behave after rotation: they implicitly encode how to interpret those fingerprints so that relative rotations yield meaningful attention. Because those weights have only ever seen fingerprint patterns arising from the training positional range, they internalise mappings tailored to that phase space. If you extend to much larger positions, the rotations generate fingerprint patterns that lie in phase regions never encountered in training, and the learned projections no longer know how to map them consistently. leading attention to misalign, explode, or degrade. This is explained more in next section.

## 3.2 Position Interpolation

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
  <img src="https://github.com/user-attachments/assets/1da3ea7c-4865-47a2-bf8e-ca4e88a4a696" alt="Image 5" width="80%">
</p>

## 3.3 The NTK-Aware Approach

The primary limitation of simple Position Interpolation (PI) is that it uniformly compresses all of the model's learned positional frequencies, destroying the critical high-frequency information responsible for local token relationships.

The "NTK-Aware" approach, first proposed in a [reddit post](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/), solves this by modifying the rotational base of RoPE. This change is calculated to selectively apply interpolation pressure, ensuring that high frequencies are scaled less (or not at all), while low frequencies are scaled the most.

### 3.3.1 The Core Problem

Recall that RoPE encodes position using a set of paired dimensions, each associated with a unique frequency $\theta_i$.

The frequency for dimension pair $i$ (where $i=0$ is the fastest pair) is:
$$\theta_i = \mathbf{b}^{-2i/d}$$

| Dimension Index i | θᵢ (Frequency) | Wavelength (λᵢ) | Positional Information Encoded |
| :---: | :---: | :---: | :---: |
| **Small i (e.g., i=0)** | **Highest** | **Shortest** (≈6 tokens) | **Local, fine-grained relationships** |
| **Large i (e.g., i=d/2−1)** | **Lowest** | **Longest** (up to ≈b tokens) | **Global, long-range relationships** |

Simple linear interpolation crushes all these frequencies equally, causing the high-frequency clocks to spin so slowly that adjacent tokens become positionally indistinguishable.

### 3.3.2 The NTK-Aware Solution: Changing the Base

The NTK-Aware method addresses this by calculating a new base ($\mathbf{b}_{\text{new}}$) designed to achieve two goals:

1.  **Preserve the Highest Frequency:** The $i=0$ (local) dimension must remain unchanged ($\theta_{0, \text{new}} \approx \theta_{0, \text{orig}}$).
2.  **Interpolate the Lowest Frequency:** The final dimension ($i=d/2-1$) must be compressed by the context extension factor $\alpha$.

The required adjustment to the base $\mathbf{b}$ to accomplish this dimension-dependent scaling is:

$$\mathbf{b}_{\text{new}} = \mathbf{b}_{\text{original}} \times \alpha^{d/(d-2)}$$

As the original post stated:

> Instead of the simple linear interpolation scheme, I've tried to design a nonlinear interpolation scheme using tools from NTK literature. Basically this interpolation scheme changes the base of the RoPE instead of the scale, which intuitively changes the "spinning" speed which each of the RoPE's dimension vectors compared to the next. Because it does not scale the fourier features directly, all the positions are perfectly distinguishable from each other...

By applying the new, larger base $\mathbf{b}_{\text{new}}$, the interpolation pressure is naturally distributed: the scaling factor is near $1.0$ for the highest frequencies and gradually increases towards $1/\alpha$ for the lowest frequencies.

### 3.3.3 Numerical Example: Selective Scaling

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
  <img src="https://github.com/user-attachments/assets/f7db85d4-c8f2-45e7-bd5d-ffb8b4404f92" alt="Image 7" width="70%">
</p>

The figure from post shows the perplexity comparison of the different context extension methods we have been exploring on Llama 7B. The gray line presents the baseline (scale=1), blue corresponds to linear interpolation with scale=4 and then green line corresponds to the NTK-aware scaling with alpha=8. As seen the NTK-aware scaling maintains much lower perplexity across extended content lengths without any fine-tuning

The figure above from the original post compares the perplexity of different RoPE context extension methods on LLaMA 7B. The gray line shows the baseline model with the original RoPE configuration (scale=1), limited to a 2k context. The blue dashed line represents linear position interpolation with a scale of 4, which does extend the context but increases in perplexity as the sequence grows longer. Finally, the green line corresponds to the NTK-aware scaling method with $$\alpha = 8$$, which maintains much lower perplexity across extended context lengths and notably, this is achieved without any fine-tuning.

> To my surprise, this method works extremely well, so much so that you don't even need to fine tune the LLaMA 7B model for 4096 context size! The perplexity degradation is minimal.


### 3.4 The "NTK-by-parts" Interpolation

The core issue that necessitated the "NTK-by-parts" method was the realization that the initial NTK-Aware method, while excellent for extrapolation **without fine-tuning**, introduced a catastrophic instability when the model was trained on long-context data.

This happened because NTK-Aware, in its effort to preserve the high frequencies (fast clocks), allowed their rotation angles to exceed the model's training domain, forcing them to perform **out-of-distribution extrapolation** on the most critical, local patterns.

The solution is to create a frequency-aware interpolation that guarantees **interpolation** (stability) for the fast clocks and allows **NTK-aware scaling** (maximum context extension) for the slow clocks.

### 3.4.1 The Core Mechanism: The Ratio $r(d)$

To distinguish between the fast and slow clocks, the "NTK-by-parts" method uses a variable $\mathbf{r(d)}$ defined as the ratio of the original context length ($L$) to the wavelength ($\lambda_d$) of the current dimension $d$:

$$r(d) = \frac{L}{\lambda_d}$$

Recall the definition from before: The wavelength $\lambda_d$ is the number of tokens it takes for the clock hand to complete one full revolution ($\lambda_d = 2\pi/\theta_d$).

The ratio $r(d)$ gives us a measure of frequency:
* **Large $r(d)$ (e.g., $r(d)>32$):** The wavelength ($\lambda_d$) is very small, meaning the wave completes **many cycles** within $L$. This is a **High-Frequency (Fast) Clock**, crucial for local relationships.
* **Small $r(d)$ (e.g., $r(d)<1$):** The wavelength ($\lambda_d$) is large (even greater than $L$), meaning the wave completes **less than one cycle** within $L$. This is a **Low-Frequency (Slow) Clock**, crucial for global relationships.

### 3.4.2 $\alpha$ and $\beta$: Defining the Three Scaling Zones

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

### 3.5 YaRN: Yet Another RoPE Extension

In 2023, researchers from Nous Research, [EleutherAI](https://www.eleuther.ai) and University of Geneva introduced [YaRN (Yet Another RoPE Extension)](https://arxiv.org/pdf/2309.00071). YaRN takes the best frequency-aware scaling method (NTK-by-parts) and adds one crucial innovation to address a downstream effect of interpolation: **Attention Temperature Scaling**.

YaRN combines two key techniques:

1.  **NTK-by-parts Interpolation:** Frequency-aware scaling (from the previous section).
2.  **Attention Temperature Scaling:** A novel mechanism to stabilise attention scores.

#### 3.5.1 The Problem with Pure Interpolation: Softmax Sharpening

While Position Interpolation (PI) and NTK-by-parts successfully extend the context window, they both share a limitation rooted in the geometry of the positional embeddings:

When you compress the positional indices (which all interpolation methods must do), you are geometrically squeezing the angular distance between the $Q$ and $K$ vectors.

- **Issue:** This compression reduces the angular separation between distant tokens. Because the attention score is calculated via the dot product ($q^{\mathsf{T}} k$), a smaller angle leads to a systematically higher dot product score than the model was trained for. The scores are artificially inflated for certain compressed positions.
  
- **Result (Sharpening):** When these inflated scores hit the Softmax function, the resulting probability distribution becomes exaggeratedly sharp. The attention mechanism over-relies on a single, high-scoring key and suppresses all others. This damages the model's ability to maintain fine-grained distinctions among compressed positions, which is crucial for complex reasoning.

### 3.5.2 Attention Temperature Scaling

YaRN solves the "sharpening" problem by introducing a temperature parameter, $t$, to the attention logits before the Softmax operation. This process is called **Attention Temperature Scaling**.

The theoretical modification is to include the temperature $t$:

$$\text{softmax}\left(\frac{q_n^{\mathsf{T}} k_m}{t\sqrt{D}}\right)$$

Where $t$ is calculated based on the scale factor $s$ (the context extension factor, $L_{\text{new}} / L_{\text{orig}}$):

$$t = \sqrt{1/s} \cdot \ln(s) + 1$$

The Intuition of Softening the Attention:

> This may seem counter-intuitive - a higher temperature actually **softens** the attention distribution, making the model pay attention to more tokens rather than focusing sharply. However, this is precisely why it works: position interpolation compresses positional information, which can create artifacts where certain keys get artificially inflated scores. By softening the Softmax, YaRN prevents the model from over-relying on a single, potentially incorrect high-scoring key. Instead, it forces the model to consider a broader range of keys, making its decisions more robust to the slight loss of precision from position interpolation. It’s a counter-intuitive but powerful idea - deliberately making attention “fuzzier” to handle compressed positions better.

### 3.5.3 The "Length Scaling" Trick

The actual implementation avoids modifying the attention code entirely. By recognizing that the dot product is symmetric, dividing the logits by $t$ is mathematically equivalent to scaling the Query and Key vectors by a factor of $1/\sqrt{t}$:

$$\text{softmax}\left(\frac{(\frac{q_n}{\sqrt{t}})^{\mathsf{T}} (\frac{k_m}{\sqrt{t}})}{\sqrt{D}}\right) = \text{softmax}\left(\frac{q_n^{\mathsf{T}} k_m}{t\sqrt{D}}\right)$$

YaRN implements this by multiplying the complex RoPE embeddings by the constant factor $\mathbf{1/\sqrt{t}}$. This "length scaling" trick effectively alters the attention mechanism with zero overhead during inference or training, as the RoPE embeddings are generated once in advance.

This dual approach, combining the stable NTK-by-parts frequency-aware scaling with the elegant **Attention Temperature Scaling**, allows YaRN to extend context with minimal perplexity degradation and maintain fine-grained positional discrimination. Therefore, this is the method OpenAI used in their official implementation and the one I use in this repo. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/144bbe63-aeb8-441a-80db-b3d639078274" alt="Image 9" width="70%">
</p>


This plot shows the experimental impact of YaRN's Attention Temperature Scaling on the perplexity (PPL) change ratio over long-context documents, specifically for a context extension factor of $s=8$. The X-axis represents the scaling factor $1/\sqrt{t}$, which controls the degree of softening applied to the Softmax distribution. The curve demonstrates the existence of an optimal "sweet spot" for this temperature correction. As $1/\sqrt{t}$ increases from the initial reference point of $1.0$ up to about $1.25$, the perplexity rapidly improves, hitting a peak improvement of approximately $-0.3$, which indicates a large performance boost. The curve reaches its minimum (best performance) when $1/\sqrt{t}$ is approximately $1.25$ to $1.3$. This point is the temperature sweet spot where the Softmax is sufficiently softened to counteract the compression artifacts without losing necessary focus. Conversely, as $1/\sqrt{t}$ continues to increase past this optimum, the perplexity starts to worsen again, demonstrating that the Softmax is becoming too soft or over-cooled, which causes the model to lose the necessary distinction between relevant and irrelevant tokens. This experiment clearly validates YaRN's approach of actively softening the attention to achieve robust long-context performance.


## 4. Mixture-of-Experts (MoE)

The foundational idea behind the Mixture-of-Experts (MoE) architecture was, in fact, introduced long before the recent deep learning traction, dating back to the 1990s. The concept was first presented in the paper **[Adaptive Mixtures of Local Experts](https://direct.mit.edu/neco/article-abstract/3/1/79/5560/Adaptive-Mixtures-of-Local-Experts?redirectedFrom=fulltext)** by Robert Jacobs, Geoffrey Hinton, and other colleagues. They introduced the idea of dividing a single neural network into multiple specialised **"experts"** managed by a **gating network**.

As deep learning picked up momentum with Large Language Models (LLMs), MoE resurfaced in 2017. Noam Shazeer (one of the main authors of the ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762) paper), alongside other colleagues (including Geoffrey Hinton again), proposed the **[Sparsely-Gated Mixture-of-Experts](https://www.semanticscholar.org/paper/Outrageously-Large-Neural-Networks%3A-The-Layer-Shazeer-Mirhoseini/510e26733aaff585d65701b9f1be7ca9d5afc586)** layer for recurrent neural language models.

The Sparsely-Gated Mixture-of-Experts Layer consists of multiple **experts** (feed-forward networks) and a trainable **gating network** that selects the combination of experts to process each input. The gating mechanism enables **conditional computation** within the network, ensuring that the experts most suited to the input text are selected.

As mentioned in the Model Architecture section, GPT-OSS, along with most contemporary state-of-the-art LLMs, integrates such MoE layers, replacing the traditional feed-forward layer in the original Transformer block. The key components of MoE layers are the **experts**, the **gating mechanism**, and the **load balancing**.

### 4.1 Experts

The fundamental idea of the MoE approach is to introduce **sparsity** within the neural network layers. In a conventional dense layer, all parameters are active for every input token. In contrast, an MoE layer consists of several specialized **"expert" sub-layers**. This design introduces sparsity because only a small **subset of the model's parameters** are utilised for each input token during the forward pass.

In Transformer-based architectures, MoE layers are typically integrated in place of the standard feed-forward layers. The exact implementation strategy varies based on the design goals:

* Some architectures, like GPT-OSS, maximise sparsity by replacing **all** feed-forward layers with MoEs.
* Others may involve replacing only a subset of the feed-forward layers.
* Some advanced models even feature a hierarchical structure where one MoE delegates to another MoE.

Crucially, all other LLM layers and their parameters remain unchanged, and these parameters are shared across the various experts.

### 4.2 Gating Mechanism

During the training of an MoE LLM, all expert parameters are updated. The primary role of the **gating mechanism** is to learn how to efficiently distribute input tokens to the most appropriate expert(s). It acts much like a router or a team manager, delegating specific tasks based on each expert's specialisation.

The gating component itself is a **trainable component** within the network, meaning it learns its own set of parameters simultaneously with the other network parameters during the training process.

The following image demonstrates the role of the gating mechanism: it routes the input only to Expert 1 and Expert 3. Consequently, during inference, only the parameters of those selected experts are active and fetched from memory, while the parameters of the unselected experts are not used.

<p align="center">
<img src="https://github.com/user-attachments/assets/aa491573-a608-4624-8d48-7bd43b11698b" alt="Image 10" width="50%">
</p>

To compute the output of an MoE module, we take a weighted combination of expert outputs. Consider an MoE layer consisting of $n$ experts, denoted as $E_i(x)$ with $i=1,\dots,n$, that takes input $x$. The final MoE layer output ($y$) is calculated as:

$$
y = \sum_{i=1}^{n} G(x)_i \, E_i(x)
$$

where $G(x)_i$ is the $i^{th}$ expert’s final score, and $s_i$ is the initial score modeled based on the Softmax function:

$$
G(x)_i =
\begin{cases}
s_i, & \text{if } i \text{ is in the Top-k selection} \\
0, & \text{otherwise}
\end{cases}
$$

$$
s_i = \mathrm{Softmax}_i(x \cdot W_{\text{g}})
$$

Here, the gating layer’s final output $G(x)_i$ is used as the weight when averaging the selected experts’ outputs to compute the MoE layer’s final output. If $G(x)_i$ is zero, we can forgo computing the expert function $E_i(x)$ entirely, which is the source of sparsity.

**Top-k** specifies how many experts are selected to be active per input token during inference. For example, Top-1 gating means each token is directed to one expert, Top-2 to two experts, and so on. For GPT-OSS-20B,based on `ModelArgs`, we have a total of $n=32$ experts but implements **Top-4** gating, meaning that only 4 of the available experts are activated for each token.

## 5. Self-Attention
### 5.1 Scaled Dot-Product Attention

In transformer-based architectures, attention heads are essential for learning long-range dependencies. The traditional *Multi-Head Attention (MHA)* mechanism introduced in the [*Attention Is All You Need*](https://arxiv.org/pdf/1706.03762) paper first formalised this concept. It describes attention as:

> An attention function maps a query and a set of key–value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is determined by a compatibility function of the query with the corresponding key.

At its core, the attention mechanism computes how similar each query vector is to all key vectors through a dot product. The resulting scores determine how much each token should attend to others:

$$
\text{Attention}(Q, K, V) = \text{softmax}(QK^{\top})V
$$

However, when applied in practice, particularly within each attention head,these dot products can become large when the vector dimensionality is high, leading to small gradients after the softmax. To counter this, we scale the dot products by the inverse square root of the per-head dimension, giving rise to the **scaled dot-product attention** used inside Multi-Head Attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

Here, $d_k$ corresponds to the dimensionality of each head (`head_dim` in code).

This formulation applies primarily during **training** and the **prefill phase** of inference. During **single-token decoding**, the same operation reduces to a vector–matrix multiplication since the query represents only the current token.  


### 5.2 Multi-Head Attention (MHA)

Instead of computing a single attention function over the entire `hidden_size`, the model splits this dimension into multiple smaller **heads**. Each head has its own set of learnable parameters for $W_Q$, $W_K$, and $W_V$, allowing the model to capture different types of relationships in parallel. In implementation, these projections are usually stored within the same tensor, with an additional dimension representing the number of heads.

Each head operates on sub-vectors of dimension `head_dim` and computes scaled dot-product attention independently. The results from all heads are then concatenated and projected back to the model dimension through an output projection matrix $W_O$:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O
$$

where each head performs:

$$
\text{head}_i = \text{Attention}(QW_Q^{(i)}, KW_K^{(i)}, VW_V^{(i)}) =
\text{softmax}\!\left(\frac{Q_i K_i^{\top}}{\sqrt{d_k}}\right)V_i
$$

Note that each token has its own distinct query, key, and value vectors, ensuring that each head learns to specialise in a particular aspect of the attention pattern such as local syntactic relations, global context, or specific token dependencies.

### 5.3 Key-Value (KV) Caching
LLM training and inference have fundamentally different bottlenecks. Training is typically compute-bound, while inference especially autoregressive decoding is memory-bound.  

During inference, the GPU must repeatedly load model weights from HBM and read the growing KV cache. Since each decode step processes only a single token but requires loading all weights, arithmetic intensity is low, and the GPU spends more time moving data than computing. HBM, while large, has limited bandwidth, creating a bottleneck.  

The sequential nature of autoregressive generation exacerbates this: to generate token *t + 1*, we need all previous tokens *1 … t*. This severely underutilises GPU parallelism, as we cannot decode multiple tokens independently (Considering naive decoding here, as there are other optimisations that sort of parallelise this process)

KV caching is the core optimisation that makes decoding practical. In the attention mechanism, each new token’s computation requires the key (K) and value (V) tensors of all previous tokens. Without caching, we would recompute these tensors for the entire sequence at every step, which is wasteful and prohibitively slow.  

Instead, we cache K and V tensors in GPU memory:  

- During **prefill**, K and V for all input tokens are computed and stored.  
- During **decode**, each new token’s K and V are appended to the cache.  
- Subsequent steps simply read from this cache rather than recomputing.  

This transforms what would be *O(n²)* recomputation into *O(n)* memory reads, making generation feasible.

<p align="center">
<img src="https://github.com/user-attachments/assets/756e2465-608f-4513-bfab-3c0854499e5c" alt="Image 10" width="50%">
</p>

### 5.4 Grouped Query Attention (GQA)

While the KV cache avoids redundant computations, the memory-bandwidth cost of repeatedly loading and updating the K and V tensors remains a bottleneck. With **Multi-Head Attention (MHA)**, each head maintains its own K and V projections, so both storage and bandwidth scale directly with the number of heads. For large models, this makes the KV cache one of the main bottlenecks in inference.  

To reduce these costs, several attention mechanisms have been proposed that aim to shrink the KV footprint or reduce memory transfers while preserving model quality.  

[Multi-Query Attention (MQA)]((https://arxiv.org/abs/1911.02150)), introduced by Noam Shazeer (one of the original *Attention Is All You Need* authors), took an aggressive approach: use a single shared set of key and value projections across all query heads. This significantly reduced memory bandwidth costs and sped up decoding, as keys and values only needed to be loaded once per layer rather than once per head.  

However, MQA’s simplification came at a cost to model quality. While query heads could still learn different attention patterns, they all attended to the same key-value representations. This reduced representational diversity as one of MHA’s strengths is that different heads can extract different features from different subspaces. By forcing all queries to “look at” the same keys and values, MQA limited the model’s ability to capture nuanced relationships, leading to degraded performance.  

**Grouped Query Attention (GQA)**, introduced in [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245), provides a middle ground. GQA divides query heads into **G groups**, each sharing a single key head and value head. This balances MHA’s expressiveness with MQA’s efficiency:  

- **GQA-1** (one group) is equivalent to MQA  
- **GQA-H** (H groups, where H = number of heads) is equivalent to MHA  
- **GQA-G** (intermediate grouping) provides a tunable trade-off  

In practice, GQA achieves most of MQA’s speed and memory benefits while maintaining model quality much closer to full MHA striking a good balance and therefore a solid choice for SOTA LLMs.

<p align="center">
<img src="https://github.com/user-attachments/assets/e318ea0a-2d7c-451f-a8d3-255c89f39fe6" alt="Image 10" width="70%">
</p>

### 5.5 Banded (Sliding Window) Attention
### 5.6 Attention Sinks
