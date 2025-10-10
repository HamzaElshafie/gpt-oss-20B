# GPT-OSS-20B
A PyTorch + Triton implementation of the GPT-OSS-20B architecture focused on efficient inference. All components are coded from scratch: RoPE with YaRN, RMSNorm, SwiGLU with clamping and residual connection, Mixture-of-Experts (MoE), plus a Triton FlashAttention V2 algorithm with learned sinks, banded attention, and GQA, and KV-cache.

## Contents

1. [Setup Instructions](#1-setup-instructions)  
2. [Rotary Position Embedding (RoPE)](#2-rotary-position-embedding-rope)  
&nbsp;&nbsp;&nbsp;&nbsp;2.1 [Original RoPE](#21-original-rope)  
&nbsp;&nbsp;&nbsp;&nbsp;2.2 [Position Interpolation](#22-position-interpolation)

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

### Mathematical Definition

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

### Intuitive Explanation

The schedule $$\theta_i=b^{-2i/d}$$ creates a geometric progression of frequencies across the $$\ell=d/2$$ pairs. Small $$i$$ gives large $$\theta_i$$ (fast “clocks”) with short wavelengths for very local detail; large $$i$$ gives small $$\theta_i$$ (slow “clocks”) with long wavelengths for long-range structure. The wavelength in tokens for pair $$i$$ is $$\lambda_i = \frac{2\pi}{\theta_i}$$, i.e., how many tokens it takes that pair’s “clock hand” to complete one full revolution.

### Dimensional Trade-offs

Increasing $$d$$ gives more pairs (more clocks) and finer coverage—the gaps between adjacent frequencies shrink—at the cost of more memory, parameters, and FLOPs per token. Smaller $$d$$ is cheaper but less expressive.

### Visual Example

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

### Long-term Decay

Following Vaswani et al. (2017), we set $$\theta_i = 10000^{-\frac{2i}{d}}$$. One can prove this setting provides a long-term decay property (see §3.4.3), meaning the inner product decays as the relative distance increases, aligning with the intuition that tokens far apart should connect more weakly.

<p align="center">
  <img src="https://github.com/user-attachments/assets/18b88b79-503b-41d4-9207-1045c8959b4c" alt="Image 4" width="45%">
</p>


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




