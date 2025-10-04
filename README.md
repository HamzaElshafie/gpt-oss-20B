# gpt-oss-20B
A PyTorch + Triton implementation of the GPT-OSS-20B architecture focused on efficient inference. All components are coded from scratch: RoPE with YaRN, RMSNorm, SwiGLU with clamping and residual connection, Mixture-of-Experts (MoE), plus a Triton FlashAttention V2 algorithm with learned sinks, banded attention, and GQA, and KV-cache.

## Setup Instructions

### 1. Remote Setup

- Rent a GPU instance from a provider such as Vast.ai.  
- Start your instance, add your SSH public key, and copy one of the SSH connection commands provided.  
- Connect via SSH using your preferred code editor (e.g., VS Code or Cursor):  
  - In VS Code or Cursor, use the option to **Connect to Host...** and paste the SSH command.  
- Once connected to the remote instance, follow the setup steps below as you would on a local machine.

---

### 2. Install Prerequisites

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

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

### Login to Hugging Face

```bash
huggingface-cli login
```

---

### Download the Original Checkpoint

```bash
huggingface-cli download openai/gpt-oss-20b \
  --include "original/*" \
  --local-dir gpt-oss-20b/
```

---

### Download Tokenizer Files

```bash
huggingface-cli download openai/gpt-oss-20b \
  --include "tokenizer.json,tokenizer_config.json,special_tokens_map.json" \
  --local-dir gpt-oss-20b/
```
