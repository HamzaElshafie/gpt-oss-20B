import torch
from model import Transformer, Cache  
import time
from typing import Optional

from openai_harmony import load_harmony_encoding, HarmonyEncodingName

def get_tokenizer():
    """
    Loads the official Harmony tokenizer for GPT-OSS models instantly.
    """
    tokenizer = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return tokenizer

class TokenGenerator:
    _model: Optional[Transformer] = None
    _tokenizer = None

    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        
        if TokenGenerator._model is None:
            print(f"[DEBUG] Loading model weights from {checkpoint}...")
            start = time.time()
            TokenGenerator._model = Transformer.from_checkpoint(checkpoint, device=self.device)
            print(f"[DEBUG] ✓ Model weights loaded in {time.time()-start:.2f}s")
        else:
            print("[DEBUG] Model weights already loaded. Reusing existing instance.")
            
        self.model = TokenGenerator._model
      
        if TokenGenerator._tokenizer is None:
            print(f"[DEBUG] Loading tokenizer...")
            TokenGenerator._tokenizer = get_tokenizer()
            eot_token = TokenGenerator._tokenizer.encode("<|end|>", allowed_special="all")[0]
            print(f"[DEBUG] ✓ Tokenizer loaded, EOT token: {eot_token}")
        
        self.tokenizer = TokenGenerator._tokenizer
        self.eot_token = self.tokenizer.encode("<|end|>", allowed_special="all")[0]
      
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: list[int],
        stop_tokens: list[int],
        temperature: float = 0.7,
        max_tokens: int = 20,
        return_logprobs: bool = False,
    ):
        batch_size = 1
        # Only allocate cache for the tokens we'll actually use
        total_tokens = len(prompt_tokens) + (max_tokens if max_tokens > 0 else 1024)
        cache_size = min(
            total_tokens,
            self.model.configs.initial_context_length * int(self.model.configs.rope_scaling_factor)
        )
        
        print(f"[DEBUG] Initializing KV caches...")
        print(f"[DEBUG]   - Cache size: {cache_size} tokens")
        print(f"[DEBUG]   - Number of layers: {self.model.configs.num_hidden_layers}")
        print(f"[DEBUG]   - KV heads: {self.model.configs.num_key_value_heads}")
        print(f"[DEBUG]   - Head dim: {self.model.configs.head_dim}")
        
        cache_start = time.time()
    
        caches = [
            Cache(
                batch_size=batch_size,
                n_ctx=cache_size,
                n_kv_heads=self.model.configs.num_key_value_heads,
                d_head=self.model.configs.head_dim,
                device=self.device
            )
            for _ in range(self.model.configs.num_hidden_layers)
        ]
        print(f"[DEBUG] ✓ Caches initialized in {time.time()-cache_start:.2f}s")

        tokens = list(prompt_tokens)
        print(f"[DEBUG] Prompt length: {len(tokens)} tokens")
        
        print(f"[DEBUG] Starting prefill phase (processing {len(tokens)} tokens)...")
        prefill_start = time.time()
        input_tensor = torch.as_tensor([tokens], dtype=torch.long, device=self.device)
        print(f"[DEBUG]   - Input tensor shape: {input_tensor.shape}")
        
        logits = self.model(input_tensor, caches=caches)[:, -1, :]
        print(f"[DEBUG] ✓ Prefill complete in {time.time()-prefill_start:.2f}s")
        
        logits = logits.squeeze(0)

        print(f"[DEBUG] Sampling first token (temperature={temperature})...")
        sample_start = time.time()
        if temperature == 0.0:
            predicted_token = torch.argmax(logits, dim=-1).item()
        else:
            probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
            predicted_token = torch.multinomial(probs, num_samples=1).item()
        print(f"[DEBUG] ✓ Sampled token {predicted_token} in {time.time()-sample_start:.4f}s")

        tokens.append(predicted_token)
        
        if return_logprobs:
            logprobs = torch.log_softmax(logits, dim=-1)
            selected_logprobs = logprobs[predicted_token].item()
            yield predicted_token, selected_logprobs
        else:
            yield predicted_token

        if predicted_token in stop_tokens:
            print(f"[DEBUG] Stop token encountered, ending generation")
            return

        print(f"[DEBUG] Starting generation phase (max {max_tokens} tokens)...")
        num_generated_tokens = 1
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            iter_start = time.time()
            input_tensor = torch.as_tensor([[predicted_token]], dtype=torch.long, device=self.device)
            
            logits = self.model(input_tensor, caches=caches)[:, -1, :]
            logits = logits.squeeze(0)

            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()

            tokens.append(predicted_token)
            num_generated_tokens += 1
            
            print(f"[DEBUG] Token {num_generated_tokens}: {predicted_token} ({time.time()-iter_start:.4f}s)")

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                print(f"[DEBUG] Stop token encountered, ending generation")
                break


def main():
    print("=" * 60)
    print("GPT-OSS-20B TOKEN GENERATOR DEBUG (Single-Load Model)")
    print("=" * 60)
    
    checkpoint_path = "/workspace/gpt-oss-20B/gpt-oss-20b/original"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    prompt = "What does it mean to live a meaningful life?" 
    temperature = 0.7
    max_tokens = 100

    print(f"\n[CONFIG]")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  Prompt: {prompt}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print()

    print("=" * 60)
    print("INITIALIZATION (Weight Loading Check)")
    print("=" * 60)
    init_start = time.time()
    generator = TokenGenerator(checkpoint_path, device=device)
    print(f"\n[TIMING] TokenGenerator initialization complete: {time.time()-init_start:.2f}s\n")

    print("=" * 60)
    print("TOKENIZATION")
    print("=" * 60)
    tokens = generator.tokenizer.encode(prompt, allowed_special="all") 
    print(f"[DEBUG] Encoded prompt: {tokens}")
    print()

    print("=" * 60)
    print("GENERATION")
    print("=" * 60)
    gen_start = time.time()
    generated_tokens = []
    
    for i, (token, logprob) in enumerate(generator.generate(
        tokens,
        stop_tokens=[generator.eot_token],
        temperature=temperature,
        max_tokens=max_tokens,
        return_logprobs=True,
    )):
        generated_tokens.append(token)
        token_text = generator.tokenizer.decode([token])
        print(f"[OUTPUT] Token {i+1}: {repr(token_text)} (logprob: {logprob:.4f})")

    print(f"\n[TIMING] Total generation: {time.time()-gen_start:.2f}s")
    print()
    
    print("=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    generated_text = generator.tokenizer.decode(generated_tokens)
    print(f"Generated text: {generated_text}")
    print(f"Total tokens generated: {len(generated_tokens)}")


if __name__ == "__main__":
    main()
