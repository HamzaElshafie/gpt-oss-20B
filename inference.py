import torch
import time
from typing import Optional, List, Any, Generator, Union, Tuple
from dataclasses import dataclass, field
from model import Transformer, Cache  
from openai_harmony import load_harmony_encoding, HarmonyEncodingName

@dataclass() 
class Config:
    """
    Centralised configuration for the token generator.
    """
    debug_mode: bool = False
    checkpoint_path: str = "/workspace/gpt-oss-20B/gpt-oss-20b/original"
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature: float = 0.1
    max_tokens: int = 100 # if set to 0 -> full max context length  
    main_prompt: str = "Write a python function that prints hello world"

def debug_print(*args: Any, **kwargs: Any) -> None:
    """Prints a message only if Config.debug_mode is True."""
    if Config.debug_mode:
        print(f"[DEBUG]", *args, **kwargs)

def get_tokenizer():
    """
    Loads the official Harmony tokenizer for GPT-OSS models instantly.
    """
    tokenizer = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return tokenizer

class TokenGenerator:
    _model: Optional[Transformer] = None
    _tokenizer = None
    _eot_token: Optional[int] = None

    def __init__(self, checkpoint: str = Config.checkpoint_path, device: torch.device = Config.device):
        self.device = device
        
        if TokenGenerator._model is None:
            debug_print(f"Loading model weights from {checkpoint}...")
            start = time.time()
            TokenGenerator._model = Transformer.from_checkpoint(checkpoint, device=self.device)
            print(f"✓ Model weights loaded in {time.time()-start:.2f}s") 
        else:
            print("Model weights already loaded. Reusing existing instance.") 
            
        self.model: Transformer = TokenGenerator._model
      
        if TokenGenerator._tokenizer is None:
            print(f"Loading tokenizer...") 
            TokenGenerator._tokenizer = get_tokenizer()
            TokenGenerator._eot_token = TokenGenerator._tokenizer.encode("<|return|>", allowed_special="all")[0]
            debug_print(f"✓ Tokenizer loaded, EOT token: {TokenGenerator._eot_token}")
        
        self.tokenizer = TokenGenerator._tokenizer
        self.eot_token = TokenGenerator._eot_token
      
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[int],
        stop_tokens: List[int],
        temperature: float = Config.temperature,
        max_tokens: int = Config.max_tokens,
        return_logprobs: bool = False,
    ) -> Generator[Union[int, Tuple[int, float]], None, None]:
        
        batch_size = 1
        model_configs = self.model.configs
        
        # Determine max generation tokens, using ROPE context limit if max_tokens is 0
        max_gen_tokens = max_tokens if max_tokens > 0 else (
            model_configs.initial_context_length * int(model_configs.rope_scaling_factor)
        )
        total_tokens = len(prompt_tokens) + max_gen_tokens
        cache_size = min(
            total_tokens,
            model_configs.initial_context_length * int(model_configs.rope_scaling_factor)
        )
        
        # --- Cache Initialisation ---
        print(f"Initialising KV caches...")
        print(f"  - Cache size: {cache_size} tokens")
        print(f"  - Number of layers: {model_configs.num_hidden_layers}")
        print(f"  - KV heads: {model_configs.num_key_value_heads}")
        print(f"  - Head dim: {model_configs.head_dim}")
        
        cache_start = time.time()
        caches = [
            Cache(
                batch_size=batch_size,
                n_ctx=cache_size,
                n_kv_heads=model_configs.num_key_value_heads,
                d_head=model_configs.head_dim,
                device=self.device
            )
            for _ in range(model_configs.num_hidden_layers)
        ]
        print(f"✓ Caches Initialised in {time.time()-cache_start:.2f}s")

        tokens = list(prompt_tokens)
        print(f"Prompt length: {len(tokens)} tokens")
        
        # --- Prefill Phase ---
        print(f"Starting prefill phase (processing {len(tokens)} tokens)...")
        prefill_start = time.time()
        input_tensor = torch.as_tensor([tokens], dtype=torch.long, device=self.device)
        debug_print(f"  - Input tensor shape: {input_tensor.shape}")
        
        logits = self.model(input_tensor, caches=caches)[:, -1, :].squeeze(0)
        print(f"✓ Prefill complete in {time.time()-prefill_start:.2f}s")
        
        
        # --- Generation Phase (Decoding loop) ---
        print(f"Starting decoding phase (max {max_tokens} tokens)...")
        num_generated_tokens = 0
        is_debugging = Config.debug_mode
        
        # The loop now handles the first token generation and all subsequent tokens
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            iter_start = time.time()
            
            # For the first iteration (num_generated_tokens == 0), the input is the 
            # last token of the prompt (implicitly processed in the Prefill logits).
            # For subsequent iterations, the input is the previously predicted_token.
            if num_generated_tokens == 0:
                # Use logits from the prefill phase
                pass 
            else:
                # Use the predicted token from the last step for input
                input_tensor = torch.as_tensor([[predicted_token]], dtype=torch.long, device=self.device)
                logits = self.model(input_tensor, caches=caches)[:, -1, :].squeeze(0)

            # Sample next token
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()

            tokens.append(predicted_token)
            num_generated_tokens += 1
            
            # --- CONSOLIDATED DEBUG PRINTING ---
            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                
                if is_debugging:
                    token_text = self.tokenizer.decode([predicted_token])
                    # Print the consolidated debug line
                    print(
                        f"[DEBUG] Token {num_generated_tokens}: {predicted_token} ({time.time()-iter_start:.4f}s) "
                        # f"[OUTPUT] Token {num_generated_tokens}: {repr(token_text)} (logprob: {selected_logprobs:.4f})"
                    )

                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                print(f"Stop token encountered, ending generation")
                break
                
            # If the first token was generated and max_tokens is reached, break here
            if max_tokens > 0 and num_generated_tokens >= max_tokens:
                 break


def main():
    print("=" * 60)
    print("GPT-OSS-20B GENERATOR")
    print(f"DEBUG MODE IS {'ON' if Config.debug_mode else 'OFF'}")
    print("=" * 60)
    
    # Use configuration constants from Config
    checkpoint_path = Config.checkpoint_path
    device = Config.device
    prompt = Config.main_prompt
    temperature = Config.temperature
    max_tokens = Config.max_tokens

    print(f"\n[CONFIG]")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  Prompt: {prompt}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print()

    # --- Initialisation ---
    print("=" * 60)
    print("INITIALISATION (Weight Loading Check)")
    print("=" * 60)
    init_start = time.time()
    generator = TokenGenerator(checkpoint=checkpoint_path, device=device) 
    print(f"\n[TIMING] TokenGenerator initialisation complete: {time.time()-init_start:.2f}s\n")

    # --- Tokenisation ---
    print("=" * 60)
    print("TOKENISATION")
    print("=" * 60)
    tokens = generator.tokenizer.encode(prompt, allowed_special="all") 
    debug_print(f"Encoded prompt: {tokens}")
    print(f"Prompt length: {len(tokens)} tokens")
    print()

    # --- Generation ---
    print("=" * 60)
    print("GENERATION")
    print("=" * 60)
    gen_start = time.time()
    generated_tokens = []
    
    # The token-by-token print is now handled inside the generator.generate() function
    for i, (token, logprob) in enumerate(generator.generate(
        prompt_tokens=tokens,
        stop_tokens=[generator.eot_token],
        temperature=temperature,
        max_tokens=max_tokens,
        return_logprobs=True,
    )):
        generated_tokens.append(token)
        
    print(f"\n[TIMING] Total generation: {time.time()-gen_start:.2f}s")
    print()
    
    # --- Final Output ---
    print("=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    generated_text = generator.tokenizer.decode(generated_tokens)
    print(f"Generated text: {generated_text}")
    print(f"Total tokens generated: {len(generated_tokens)}")


if __name__ == "__main__":
    main()
