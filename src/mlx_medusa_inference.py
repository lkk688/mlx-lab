import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# ==========================================
# 1. THE ZERO-COST MEDUSA ARCHITECTURE
# ==========================================
class MLXMedusaResBlock(nn.Module):
    """A tiny, zero-cost residual block that predicts the future."""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x):
        return x + nn.silu(self.linear(x))

class MLXMedusaHeads(nn.Module):
    def __init__(self, hidden_size, num_heads=3):
        super().__init__()
        self.blocks = [MLXMedusaResBlock(hidden_size) for _ in range(num_heads)]

    def __call__(self, x):
        features = []
        curr = x
        for block in self.blocks:
            curr = block(curr)
            features.append(curr)
        return features

# ==========================================
# 2. WEIGHT LOADER & CACHE MANAGER
# ==========================================
def load_medusa_weights(module, npz_path):
    """Surgically maps the PyTorch nn.ModuleList to the MLX Python list."""
    weights = np.load(npz_path)
    for k, v in weights.items():
        parts = k.split('.')
        if parts[0] == 'blocks':
            idx = int(parts[1])
            layer_name = parts[2] # 'linear'
            param_name = parts[3] # 'weight'
            
            block = module.blocks[idx]
            target = getattr(block, layer_name)
            setattr(target, param_name, mx.array(v))
    print(f"✅ Medusa Heads loaded successfully from {npz_path}")

def rollback_mlx_cache(cache, keep_len):
    """The Apple Unified Memory zero-cost rollback."""
    for c in cache:
        c.offset = keep_len

# ==========================================
# 3. ENGINES (BASELINE vs MEDUSA)
# ==========================================
def generate_baseline_mlx(model, tokenizer, prompt, max_new_tokens=60):
    prompt_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = make_prompt_cache(model)
    
    start_time = time.time()
    
    x = model.model.embed_tokens(prompt_ids)
    for i, layer in enumerate(model.model.layers):
        x = layer(x, mask=None, cache=cache[i])
    x = model.model.norm(x)
    logits = model.lm_head(x)
    
    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
    tokens = [next_token.item()]
    
    for _ in range(max_new_tokens - 1):
        x = model.model.embed_tokens(next_token)
        for i, layer in enumerate(model.model.layers):
            x = layer(x, mask=None, cache=cache[i])
        x = model.model.norm(x)
        logits = model.lm_head(x)
        
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        token_id = next_token.item()
        tokens.append(token_id)
        if token_id == tokenizer.eos_token_id: break
            
    mx.eval(next_token)
    wall_time = time.time() - start_time
    return tokenizer.decode(tokens), len(tokens), wall_time

def generate_medusa_mlx(model, medusa, tokenizer, prompt, max_new_tokens=60, K=3):
    prompt_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = make_prompt_cache(model)
    
    start_time = time.time()
    tokens_generated = 0
    forward_steps = 0
    stats = {"drafts_accepted": 0}
    
    # --- PREFILL ---
    x = model.model.embed_tokens(prompt_ids)
    for i, layer in enumerate(model.model.layers):
        x = layer(x, mask=None, cache=cache[i])
    
    # Extract pre-norm hidden state for Medusa
    hx_t = x[:, -1:, :] 
    x_norm = model.model.norm(hx_t)
    logits = model.lm_head(x_norm)
    
    next_token = mx.argmax(logits, axis=-1)
    output_tokens = [next_token.item()]
    tokens_generated += 1
    forward_steps += 1
    
    # --- DECODING LOOP ---
    while tokens_generated < max_new_tokens:
        
        # 1. Medusa Draft (Effectively Zero Compute!)
        # We predict the next 3 tokens based entirely on the previous hidden state
        m_feats = medusa(hx_t)
        draft_tokens = []
        for feat in m_feats:
            # Note: We do NOT apply layer norm here, matching our KD training!
            draft_logits = model.lm_head(feat) 
            draft_tokens.append(mx.argmax(draft_logits, axis=-1))
            
        draft_tensor = mx.concatenate(draft_tokens, axis=1)
        
        # 2. Verification Pass (Full Model)
        verify_input = mx.concatenate([next_token, draft_tensor], axis=1)
        vx = model.model.embed_tokens(verify_input)
        
        # We record the cache offset BEFORE we pass the 4 tokens through
        base_offset = cache[0].offset 
        
        # for layer in model.model.layers:
        #     vx = layer(vx, mask=None, cache=cache)
        for i, layer in enumerate(model.model.layers):
            vx = layer(vx, mask=None, cache=cache[i])
            
        vx_norm = model.model.norm(vx)
        slow_logits = model.lm_head(vx_norm)
        forward_steps += 1
        
        # 3. Vectorized Rejection (No Python CPU-GPU Stalls)
        true_tokens = mx.argmax(slow_logits, axis=-1)
        mx.eval(true_tokens, draft_tensor, vx) # Compile and execute Metal Graph!
        
        true_list = true_tokens[0].tolist()
        draft_list = draft_tensor[0].tolist()
        
        accepted = []
        hit_eos = False
        
        # Cascading Match
        for i in range(K):
            if true_list[i] == draft_list[i]:
                accepted.append(draft_list[i])
                stats["drafts_accepted"] += 1
                if draft_list[i] == tokenizer.eos_token_id: hit_eos = True; break
            else:
                accepted.append(true_list[i])
                if true_list[i] == tokenizer.eos_token_id: hit_eos = True
                break
                
        # Bonus token if all drafts were perfect
        if len(accepted) == K and not hit_eos:
            bonus = true_list[K]
            accepted.append(bonus)
            if bonus == tokenizer.eos_token_id: hit_eos = True
                
        for t in accepted:
            output_tokens.append(t)
        tokens_generated += len(accepted)
        
        # 4. Instant Unified Memory Rollback
        rollback_mlx_cache(cache, base_offset + len(accepted))
        
        # 5. Setup Next Loop
        next_token = mx.array([[accepted[-1]]])
        
        # Extract the exact hidden state that generated the last accepted token
        last_valid_idx = len(accepted) - 1
        hx_t = vx[:, last_valid_idx:last_valid_idx+1, :]
        
        if hit_eos: break

    wall_time = time.time() - start_time
    return tokenizer.decode(output_tokens), tokens_generated, wall_time, forward_steps, stats

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-4bit"
    
    print("Loading 4-bit Base Model into Unified Memory...")
    model, tokenizer = load(MODEL_ID)
    
    print("Loading Custom Medusa Heads...")
    hidden_size = model.model.embed_tokens.weight.shape[1]
    medusa = MLXMedusaHeads(hidden_size, num_heads=3)
    load_medusa_weights(medusa, "mlx_medusa_heads.npz")
    
    print("\nWarming up Metal GPU...")
    _, _, _ = generate_baseline_mlx(model, tokenizer, "def add(a, b):", max_new_tokens=5)
    _, _, _, _, _ = generate_medusa_mlx(model, medusa, tokenizer, "def add(a, b):", max_new_tokens=5, K=3)
    
    prompts = [
        "def quicksort(arr):",
        "import torch\nimport torch.nn as nn\nclass MultiHeadAttention(nn.Module):"
    ]
    
    print("\n" + "="*60)
    print(" 🏆 MEDUSA MLX SHOWDOWN (M-SERIES)")
    print("="*60)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[Prompt {i+1}]: {prompt.strip()}")
        
        base_text, base_toks, base_time = generate_baseline_mlx(model, tokenizer, prompt)
        base_tps = base_toks / base_time
        
        spec_text, spec_toks, spec_time, spec_steps, stats = generate_medusa_mlx(model, medusa, tokenizer, prompt)
        spec_tps = spec_toks / spec_time
        speedup = spec_tps / base_tps
        
        print(f"\n--- 🐢 4-BIT AUTOREGRESSIVE ---")
        print(f"Speed: {base_tps:.2f} tokens/sec ({base_time:.2f}s)")
        
        print(f"\n--- 🐍 MEDUSA ENGINE ---")
        print(f"Speed: {spec_tps:.2f} tokens/sec ({spec_time:.2f}s)")
        print(f"Model Steps: {spec_steps} (Generated {spec_toks} tokens)")
        print(f"Drafts Accepted: {stats['drafts_accepted']}")
            
        print(f"\n🔥 TRUE WALL-CLOCK SPEEDUP: {speedup:.2f}x faster!")
        print("-" * 60)