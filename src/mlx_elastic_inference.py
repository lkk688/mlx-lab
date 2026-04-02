import time
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx.utils import tree_unflatten
# ==========================================
# 1. MLX CUSTOM ARCHITECTURE
# ==========================================
class MLXLocalAttentionDraftLayer(nn.Module):
    def __init__(self, hidden_size, window_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

    def __call__(self, x):
        # x shape: [batch, seq_len, hidden_size]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Local Sliding Window Attention (Simplified for generation step)
        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.hidden_size)
        probs = mx.softmax(scores, axis=-1)
        
        attn_output = self.o_proj(probs @ v)
        draft_features = x + attn_output
        return draft_features + self.adapter(draft_features)

class MLXMultiLayerDraftBlock(nn.Module):
    def __init__(self, hidden_size, window_size=32, num_layers=2):
        super().__init__()
        self.layers = [MLXLocalAttentionDraftLayer(hidden_size, window_size) for _ in range(num_layers)]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLXElasticComputeRouter(nn.Module):
    def __init__(self, hidden_size, num_lanes=2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size // 4, bias=False)
        self.fc2 = nn.Linear(hidden_size // 4, num_lanes, bias=False)
        self.safety_bias = mx.array([-1.0, 1.0])

        # THE FIX: Force the router to be brave! (Favor Lane 0)
        self.safety_bias = mx.array([2.0, -2.0])

    def __call__(self, x):
        # MLX natively handles lazy execution of this sequence
        x = nn.gelu(self.fc1(x))
        logits = self.fc2(x) + self.safety_bias
        return mx.argmax(logits, axis=-1)

# ==========================================
# 2. WEIGHT LOADER & CACHE MANAGER
# ==========================================
# def load_npz_to_mlx_module(module, npz_path):
#     """Converts a flat flat PyTorch npz dict into a nested MLX parameter tree."""
#     weights = np.load(npz_path)
#     tree = {}
#     for k, v in weights.items():
#         parts = k.split('.')
#         d = tree
#         for part in parts[:-1]:
#             d = d.setdefault(part, {})
#         d[parts[-1]] = mx.array(v)
#     module.update(tree)
# def load_npz_to_mlx_module(module, npz_path):
#     """Safely un-flattens PyTorch dot-notation keys into MLX lists and dicts."""
#     weights = np.load(npz_path)
#     # Convert flat numpy arrays to mx.array tuples
#     flat_weights = [(k, mx.array(v)) for k, v in weights.items()]
    
#     # Let MLX natively rebuild the nested tree (properly handling lists!)
#     tree = tree_unflatten(flat_weights)
#     module.update(tree)

def load_npz_to_mlx_module(module, npz_path):
    """Surgically translates PyTorch dictionary keys and injects weights into MLX."""
    weights = np.load(npz_path)
    
    for k, v in weights.items():
        # 1. Translate Drafter's PyTorch nn.Sequential -> MLX nn.Sequential list
        k = k.replace("adapter.0.", "adapter.layers.0.")
        k = k.replace("adapter.1.", "adapter.layers.1.")
        k = k.replace("adapter.3.", "adapter.layers.3.")
        
        # 2. Translate Router's PyTorch nn.Sequential -> MLX explicit naming
        k = k.replace("net.0.", "fc1.")
        k = k.replace("net.2.", "fc2.")
        
        # Navigate the MLX object tree using the translated path
        parts = k.split('.')
        obj = module
        
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, list) or isinstance(obj, tuple):
                obj = obj[int(part)] # Handle Python lists natively
            else:
                raise KeyError(f"Failed mapping: Could not traverse '{part}' in translated key '{k}'")
                
        # Inject the Metal GPU Array (Fixed the mx.array call!)
        last = parts[-1]
        if isinstance(obj, list) or isinstance(obj, tuple):
            obj[int(last)] = mx.array(v)
        else:
            setattr(obj, last, mx.array(v))


def rollback_mlx_cache(cache, keep_len):
    """
    The ultimate Unified Memory advantage. 
    Instead of deep-copying gigabytes of VRAM, we just move a C++ integer pointer!
    """
    for c in cache:
        c.offset = keep_len

def forward_partial(model, x, cache, start_idx, end_idx):
    """Executes a surgical slice of the base model layers."""
    for i in range(start_idx, end_idx):
        x = model.model.layers[i](x, mask=None, cache=cache[i])
    return x

# ==========================================
# 3. ENGINES (BASELINE vs ELASTIC)
# ==========================================
def generate_baseline_mlx(model, tokenizer, prompt, max_new_tokens=60):
    #prompt_ids = mx.array(tokenizer(prompt)["input_ids"])[None, :]
    prompt_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = make_prompt_cache(model)
    
    start_time = time.time()
    
    # Prefill
    x = model.model.embed_tokens(prompt_ids)
    for i, layer in enumerate(model.model.layers):
        x = layer(x, mask=None, cache=cache[i])
    x = model.model.norm(x)
    logits = model.lm_head(x)
    
    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
    tokens = [next_token.item()]
    
    # Decode
    for _ in range(max_new_tokens - 1):
        x = model.model.embed_tokens(next_token)
        for i, layer in enumerate(model.model.layers):
            x = layer(x, mask=None, cache=cache[i])
        x = model.model.norm(x)
        logits = model.lm_head(x)
        
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        token_id = next_token.item()
        tokens.append(token_id)
        
        if token_id == tokenizer.eos_token_id:
            break
            
    # Force MLX to evaluate the computation graph
    mx.eval(next_token)
    wall_time = time.time() - start_time
    
    return tokenizer.decode(tokens), len(tokens), wall_time

def generate_elastic_mlx(model, drafter, router, tokenizer, prompt, exit_idx=20, max_new_tokens=60, K=3):
    #prompt_ids = mx.array(tokenizer(prompt)["input_ids"])[None, :]
    prompt_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = make_prompt_cache(model)
    total_layers = len(model.model.layers)
    
    start_time = time.time()
    tokens_generated = 0
    forward_steps = 0
    stats = {"trusted_skips": 0, "heavy_routes": 0, "drafts_generated": 0, "drafts_accepted": 0}
    
    # --- PREFILL ---
    x = model.model.embed_tokens(prompt_ids)
    x = forward_partial(model, x, cache, 0, exit_idx)
    hidden_states = x # Capture layer 20 state
    x = forward_partial(model, x, cache, exit_idx, total_layers)
    x = model.model.norm(x)
    logits = model.lm_head(x)
    
    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
    output_tokens = [next_token.item()]
    tokens_generated += 1
    forward_steps += 1
    
    mx.eval(next_token, cache) # Compile graph
    
    # --- DECODING LOOP ---
    while tokens_generated < max_new_tokens:
        verified_len = cache[0].offset
        
        # Route
        lane = router(hidden_states[:, -1:, :]).item()
        
        if lane == 0:
            stats["trusted_skips"] += 1
            draft_tokens = []
            current_token = next_token
            
            # 1. Draft Phase (True Early Exit on Layers 0 -> 19)
            for _ in range(K):
                dx = model.model.embed_tokens(current_token)
                dx = forward_partial(model, dx, cache, 0, exit_idx)
                
                # Parasitic head
                df = drafter(dx)
                df_norm = model.model.norm(df)
                d_token = mx.argmax(model.lm_head(df_norm), axis=-1)
                
                draft_tokens.append(d_token)
                current_token = d_token
                stats["drafts_generated"] += 1
            
            draft_tensor = mx.concatenate(draft_tokens, axis=1)
            
            # Rollback cache pointers for verification!
            rollback_mlx_cache(cache, verified_len)
            
            # 2. Verification Phase (Full Model)
            verify_input = mx.concatenate([next_token, draft_tensor], axis=1)
            vx = model.model.embed_tokens(verify_input)
            
            # Pass through all layers sequentially
            for i in range(total_layers):
                if i == exit_idx:
                    verify_hidden = vx # Capture state for next router decision
                vx = model.model.layers[i](vx, mask=None, cache=cache[i])
                
            vx_norm = model.model.norm(vx)
            slow_logits = model.lm_head(vx_norm)
            forward_steps += 1
            
            # 3. Cascading Reject (Evaluated sequentially in Python for logic)
            accepted = []
            hit_eos = False
            for i in range(K):
                true_tok = mx.argmax(slow_logits[:, i:i+1, :], axis=-1)
                draft_tok = draft_tokens[i]
                
                # MLX item() forces evaluation, but it's cheap here
                if true_tok.item() == draft_tok.item():
                    accepted.append(draft_tok)
                    stats["drafts_accepted"] += 1
                    if draft_tok.item() == tokenizer.eos_token_id: hit_eos = True; break
                else:
                    accepted.append(true_tok)
                    if true_tok.item() == tokenizer.eos_token_id: hit_eos = True; break
                    break
            
            if len(accepted) == K and not hit_eos:
                bonus = mx.argmax(slow_logits[:, K:K+1, :], axis=-1)
                accepted.append(bonus)
                if bonus.item() == tokenizer.eos_token_id: hit_eos = True
            
            for t in accepted:
                output_tokens.append(t.item())
                
            tokens_generated += len(accepted)
            
            # Clean up cache and set up next loop
            rollback_mlx_cache(cache, verified_len + len(accepted))
            next_token = accepted[-1]
            hidden_states = verify_hidden[:, len(accepted)-1:len(accepted), :]
            
            if hit_eos: break

        else:
            # --- PATH B: HEAVY GLOBAL ---
            stats["heavy_routes"] += 1
            hx = model.model.embed_tokens(next_token)
            
            for i in range(total_layers):
                if i == exit_idx:
                    hidden_states = hx
                hx = model.model.layers[i](hx, mask=None, cache=cache[i])
                
            hx_norm = model.model.norm(hx)
            next_token = mx.argmax(model.lm_head(hx_norm)[:, -1:, :], axis=-1)
            
            output_tokens.append(next_token.item())
            tokens_generated += 1
            forward_steps += 1
            
            if next_token.item() == tokenizer.eos_token_id: break

        # Compile graph at the end of each step to keep memory tight
        mx.eval(next_token, cache)

    wall_time = time.time() - start_time
    return tokenizer.decode(output_tokens), tokens_generated, wall_time, forward_steps, stats

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-4bit"
    
    print("Loading 4-bit Base Model into Unified Memory...")
    model, tokenizer = load(MODEL_ID)
    
    print("Loading Custom Elastic Components...")
    hidden_size = model.model.embed_tokens.weight.shape[1]
    
    drafter = MLXMultiLayerDraftBlock(hidden_size, window_size=32, num_layers=2)
    load_npz_to_mlx_module(drafter, "mlx_drafter_weights.npz")
    
    router = MLXElasticComputeRouter(hidden_size, num_lanes=2)
    load_npz_to_mlx_module(router, "mlx_router_weights.npz")
    
    print("\nWarming up Metal GPU...")
    _, _, _ = generate_baseline_mlx(model, tokenizer, "def add(a, b):", max_new_tokens=5)
    _, _, _, _, _ = generate_elastic_mlx(model, drafter, router, tokenizer, "def add(a, b):", max_new_tokens=5, K=3)
    
    prompts = [
        "def quicksort(arr):",
        "import torch\nimport torch.nn as nn\nclass MultiHeadAttention(nn.Module):"
    ]
    
    print("\n" + "="*60)
    print(" 🏆 APPLE SILICON SHOWDOWN (M-SERIES)")
    print("="*60)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[Prompt {i+1}]: {prompt.strip()}")
        
        base_text, base_toks, base_time = generate_baseline_mlx(model, tokenizer, prompt)
        base_tps = base_toks / base_time
        
        spec_text, spec_toks, spec_time, spec_steps, stats = generate_elastic_mlx(model, drafter, router, tokenizer, prompt)
        spec_tps = spec_toks / spec_time
        speedup = spec_tps / base_tps
        
        print(f"\n--- 🐢 4-BIT AUTOREGRESSIVE ---")
        print(f"Speed: {base_tps:.2f} tokens/sec ({base_time:.2f}s)")
        
        print(f"\n--- 🚀 ELASTIC ENGINE ---")
        print(f"Speed: {spec_tps:.2f} tokens/sec ({spec_time:.2f}s)")
        print(f"Model Steps: {spec_steps} (Generated {spec_toks} tokens)")
        print(f"Router: Trusted Drafter {stats['trusted_skips']}x | Bypassed {stats['heavy_routes']}x")
        if stats['drafts_generated'] > 0:
            print(f"Drafter Accuracy: {(stats['drafts_accepted'] / stats['drafts_generated']) * 100:.1f}%")
            
        print(f"\n🔥 TRUE WALL-CLOCK SPEEDUP: {speedup:.2f}x faster!")
        print("-" * 60)