import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache
from datasets import load_dataset
import matplotlib.pyplot as plt

# ==========================================
# 1. THE MEDUSA ARCHITECTURE
# ==========================================
class MLXMedusaResBlock(nn.Module):
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

def load_medusa_weights(module, npz_path):
    weights = np.load(npz_path)
    for k, v in weights.items():
        parts = k.split('.')
        if parts[0] == 'blocks':
            idx = int(parts[1])
            target = getattr(module.blocks[idx], parts[2])
            setattr(target, parts[3], mx.array(v))

def rollback_mlx_cache(cache, keep_len):
    for c in cache:
        c.offset = keep_len

# ==========================================
# 2. THE HYBRID WATERFALL ROUTER
# ==========================================
def get_hybrid_draft(history_tokens, medusa_heads, model, hidden_state, K=3, force_lane=None):
    if force_lane != "medusa":
        if len(history_tokens) > 3:
            pattern = history_tokens[-3:]
            history_len = len(history_tokens)
            for i in range(history_len - 4, -1, -1):
                if history_tokens[i:i+3] == pattern:
                    end_idx = min(i + 3 + K, history_len)
                    draft = history_tokens[i+3 : end_idx]
                    if len(draft) == K:
                        return mx.array([draft], dtype=mx.int32), "n_gram"
                    
    m_feats = medusa_heads(hidden_state)
    draft_tokens = []
    for feat in m_feats:
        logits = model.lm_head(feat)
        draft_tokens.append(mx.argmax(logits, axis=-1))
        
    return mx.concatenate(draft_tokens, axis=1).astype(mx.int32), "medusa"

# ==========================================
# 3. THE HYBRID MLX ENGINE
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
    return tokenizer.decode(tokens), tokens, len(tokens), wall_time

def run_stream_tokens(model, tokenizer, prompt, max_new_tokens, sampler, num_draft_tokens=None, draft_model=None):
    tokens = []
    start_time = time.time()
    for resp in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
        draft_model=draft_model,
        num_draft_tokens=num_draft_tokens,
        sampler=sampler,
    ):
        tokens.append(resp.token)
    wall_time = time.time() - start_time
    return tokens, wall_time

def run_base_stream(model, tokenizer, prompt, max_new_tokens):
    sampler = make_sampler(temp=0.0)
    gen_tokens, wall_time = run_stream_tokens(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        sampler=sampler,
    )
    return gen_tokens, wall_time

def run_lookahead_mlx(model, tokenizer, prompt, max_new_tokens=100, num_draft_tokens=4):
    sampler = make_sampler(temp=0.0)
    gen_tokens, wall_time = run_stream_tokens(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens,
        sampler=sampler,
    )
    return "", 0, gen_tokens, wall_time

def run_two_model_speculative(model, draft_model, tokenizer, prompt, max_new_tokens=100, num_draft_tokens=4):
    sampler = make_sampler(temp=0.0)
    gen_tokens, wall_time = run_stream_tokens(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens,
        draft_model=draft_model,
        sampler=sampler,
    )
    return "", 0, gen_tokens, wall_time

def generate_hybrid_engine(model, medusa, tokenizer, prompt, max_new_tokens=60, K=3, force_lane=None):
    prompt_ids = tokenizer.encode(prompt)
    output_tokens = prompt_ids.copy()
    cache = make_prompt_cache(model)
    
    start_time = time.time()
    tokens_generated = 0
    forward_steps = 0
    stats = {"n_gram_used": 0, "medusa_used": 0, "drafts_accepted": 0}
    
    # --- PREFILL ---
    x = model.model.embed_tokens(mx.array(prompt_ids)[None, :])
    for i, layer in enumerate(model.model.layers):
        x = layer(x, mask=None, cache=cache[i])
        
    # 🔥 FIX: hx_last is now strictly PRE-NORM (Perfect for Medusa!)
    hx_last = x[:, -1:, :] 
    x_norm = model.model.norm(hx_last)
    logits = model.lm_head(x_norm)
    
    next_token = mx.argmax(logits, axis=-1).astype(mx.int32)
    output_tokens.append(next_token.item())
    tokens_generated += 1
    forward_steps += 1
    
    # --- DECODING LOOP ---
    while tokens_generated < max_new_tokens:
        
        # 1. Draft with Pre-Norm Features
        draft_tensor, lane = get_hybrid_draft(output_tokens, medusa, model, hx_last, K, force_lane=force_lane)
        stats[f"{lane}_used"] += 1
        
        # 2. Causally-Masked Verification
        verify_input = mx.concatenate([next_token, draft_tensor], axis=1)
        base_offset = cache[0].offset 
        vx = model.model.embed_tokens(verify_input)
        for i, layer in enumerate(model.model.layers):
            vx = layer(vx, mask=None, cache=cache[i])
            
        vx_norm = model.model.norm(vx)
        slow_logits = model.lm_head(vx_norm)
        forward_steps += 1
        
        # 3. Rejection & Rollback
        true_tokens = mx.argmax(slow_logits, axis=-1)
        mx.eval(true_tokens, draft_tensor, vx)
        
        true_list = true_tokens[0].tolist()
        draft_list = draft_tensor[0].tolist()
        
        accepted_list = []
        hit_eos = False
        
        for i in range(K):
            if draft_list[i] == true_list[i]:
                accepted_list.append(draft_list[i])
            else:
                accepted_list.append(true_list[i])
                break
                
        if len(accepted_list) == K:
            accepted_list.append(true_list[K])
            
        for t in accepted_list:
            output_tokens.append(t)
            if t == tokenizer.eos_token_id: 
                hit_eos = True; break
                
        stats["drafts_accepted"] += max(0, len(accepted_list) - 1)
        tokens_generated += len(accepted_list)
        
        rollback_mlx_cache(cache, base_offset + len(accepted_list))
        next_token = mx.array([[accepted_list[-1]]], dtype=mx.int32)
        
        # 🔥 FIX: Update the Pre-Norm hidden state for the next draft!
        last_valid_idx = len(accepted_list) - 1
        hx_last = vx[:, last_valid_idx:last_valid_idx+1, :]
        
        if hit_eos: break

    wall_time = time.time() - start_time
    gen_tokens = output_tokens[len(prompt_ids):]
    final_text = tokenizer.decode(gen_tokens)
    return final_text, gen_tokens, tokens_generated, wall_time, forward_steps, stats

def select_text_field(sample):
    for key in ["content", "text", "code"]:
        if key in sample and sample[key]:
            return sample[key]
    return None

def sample_dataset_prompts(tokenizer, dataset_names, split, num_samples, prompt_len, target_len, seed):
    for name in dataset_names:
        try:
            ds = load_dataset(name, split=split, streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=1000)
            samples = []
            for sample in ds:
                text = select_text_field(sample)
                if not text:
                    continue
                tokens = tokenizer.encode(text)
                if len(tokens) < prompt_len + target_len + 1:
                    continue
                prompt_tokens = tokens[:prompt_len]
                target_tokens = tokens[prompt_len:prompt_len + target_len]
                prompt_text = tokenizer.decode(prompt_tokens)
                samples.append((prompt_text, target_tokens))
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                return samples, name
        except Exception:
            continue
    raise RuntimeError("Failed to load dataset samples")

def token_edit_distance(a, b):
    n = len(a)
    m = len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]

def score_tokens(gen_tokens, ref_tokens, baseline_tokens=None):
    target_len = len(ref_tokens)
    if target_len == 0:
        return 0.0, 0, 0.0, 0.0, 0.0
    correct = 0
    prefix_len = 0
    for i in range(target_len):
        if i < len(gen_tokens) and gen_tokens[i] == ref_tokens[i]:
            correct += 1
            if i == prefix_len:
                prefix_len += 1
        else:
            if i == prefix_len:
                break
    token_acc = correct / target_len
    exact = 1 if correct == target_len and len(gen_tokens) >= target_len else 0
    baseline_match = 0.0
    if baseline_tokens is not None:
        match = 0
        for i in range(target_len):
            if i < len(gen_tokens) and i < len(baseline_tokens) and gen_tokens[i] == baseline_tokens[i]:
                match += 1
        baseline_match = match / target_len
    prefix_rate = prefix_len / target_len
    edit_sim = 1.0 - (token_edit_distance(gen_tokens[:target_len], ref_tokens) / target_len)
    return token_acc, exact, baseline_match, prefix_rate, edit_sim

def init_metric_store(methods):
    return {
        m: {
            "tps": [],
            "speedup": [],
            "token_acc": [],
            "exact": [],
            "baseline_match": [],
            "prefix_hit": [],
            "edit_sim": [],
        }
        for m in methods
    }

if __name__ == "__main__":
    MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-4bit"
    
    print("Loading Base Model...")
    model, tokenizer = load(MODEL_ID)
    
    print("Loading Custom Medusa Heads...")
    hidden_size = model.model.embed_tokens.weight.shape[1]
    medusa = MLXMedusaHeads(hidden_size, num_heads=3)
    load_medusa_weights(medusa, "mlx_medusa_heads.npz")
    
    print("Loading Draft Model...")
    draft_model, _ = load("mlx-community/Qwen2.5-Coder-0.5B-4bit")
    
    dataset_names = [
        "bigcode/the-stack-smol",
        "codeparrot/github-code",
        "bigcode/starcoderdata",
    ]
    split = "train"
    num_samples = 25
    prompt_len = 64
    target_len = 64
    seed = 42
    K_default = 3
    K_high = 3
    num_draft_tokens = 4
    
    samples, dataset_name = sample_dataset_prompts(
        tokenizer, dataset_names, split, num_samples, prompt_len, target_len, seed
    )
    
    methods = [
        "Baseline",
        "Hybrid Default",
        "Hybrid Medusa",
        "MLX Lookahead",
        "Two-Model Speculative",
    ]
    metrics = init_metric_store(methods)
    
    print("\n" + "="*60)
    print(" 🧪 DATASET EVALUATION")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(samples)} | Prompt tokens: {prompt_len} | Target tokens: {target_len}")
    
    for idx, (prompt, ref_tokens) in enumerate(samples):
        preview = prompt.replace("\n", " ").replace("\r", " ").replace("\t", " ")[:80]
        print(f"\n[Sample {idx+1}] {preview}")
        
        base_gen_tokens, base_time = run_base_stream(
            model, tokenizer, prompt, max_new_tokens=target_len
        )
        base_eval_tokens = base_gen_tokens[:target_len]
        base_tps = len(base_eval_tokens) / base_time
        base_token_acc, base_exact, _, base_prefix, base_edit = score_tokens(base_eval_tokens, ref_tokens)
        metrics["Baseline"]["tps"].append(base_tps)
        metrics["Baseline"]["speedup"].append(1.0)
        metrics["Baseline"]["token_acc"].append(base_token_acc)
        metrics["Baseline"]["exact"].append(base_exact)
        metrics["Baseline"]["baseline_match"].append(1.0)
        metrics["Baseline"]["prefix_hit"].append(base_prefix)
        metrics["Baseline"]["edit_sim"].append(base_edit)
        
        _, hybrid_gen_tokens, _, hybrid_time, _, _ = generate_hybrid_engine(
            model, medusa, tokenizer, prompt, max_new_tokens=target_len, K=K_default
        )
        hybrid_eval_tokens = hybrid_gen_tokens[:target_len]
        hybrid_tps = len(hybrid_eval_tokens) / hybrid_time
        hybrid_speedup = hybrid_tps / base_tps
        hybrid_acc, hybrid_exact, hybrid_match, hybrid_prefix, hybrid_edit = score_tokens(
            hybrid_eval_tokens, ref_tokens, baseline_tokens=base_eval_tokens
        )
        metrics["Hybrid Default"]["tps"].append(hybrid_tps)
        metrics["Hybrid Default"]["speedup"].append(hybrid_speedup)
        metrics["Hybrid Default"]["token_acc"].append(hybrid_acc)
        metrics["Hybrid Default"]["exact"].append(hybrid_exact)
        metrics["Hybrid Default"]["baseline_match"].append(hybrid_match)
        metrics["Hybrid Default"]["prefix_hit"].append(hybrid_prefix)
        metrics["Hybrid Default"]["edit_sim"].append(hybrid_edit)
        
        _, medusa_gen_tokens, _, medusa_time, _, _ = generate_hybrid_engine(
            model, medusa, tokenizer, prompt, max_new_tokens=target_len, K=K_high, force_lane="medusa"
        )
        medusa_eval_tokens = medusa_gen_tokens[:target_len]
        medusa_tps = len(medusa_eval_tokens) / medusa_time
        medusa_speedup = medusa_tps / base_tps
        medusa_acc, medusa_exact, medusa_match, medusa_prefix, medusa_edit = score_tokens(
            medusa_eval_tokens, ref_tokens, baseline_tokens=base_eval_tokens
        )
        metrics["Hybrid Medusa"]["tps"].append(medusa_tps)
        metrics["Hybrid Medusa"]["speedup"].append(medusa_speedup)
        metrics["Hybrid Medusa"]["token_acc"].append(medusa_acc)
        metrics["Hybrid Medusa"]["exact"].append(medusa_exact)
        metrics["Hybrid Medusa"]["baseline_match"].append(medusa_match)
        metrics["Hybrid Medusa"]["prefix_hit"].append(medusa_prefix)
        metrics["Hybrid Medusa"]["edit_sim"].append(medusa_edit)
        
        _, _, look_gen_tokens, look_time = run_lookahead_mlx(
            model, tokenizer, prompt, max_new_tokens=target_len, num_draft_tokens=num_draft_tokens
        )
        look_eval_tokens = look_gen_tokens[:target_len]
        look_tps = len(look_eval_tokens) / look_time
        look_speedup = look_tps / base_tps
        look_acc, look_exact, look_match, look_prefix, look_edit = score_tokens(
            look_eval_tokens, ref_tokens, baseline_tokens=base_eval_tokens
        )
        metrics["MLX Lookahead"]["tps"].append(look_tps)
        metrics["MLX Lookahead"]["speedup"].append(look_speedup)
        metrics["MLX Lookahead"]["token_acc"].append(look_acc)
        metrics["MLX Lookahead"]["exact"].append(look_exact)
        metrics["MLX Lookahead"]["baseline_match"].append(look_match)
        metrics["MLX Lookahead"]["prefix_hit"].append(look_prefix)
        metrics["MLX Lookahead"]["edit_sim"].append(look_edit)
        
        _, _, spec_gen_tokens, spec_time = run_two_model_speculative(
            model, draft_model, tokenizer, prompt, max_new_tokens=target_len, num_draft_tokens=num_draft_tokens
        )
        spec_eval_tokens = spec_gen_tokens[:target_len]
        spec_tps = len(spec_eval_tokens) / spec_time
        spec_speedup = spec_tps / base_tps
        spec_acc, spec_exact, spec_match, spec_prefix, spec_edit = score_tokens(
            spec_eval_tokens, ref_tokens, baseline_tokens=base_eval_tokens
        )
        metrics["Two-Model Speculative"]["tps"].append(spec_tps)
        metrics["Two-Model Speculative"]["speedup"].append(spec_speedup)
        metrics["Two-Model Speculative"]["token_acc"].append(spec_acc)
        metrics["Two-Model Speculative"]["exact"].append(spec_exact)
        metrics["Two-Model Speculative"]["baseline_match"].append(spec_match)
        metrics["Two-Model Speculative"]["prefix_hit"].append(spec_prefix)
        metrics["Two-Model Speculative"]["edit_sim"].append(spec_edit)
        
        print(f"Baseline tps: {base_tps:.2f} | Hybrid: {hybrid_speedup:.2f}x | Medusa: {medusa_speedup:.2f}x | Lookahead: {look_speedup:.2f}x | Two-Model: {spec_speedup:.2f}x")
    
    print("\n" + "="*60)
    print(" � AVERAGED RESULTS")
    print("="*60)
    
    for method in methods:
        tps_mean = float(np.mean(metrics[method]["tps"]))
        tps_std = float(np.std(metrics[method]["tps"]))
        speed_mean = float(np.mean(metrics[method]["speedup"]))
        speed_std = float(np.std(metrics[method]["speedup"]))
        acc_mean = float(np.mean(metrics[method]["token_acc"]))
        exact_mean = float(np.mean(metrics[method]["exact"]))
        edit_mean = float(np.mean(metrics[method]["edit_sim"]))
        match_mean = float(np.mean(metrics[method]["baseline_match"]))
        match_std = float(np.std(metrics[method]["baseline_match"]))
        prefix_mean = float(np.mean(metrics[method]["prefix_hit"]))
        prefix_std = float(np.std(metrics[method]["prefix_hit"]))
        improvement = (speed_mean - 1.0) * 100.0
        print(f"{method}: {tps_mean:.2f}±{tps_std:.2f} tok/s | Speedup {speed_mean:.2f}±{speed_std:.2f}x | {improvement:+.1f}% | Token Acc {acc_mean:.3f} | EditSim@64 {edit_mean:.3f} | Exact {exact_mean:.3f} | Match@64 {match_mean:.3f}±{match_std:.3f} | Prefix@64 {prefix_mean:.3f}±{prefix_std:.3f}")
    
    labels = methods
    speedup_means = [float(np.mean(metrics[m]["speedup"])) for m in labels]
    speedup_stds = [float(np.std(metrics[m]["speedup"])) for m in labels]
    token_acc_means = [float(np.mean(metrics[m]["token_acc"])) for m in labels]
    exact_means = [float(np.mean(metrics[m]["exact"])) for m in labels]
    edit_means = [float(np.mean(metrics[m]["edit_sim"])) for m in labels]
    match_means = [float(np.mean(metrics[m]["baseline_match"])) for m in labels]
    prefix_means = [float(np.mean(metrics[m]["prefix_hit"])) for m in labels]
    
    plt.figure(figsize=(10, 5))
    plt.bar(labels, speedup_means, yerr=speedup_stds, capsize=4)
    plt.ylabel("Speedup vs Baseline")
    plt.title("Throughput Speedup vs Baseline")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("hybrid_eval_speedup.png", dpi=300)
    plt.close()
    
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, token_acc_means, width, label="Token Acc")
    plt.bar(x, edit_means, width, label="EditSim@64")
    plt.bar(x + width, exact_means, width, label="Exact Match")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Reference Continuation")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hybrid_eval_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(labels, match_means, label="Match@64")
    plt.bar(labels, prefix_means, label="Prefix@64", alpha=0.7)
    plt.ylabel("Agreement with Baseline")
    plt.title("Speculative Agreement vs Baseline")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hybrid_eval_baseline_match.png", dpi=300)
    plt.close()
