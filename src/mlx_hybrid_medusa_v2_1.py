# mlx_hybrid_medusa_v3_dynamic_eval.py
# ============================================================
# MLX Hybrid Inference (Aligned with medusa_train_kd_adaptive.py) + Evaluation
#
# Adds:
#   1) Dynamic K (adaptive m per step) based on Medusa confidence + optional N-gram lane
#   2) Verify only (1 + m) tokens (reduces verifier work when confidence is low)
#   3) Low-overhead acceptance (no true_tokens.tolist(), no Python token-by-token loops)
#   4) Faster N-gram lookup via incremental hashmap index
#   5) Evaluation: throughput + speedup + correctness (token acc / exact / edit sim / prefix hit / baseline match)
#
# IMPORTANT alignment note:
#   - If your training uses student_logits = lm_head(norm(feat)) then set APPLY_NORM_BEFORE_LM_HEAD=True (recommended).
#   - Otherwise set it False, but acceptance will likely be lower if verifier uses norm().
# ============================================================

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache
from datasets import load_dataset


# ============================================================
# 1) MEDUSA (MLX)
# ============================================================
class MLXMedusaResBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x):
        return x + nn.silu(self.linear(x))


class MLXMedusaHeads(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.blocks = [MLXMedusaResBlock(hidden_size) for _ in range(num_heads)]

    def __call__(self, x):
        feats = []
        cur = x
        for block in self.blocks:
            cur = block(cur)
            feats.append(cur)
        return feats


def load_medusa_weights(module: MLXMedusaHeads, npz_path: str):
    """
    Load weights into MLX Medusa heads. Expected keys like:
      blocks.{i}.linear.weight
    """
    weights = np.load(npz_path)
    for k, v in weights.items():
        parts = k.split(".")
        if parts[0] != "blocks":
            continue
        idx = int(parts[1])
        # blocks, {idx}, linear, weight
        target = getattr(module.blocks[idx], parts[2])
        setattr(target, parts[3], mx.array(v))


def rollback_mlx_cache(cache, keep_len: int):
    for c in cache:
        c.offset = keep_len


# ============================================================
# 2) FAST N-GRAM INDEX (INCREMENTAL)
# ============================================================
class NGramIndex:
    """
    Incremental n-gram index for tokens. Maintains last-seen positions for n-grams.
    Avoids O(n) backward scans each step.
    """
    def __init__(self, n: int = 3):
        self.n = n
        self.map = {}  # tuple(tokens[i:i+n]) -> last i

    def build(self, tokens):
        self.map.clear()
        n = self.n
        if len(tokens) < n:
            return
        # record last occurrence start index for each n-gram
        for i in range(len(tokens) - n):
            self.map[tuple(tokens[i:i+n])] = i

    def update_with_new_tokens(self, tokens, new_tokens):
        n = self.n
        L = len(tokens)
        # update windows that could include new tail tokens
        start_min = max(0, L - len(new_tokens) - n)
        start_max = max(0, L - n)
        for i in range(start_min, start_max + 1):
            if i + n <= L:
                self.map[tuple(tokens[i:i+n])] = i

    def try_draft(self, tokens, K: int):
        """
        If the last n tokens occurred before, draft next K tokens from that earlier location.
        """
        n = self.n
        if len(tokens) < n + K:
            return None
        key = tuple(tokens[-n:])
        pos = self.map.get(key, None)
        if pos is None:
            return None
        start = pos + n
        end = start + K
        if end <= len(tokens):
            draft = tokens[start:end]
            if len(draft) == K:
                return draft
        return None


# ============================================================
# 3) MEDUSA DRAFT + CONFIDENCE (ALIGNED WITH VERIFIER)
# ============================================================
def _logits_from_feat(model, feat, apply_norm_before_lm_head: bool):
    if apply_norm_before_lm_head:
        return model.lm_head(model.model.norm(feat))
    return model.lm_head(feat)


def medusa_draft_tokens_and_conf(
    model,
    medusa: MLXMedusaHeads,
    hx_last_pre_norm,          # [1,1,H]
    K: int,
    apply_norm_before_lm_head: bool = True,
):
    """
    Returns:
      draft_tokens: [1,K] int32 (argmax per head)
      conf: dict with per-head margins + entropies (mx arrays)
    """
    feats = medusa(hx_last_pre_norm)  # list of [1,1,H]
    toks = []
    margins = []
    entropies = []

    for f in feats[:K]:
        logits = _logits_from_feat(model, f, apply_norm_before_lm_head)  # [1,1,V]
        idx_sorted = mx.argsort(logits, axis=-1)                         # [1,1,V]
        top2_idx = idx_sorted[..., -2:]                                  # [1,1,2]
        top1_idx = top2_idx[..., -1:]                                    # [1,1,1]
        top2_idx_only = top2_idx[..., -2:-1]                             # [1,1,1]
        top1_val = mx.take_along_axis(logits, top1_idx, axis=-1)         # [1,1,1]
        top2_val = mx.take_along_axis(logits, top2_idx_only, axis=-1)    # [1,1,1]
        p = mx.softmax(logits, axis=-1)                      # [1,1,V]
        # entropy
        ent = -mx.sum(p * mx.log(p + 1e-9), axis=-1)         # [1,1]
        # margin = top1 - top2 (logit margin)
        m = top1_val[..., 0] - top2_val[..., 0]              # [1,1]

        tok = top1_idx[..., 0]                               # [1,1]
        toks.append(tok)
        margins.append(m)
        entropies.append(ent)

    draft_tokens = mx.concatenate(toks, axis=1).astype(mx.int32)         # [1,K]
    margins = mx.concatenate(margins, axis=1)                            # [1,K]
    entropies = mx.concatenate(entropies, axis=1)                        # [1,K]
    return draft_tokens, {"margins": margins, "entropies": entropies}


def choose_dynamic_m_from_conf(
    margins: mx.array,          # [1,K]
    entropies: mx.array,        # [1,K]
    K: int,
    margin_thresh: float = 2.0,
    entropy_thresh: float = 4.0,
    min_m: int = 1,
):
    """
    Decide m in [1..K] using simple thresholds:
      accept head k if margin >= margin_thresh AND entropy <= entropy_thresh
    Returns an int m with only ONE scalar host read.
    """
    ok = (margins >= margin_thresh) & (entropies <= entropy_thresh)   # [1,K] bool
    ok_i = ok.astype(mx.int32)[0]                                     # [K]
    prefix = mx.cumprod(ok_i, axis=0)                                 # [K] 1s until first 0
    m = int(mx.sum(prefix).item())                                    # one scalar host read
    if m < min_m:
        m = min_m
    if m > K:
        m = K
    return m


# ============================================================
# 4) ON-DEVICE PREFIX ACCEPT (NO TRUE_TOKENS .tolist())
# ============================================================
def accept_prefix_len(draft_tensor, true_tokens, m: int) -> int:
    """
    draft_tensor: [1,m]
    true_tokens:  [1,m+1] (argmax logits for verify block)
    Returns prefix_len in [0..m], with one scalar host read.
    """
    match = (draft_tensor == true_tokens[:, :m])          # [1,m]
    match_i = match.astype(mx.int32)[0]                   # [m]
    prefix = mx.cumprod(match_i, axis=0)                  # [m]
    prefix_len = int(mx.sum(prefix).item())               # one scalar read
    return prefix_len


def build_accepted_tensor(draft_tensor, true_tokens, prefix_len: int, m: int):
    """
    accepted:
      - accept draft prefix of length prefix_len
      - if prefix_len < m: accept true token at mismatch position
      - else: accept true token at position m (next token after m drafts)
    """
    accepted_prefix = draft_tensor[:, :prefix_len]  # [1,prefix_len]
    if prefix_len < m:
        next_tok = true_tokens[:, prefix_len:prefix_len+1]
        accepted = mx.concatenate([accepted_prefix, next_tok], axis=1)
    else:
        next_tok = true_tokens[:, m:m+1]
        accepted = mx.concatenate([accepted_prefix, next_tok], axis=1)
    return accepted.astype(mx.int32)


# ============================================================
# 5) HYBRID ENGINE V3: DYNAMIC m + VERIFY ONLY (1+m)
# ============================================================
def generate_hybrid_engine_v3(
    model,
    medusa: MLXMedusaHeads,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    K: int = 3,
    force_lane: str | None = None,
    ngram_n: int = 3,
    apply_norm_before_lm_head: bool = True,
    dynamic_k: bool = True,
    # Confidence thresholds (tune)
    margin_thresh: float = 2.0,
    entropy_thresh: float = 4.0,
    min_m: int = 1,
):
    """
    Low-overhead hybrid engine with dynamic m:
      - propose up to K tokens (ngram or medusa)
      - if dynamic_k: choose m <= K based on Medusa confidence
      - verify only (1+m) tokens
      - accept prefix using on-device compare
      - rollback cache to accepted length
    """
    prompt_ids = tokenizer.encode(prompt)
    output_tokens = list(prompt_ids)

    # N-gram index
    ngram = NGramIndex(n=ngram_n)
    ngram.build(output_tokens)

    cache = make_prompt_cache(model)

    start_time = time.time()
    tokens_generated = 0
    forward_calls = 0
    stats = {
        "n_gram_used": 0,
        "medusa_used": 0,
        "draft_tokens_accepted": 0,
        "verify_tokens": 0,
        "avg_m": 0.0,
        "m_counts": {i: 0 for i in range(1, K + 1)},
    }

    # -----------------------
    # PREFILL
    # -----------------------
    x = model.model.embed_tokens(mx.array(prompt_ids, dtype=mx.int32)[None, :])
    for i, layer in enumerate(model.model.layers):
        x = layer(x, mask=None, cache=cache[i])

    hx_last = x[:, -1:, :]  # pre-norm

    logits0 = model.lm_head(model.model.norm(hx_last))
    next_token = mx.argmax(logits0, axis=-1).astype(mx.int32)
    mx.eval(next_token)

    nt = int(next_token.item())
    output_tokens.append(nt)
    ngram.update_with_new_tokens(output_tokens, [nt])

    tokens_generated += 1
    forward_calls += 1

    # -----------------------
    # DECODE LOOP
    # -----------------------
    while tokens_generated < max_new_tokens:
        # 1) Draft proposal
        lane = "medusa"
        draft_tensor = None
        m = K

        if force_lane != "medusa":
            draft_list = ngram.try_draft(output_tokens, K)
            if draft_list is not None:
                lane = "n_gram"
                draft_tensor = mx.array([draft_list], dtype=mx.int32)
                # n-gram drafts are "exact-match repeats": if found, trust full K unless user disables
                m = K
                stats["n_gram_used"] += 1

        if draft_tensor is None:
            # Medusa propose + confidence
            draft_full, conf = medusa_draft_tokens_and_conf(
                model=model,
                medusa=medusa,
                hx_last_pre_norm=hx_last,
                K=K,
                apply_norm_before_lm_head=apply_norm_before_lm_head,
            )
            stats["medusa_used"] += 1

            if dynamic_k:
                # Choose m based on confidence (one scalar host read)
                m = choose_dynamic_m_from_conf(
                    margins=conf["margins"],
                    entropies=conf["entropies"],
                    K=K,
                    margin_thresh=margin_thresh,
                    entropy_thresh=entropy_thresh,
                    min_m=min_m,
                )
            else:
                m = K

            draft_tensor = draft_full[:, :m].astype(mx.int32)

        stats["m_counts"][m] += 1
        # We'll update avg_m at end
        # 2) Verify only (1+m)
        verify_input = mx.concatenate([next_token, draft_tensor], axis=1)  # [1,1+m]
        base_offset = cache[0].offset

        vx = model.model.embed_tokens(verify_input)
        for i, layer in enumerate(model.model.layers):
            vx = layer(vx, mask=None, cache=cache[i])

        slow_logits = model.lm_head(model.model.norm(vx))                 # [1,1+m,V]
        true_tokens = mx.argmax(slow_logits, axis=-1).astype(mx.int32)    # [1,1+m]
        mx.eval(true_tokens, draft_tensor, vx)

        forward_calls += 1
        stats["verify_tokens"] += (m + 1)

        # 3) Accept / reject (on-device)
        p = accept_prefix_len(draft_tensor, true_tokens, m)
        accepted = build_accepted_tensor(draft_tensor, true_tokens, p, m)  # [1,<=m+1]
        mx.eval(accepted)

        accepted_len = accepted.shape[1]
        rollback_mlx_cache(cache, base_offset + accepted_len)

        accepted_host = accepted[0].tolist()   # one host conversion per verify step
        output_tokens.extend(accepted_host)
        ngram.update_with_new_tokens(output_tokens, accepted_host)

        tokens_generated += accepted_len
        stats["draft_tokens_accepted"] += max(0, accepted_len - 1)

        last_tok = accepted_host[-1]
        next_token = mx.array([[last_tok]], dtype=mx.int32)

        last_idx = accepted_len - 1
        hx_last = vx[:, last_idx:last_idx+1, :]

        if last_tok == tokenizer.eos_token_id:
            break

    wall_time = time.time() - start_time
    gen_tokens = output_tokens[len(prompt_ids):]
    final_text = tokenizer.decode(gen_tokens)

    # finalize avg m
    total_steps = sum(stats["m_counts"].values())
    if total_steps > 0:
        stats["avg_m"] = sum(k * v for k, v in stats["m_counts"].items()) / total_steps

    return final_text, gen_tokens, tokens_generated, wall_time, forward_calls, stats


# ============================================================
# 6) Baselines (MLX stream_generate)
# ============================================================
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
        model, tokenizer, prompt, max_new_tokens=max_new_tokens, sampler=sampler
    )
    return gen_tokens, wall_time


def run_lookahead_mlx(model, tokenizer, prompt, max_new_tokens=100, num_draft_tokens=4):
    sampler = make_sampler(temp=0.0)
    gen_tokens, wall_time = run_stream_tokens(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens, sampler=sampler
    )
    return gen_tokens, wall_time


def run_two_model_speculative(model, draft_model, tokenizer, prompt, max_new_tokens=100, num_draft_tokens=4):
    sampler = make_sampler(temp=0.0)
    gen_tokens, wall_time = run_stream_tokens(
        model, tokenizer, prompt, max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens, draft_model=draft_model, sampler=sampler
    )
    return gen_tokens, wall_time


# ============================================================
# 7) Correctness Metrics
# ============================================================
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


# ============================================================
# 8) Dataset prompt sampling (reference = next tokens from dataset text)
# ============================================================
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
                toks = tokenizer.encode(text)
                if len(toks) < prompt_len + target_len + 1:
                    continue
                prompt_tokens = toks[:prompt_len]
                ref_tokens = toks[prompt_len:prompt_len + target_len]
                prompt_text = tokenizer.decode(prompt_tokens)
                samples.append((prompt_text, ref_tokens))
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                return samples, name
        except Exception:
            continue
    raise RuntimeError("Failed to load dataset samples")


# ============================================================
# 9) Main: Evaluate speed + correctness
# ============================================================
if __name__ == "__main__":
    BASE_MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-4bit"
    DRAFT_MODEL_ID = "mlx-community/Qwen2.5-Coder-0.5B-4bit"

    # If you changed training as recommended:
    #   student_logits = lm_head(norm(feat))
    # then keep this True.
    APPLY_NORM_BEFORE_LM_HEAD = True

    # Dynamic-k policy knobs (tune):
    K = 3
    DYNAMIC_K = True
    MARGIN_THRESH = 2.0
    ENTROPY_THRESH = 4.0
    MIN_M = 1

    # Eval setup
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

    num_draft_tokens = 4  # for MLX lookahead and two-model speculative baselines

    print("Loading base model...")
    model, tokenizer = load(BASE_MODEL_ID)

    print("Loading Medusa heads...")
    hidden_size = model.model.embed_tokens.weight.shape[1]
    medusa = MLXMedusaHeads(hidden_size, num_heads=K)
    load_medusa_weights(medusa, "mlx_medusa_heads.npz")

    print("Loading draft model for baseline two-model speculative...")
    draft_model, _ = load(DRAFT_MODEL_ID)

    print("Sampling dataset prompts...")
    samples, dataset_name = sample_dataset_prompts(
        tokenizer, dataset_names, split, num_samples, prompt_len, target_len, seed
    )

    methods = [
        "Baseline",
        "HybridV3 Dynamic",
        "HybridV3 MedusaOnly",
        "MLX Lookahead",
        "Two-Model Speculative",
    ]
    metrics = init_metric_store(methods)

    print("\n" + "=" * 70)
    print("🧪 DATASET EVALUATION (Speed + Correctness)")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(samples)} | Prompt tokens: {prompt_len} | Target tokens: {target_len}")
    print(f"HybridV3: K={K}, dynamic={DYNAMIC_K}, margin>={MARGIN_THRESH}, entropy<={ENTROPY_THRESH}, min_m={MIN_M}")

    for idx, (prompt, ref_tokens) in enumerate(samples):
        preview = prompt.replace("\n", " ").replace("\r", " ").replace("\t", " ")[:80]
        print(f"\n[Sample {idx+1}] {preview}")

        # -------- Baseline --------
        base_gen_tokens, base_time = run_base_stream(model, tokenizer, prompt, max_new_tokens=target_len)
        base_eval_tokens = base_gen_tokens[:target_len]
        base_tps = len(base_eval_tokens) / max(base_time, 1e-9)

        base_acc, base_exact, _, base_prefix, base_edit = score_tokens(base_eval_tokens, ref_tokens)
        metrics["Baseline"]["tps"].append(base_tps)
        metrics["Baseline"]["speedup"].append(1.0)
        metrics["Baseline"]["token_acc"].append(base_acc)
        metrics["Baseline"]["exact"].append(base_exact)
        metrics["Baseline"]["baseline_match"].append(1.0)
        metrics["Baseline"]["prefix_hit"].append(base_prefix)
        metrics["Baseline"]["edit_sim"].append(base_edit)

        # -------- HybridV3 Dynamic --------
        _, hyb_tokens, _, hyb_time, _, hyb_stats = generate_hybrid_engine_v3(
            model=model,
            medusa=medusa,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=target_len,
            K=K,
            force_lane=None,
            ngram_n=3,
            apply_norm_before_lm_head=APPLY_NORM_BEFORE_LM_HEAD,
            dynamic_k=DYNAMIC_K,
            margin_thresh=MARGIN_THRESH,
            entropy_thresh=ENTROPY_THRESH,
            min_m=MIN_M,
        )
        hyb_eval_tokens = hyb_tokens[:target_len]
        hyb_tps = len(hyb_eval_tokens) / max(hyb_time, 1e-9)
        hyb_speedup = hyb_tps / max(base_tps, 1e-9)
        hyb_acc, hyb_exact, hyb_match, hyb_prefix, hyb_edit = score_tokens(
            hyb_eval_tokens, ref_tokens, baseline_tokens=base_eval_tokens
        )
        metrics["HybridV3 Dynamic"]["tps"].append(hyb_tps)
        metrics["HybridV3 Dynamic"]["speedup"].append(hyb_speedup)
        metrics["HybridV3 Dynamic"]["token_acc"].append(hyb_acc)
        metrics["HybridV3 Dynamic"]["exact"].append(hyb_exact)
        metrics["HybridV3 Dynamic"]["baseline_match"].append(hyb_match)
        metrics["HybridV3 Dynamic"]["prefix_hit"].append(hyb_prefix)
        metrics["HybridV3 Dynamic"]["edit_sim"].append(hyb_edit)

        # -------- HybridV3 MedusaOnly (force_lane="medusa") --------
        _, med_tokens, _, med_time, _, med_stats = generate_hybrid_engine_v3(
            model=model,
            medusa=medusa,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=target_len,
            K=K,
            force_lane="medusa",
            ngram_n=3,
            apply_norm_before_lm_head=APPLY_NORM_BEFORE_LM_HEAD,
            dynamic_k=DYNAMIC_K,
            margin_thresh=MARGIN_THRESH,
            entropy_thresh=ENTROPY_THRESH,
            min_m=MIN_M,
        )
        med_eval_tokens = med_tokens[:target_len]
        med_tps = len(med_eval_tokens) / max(med_time, 1e-9)
        med_speedup = med_tps / max(base_tps, 1e-9)
        med_acc, med_exact, med_match, med_prefix, med_edit = score_tokens(
            med_eval_tokens, ref_tokens, baseline_tokens=base_eval_tokens
        )
        metrics["HybridV3 MedusaOnly"]["tps"].append(med_tps)
        metrics["HybridV3 MedusaOnly"]["speedup"].append(med_speedup)
        metrics["HybridV3 MedusaOnly"]["token_acc"].append(med_acc)
        metrics["HybridV3 MedusaOnly"]["exact"].append(med_exact)
        metrics["HybridV3 MedusaOnly"]["baseline_match"].append(med_match)
        metrics["HybridV3 MedusaOnly"]["prefix_hit"].append(med_prefix)
        metrics["HybridV3 MedusaOnly"]["edit_sim"].append(med_edit)

        # -------- MLX Lookahead (single-model) --------
        look_tokens, look_time = run_lookahead_mlx(
            model, tokenizer, prompt, max_new_tokens=target_len, num_draft_tokens=num_draft_tokens
        )
        look_eval_tokens = look_tokens[:target_len]
        look_tps = len(look_eval_tokens) / max(look_time, 1e-9)
        look_speedup = look_tps / max(base_tps, 1e-9)
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

        # -------- Two-model speculative baseline --------
        spec_tokens, spec_time = run_two_model_speculative(
            model, draft_model, tokenizer, prompt, max_new_tokens=target_len, num_draft_tokens=num_draft_tokens
        )
        spec_eval_tokens = spec_tokens[:target_len]
        spec_tps = len(spec_eval_tokens) / max(spec_time, 1e-9)
        spec_speedup = spec_tps / max(base_tps, 1e-9)
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

        print(
            f"Baseline {base_tps:.2f} t/s | "
            f"HybridDyn {hyb_speedup:.2f}x | MedusaOnly {med_speedup:.2f}x | "
            f"Lookahead {look_speedup:.2f}x | TwoModel {spec_speedup:.2f}x"
        )
        print(f"  HybridDyn stats: avg_m={hyb_stats['avg_m']:.2f}, verify_tokens={hyb_stats['verify_tokens']}, "
              f"accepted_draft_tokens={hyb_stats['draft_tokens_accepted']}, lane(n={hyb_stats['n_gram_used']},m={hyb_stats['medusa_used']})")

    print("\n" + "=" * 70)
    print("📊 AVERAGED RESULTS")
    print("=" * 70)

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

        print(
            f"{method}: {tps_mean:.2f}±{tps_std:.2f} tok/s | "
            f"Speedup {speed_mean:.2f}±{speed_std:.2f}x ({improvement:+.1f}%) | "
            f"TokenAcc {acc_mean:.3f} | EditSim@{target_len} {edit_mean:.3f} | "
            f"Exact {exact_mean:.3f} | Match@{target_len} {match_mean:.3f}±{match_std:.3f} | "
            f"Prefix@{target_len} {prefix_mean:.3f}±{prefix_std:.3f}"
        )
