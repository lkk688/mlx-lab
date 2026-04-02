#!/usr/bin/env python3
# mlx_hybrid_medusa_v4_dynamic_eval.py
# ============================================================
# MLX Hybrid Inference (Medusa + N-gram) + Evaluation (Speed + Correctness)
#
# Improvements over v3:
#   - Use mx.topk(k=2) instead of argsort for margin (much faster)
#   - Cost-aware dynamic m selection (expected benefit heuristic)
#   - Optional dynamic m for n-gram lane
#   - Extra diagnostics: acceptance histogram + verifier efficiency
#   - CLI args for all inputs/outputs and knobs
# ============================================================

import argparse
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

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
    weights = np.load(npz_path)
    for k, v in weights.items():
        parts = k.split(".")
        if parts[0] != "blocks":
            continue
        idx = int(parts[1])
        target = getattr(module.blocks[idx], parts[2])  # linear
        setattr(target, parts[3], mx.array(v))          # weight


def rollback_mlx_cache(cache, keep_len: int):
    for c in cache:
        c.offset = keep_len


# ============================================================
# 2) FAST N-GRAM INDEX (INCREMENTAL)
# ============================================================
class NGramIndex:
    def __init__(self, n: int = 3):
        self.n = n
        self.map = {}  # tuple -> last start

    def build(self, tokens: List[int]):
        self.map.clear()
        n = self.n
        if len(tokens) < n:
            return
        for i in range(len(tokens) - n):
            self.map[tuple(tokens[i:i+n])] = i

    def update_with_new_tokens(self, tokens: List[int], new_tokens: List[int]):
        n = self.n
        L = len(tokens)
        start_min = max(0, L - len(new_tokens) - n)
        start_max = max(0, L - n)
        for i in range(start_min, start_max + 1):
            if i + n <= L:
                self.map[tuple(tokens[i:i+n])] = i

    def try_draft(self, tokens: List[int], K: int) -> Optional[List[int]]:
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
# 3) MEDUSA DRAFT + CONFIDENCE
# ============================================================
def _logits_from_feat(model, feat, apply_norm_before_lm_head: bool):
    if apply_norm_before_lm_head:
        return model.lm_head(model.model.norm(feat))
    return model.lm_head(feat)


def medusa_draft_tokens_and_conf(
    model,
    medusa: MLXMedusaHeads,
    hx_last_pre_norm,    # [1,1,H]
    K: int,
    apply_norm_before_lm_head: bool,
):
    """
    Returns:
      draft_tokens: [1,K] int32
      conf: margins [1,K], entropies [1,K]
    Uses mx.topk(k=2) for speed.
    """
    feats = medusa(hx_last_pre_norm)
    toks, margins, entropies = [], [], []

    for f in feats[:K]:
        logits = _logits_from_feat(model, f, apply_norm_before_lm_head)  # [1,1,V]

        idx_sorted = mx.argsort(logits, axis=-1)                         # [1,1,V]
        top2_idx = idx_sorted[..., -2:]                                  # [1,1,2]
        top1_idx = top2_idx[..., -1:]                                    # [1,1,1]
        top2_idx_only = top2_idx[..., -2:-1]                             # [1,1,1]
        top1_val = mx.take_along_axis(logits, top1_idx, axis=-1)         # [1,1,1]
        top2_val = mx.take_along_axis(logits, top2_idx_only, axis=-1)    # [1,1,1]
        tok = top1_idx[..., 0]                                           # [1,1]

        # logit margin
        m = top1_val[..., 0] - top2_val[..., 0]                          # [1,1]

        # entropy
        p = mx.softmax(logits, axis=-1)
        ent = -mx.sum(p * mx.log(p + 1e-9), axis=-1)                     # [1,1]

        toks.append(tok)
        margins.append(m)
        entropies.append(ent)

    draft_tokens = mx.concatenate(toks, axis=1).astype(mx.int32)         # [1,K]
    margins = mx.concatenate(margins, axis=1)                            # [1,K]
    entropies = mx.concatenate(entropies, axis=1)                        # [1,K]
    return draft_tokens, {"margins": margins, "entropies": entropies}


# ============================================================
# 4) Dynamic m selection
# ============================================================
def choose_dynamic_m_threshold(
    margins: mx.array, entropies: mx.array, K: int,
    margin_thresh: float, entropy_thresh: float, min_m: int
) -> int:
    ok = (margins >= margin_thresh) & (entropies <= entropy_thresh)  # [1,K]
    ok_i = ok.astype(mx.int32)[0]                                    # [K]
    prefix = mx.cumprod(ok_i, axis=0)
    m = int(mx.sum(prefix).item())  # one scalar host read
    m = max(min_m, min(K, m))
    return m


def choose_dynamic_m_cost_aware(
    margins: mx.array, entropies: mx.array, K: int,
    base_verify_cost: float,
    per_token_verify_cost: float,
    benefit_per_accepted_token: float,
    margin_scale: float,
    entropy_scale: float,
    min_m: int,
) -> int:
    """
    Cost-aware heuristic: choose m that maximizes expected net benefit.
    We approximate acceptance probability from margin and entropy.

    p_accept_k ~ sigmoid(margin/margin_scale) * sigmoid((entropy_scale - ent)/entropy_scale)
    expected accepted draft tokens ~ sum_{i=1..m} Π_{j<=i} p_accept_j

    net(m) = benefit_per_accepted_token * E[accepted_draft_tokens]
             - (base_verify_cost + per_token_verify_cost*(m+1))

    Returns m with ONE scalar host read (we compute scores on device, then argmax).
    """
    # Normalize signals
    # sigmoid for margin: higher margin => higher accept prob
    p_m = 1 / (1 + mx.exp(-(margins / max(1e-6, margin_scale))))  # [1,K]
    # sigmoid for entropy: lower entropy => higher accept prob
    p_e = 1 / (1 + mx.exp((entropies - entropy_scale) / max(1e-6, entropy_scale)))  # [1,K]
    p = (p_m * p_e)[0]  # [K]

    # Prefix product: prob all first i tokens accepted
    pref = mx.cumprod(p, axis=0)  # [K]
    # Expected accepted draft tokens for m: sum_{i=1..m} pref[i-1]
    # Build cumulative sum
    csum = mx.cumsum(pref, axis=0)  # [K]
    # For each m, expected accepted draft tokens = csum[m-1]
    # Compute net score for each m=1..K
    ms = mx.arange(1, K+1)
    exp_acc = csum  # index 0..K-1 corresponds to m=1..K
    verify_cost = base_verify_cost + per_token_verify_cost * (ms + 1)  # (m + 1) tokens verified
    net = benefit_per_accepted_token * exp_acc - verify_cost

    # Enforce min_m: set net below min_m to very negative
    if min_m > 1:
        mask = ms < min_m
        net = mx.where(mask, mx.array(-1e9), net)

    # Argmax on device; then one scalar host read
    best_idx = int(mx.argmax(net).item())  # 0..K-1
    m = best_idx + 1
    return m


# ============================================================
# 5) Acceptance (on-device)
# ============================================================
def accept_prefix_len(draft_tensor, true_tokens, m: int) -> int:
    match = (draft_tensor == true_tokens[:, :m])          # [1,m]
    match_i = match.astype(mx.int32)[0]                   # [m]
    prefix = mx.cumprod(match_i, axis=0)
    return int(mx.sum(prefix).item())                     # one scalar host read


def build_accepted_tensor(draft_tensor, true_tokens, prefix_len: int, m: int):
    accepted_prefix = draft_tensor[:, :prefix_len]
    if prefix_len < m:
        next_tok = true_tokens[:, prefix_len:prefix_len+1]
        accepted = mx.concatenate([accepted_prefix, next_tok], axis=1)
    else:
        next_tok = true_tokens[:, m:m+1]
        accepted = mx.concatenate([accepted_prefix, next_tok], axis=1)
    return accepted.astype(mx.int32)


# ============================================================
# 6) Engine
# ============================================================
def generate_hybrid_engine(
    model,
    medusa: MLXMedusaHeads,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    K: int,
    ngram_n: int,
    apply_norm_before_lm_head: bool,
    force_lane: Optional[str],
    dynamic_mode: str,
    # Threshold policy
    margin_thresh: float,
    entropy_thresh: float,
    min_m: int,
    # Cost-aware policy params
    base_verify_cost: float,
    per_token_verify_cost: float,
    benefit_per_accepted_token: float,
    margin_scale: float,
    entropy_scale: float,
    ngram_dynamic_m: bool,
):
    prompt_ids = tokenizer.encode(prompt)
    output_tokens = list(prompt_ids)

    ngram = NGramIndex(n=ngram_n)
    ngram.build(output_tokens)

    cache = make_prompt_cache(model)

    start_time = time.time()
    tokens_generated = 0
    forward_calls = 0

    stats = {
        "n_gram_used": 0,
        "medusa_used": 0,
        "verify_tokens": 0,
        "draft_tokens_accepted": 0,
        "accept_len_hist": {i: 0 for i in range(1, K+2)},  # accepted_len in [1..K+1]
        "m_hist": {i: 0 for i in range(1, K+1)},
        "avg_m": 0.0,
    }

    # ---- PREFILL ----
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

    # ---- LOOP ----
    while tokens_generated < max_new_tokens:
        lane = "medusa"
        draft_tensor = None
        m = K

        # N-gram proposal
        if force_lane != "medusa":
            draft_list = ngram.try_draft(output_tokens, K)
            if draft_list is not None:
                lane = "n_gram"
                stats["n_gram_used"] += 1
                draft_tensor_full = mx.array([draft_list], dtype=mx.int32)

                # Optional: dynamic m even for n-gram lane (can reduce wasted verify work)
                if ngram_dynamic_m and dynamic_mode != "off":
                    # treat n-gram as high confidence: set m=K
                    m = K
                else:
                    m = K

                draft_tensor = draft_tensor_full[:, :m]
        # Medusa proposal
        if draft_tensor is None:
            stats["medusa_used"] += 1
            draft_full, conf = medusa_draft_tokens_and_conf(
                model=model,
                medusa=medusa,
                hx_last_pre_norm=hx_last,
                K=K,
                apply_norm_before_lm_head=apply_norm_before_lm_head,
            )

            if dynamic_mode == "off":
                m = K
            elif dynamic_mode == "threshold":
                m = choose_dynamic_m_threshold(
                    conf["margins"], conf["entropies"], K,
                    margin_thresh=margin_thresh, entropy_thresh=entropy_thresh, min_m=min_m
                )
            elif dynamic_mode == "cost":
                m = choose_dynamic_m_cost_aware(
                    conf["margins"], conf["entropies"], K,
                    base_verify_cost=base_verify_cost,
                    per_token_verify_cost=per_token_verify_cost,
                    benefit_per_accepted_token=benefit_per_accepted_token,
                    margin_scale=margin_scale,
                    entropy_scale=entropy_scale,
                    min_m=min_m,
                )
            else:
                raise ValueError(f"Unknown dynamic_mode: {dynamic_mode}")

            draft_tensor = draft_full[:, :m].astype(mx.int32)

        stats["m_hist"][m] += 1

        # ---- VERIFY only (1+m) ----
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

        # ---- ACCEPT ----
        p = accept_prefix_len(draft_tensor, true_tokens, m)
        accepted = build_accepted_tensor(draft_tensor, true_tokens, p, m)
        mx.eval(accepted)

        accepted_len = int(accepted.shape[1])
        stats["accept_len_hist"][accepted_len] += 1
        rollback_mlx_cache(cache, base_offset + accepted_len)

        accepted_host = accepted[0].tolist()  # one host conversion per step
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

    total_steps = sum(stats["m_hist"].values())
    if total_steps > 0:
        stats["avg_m"] = sum(k * v for k, v in stats["m_hist"].items()) / total_steps

    # Extra efficiency stats
    stats["verify_tokens_per_generated"] = stats["verify_tokens"] / max(1, tokens_generated)
    stats["accepted_draft_per_verify_token"] = stats["draft_tokens_accepted"] / max(1, stats["verify_tokens"])

    return final_text, gen_tokens, tokens_generated, wall_time, forward_calls, stats


# ============================================================
# 7) Baselines
# ============================================================
def run_stream_tokens(model, tokenizer, prompt, max_new_tokens, sampler, num_draft_tokens=None, draft_model=None):
    tokens = []
    start_time = time.time()
    for resp in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_new_tokens,
        draft_model=draft_model, num_draft_tokens=num_draft_tokens, sampler=sampler
    ):
        tokens.append(resp.token)
    return tokens, time.time() - start_time


def run_base_stream(model, tokenizer, prompt, max_new_tokens):
    sampler = make_sampler(temp=0.0)
    return run_stream_tokens(model, tokenizer, prompt, max_new_tokens, sampler)


def run_lookahead_mlx(model, tokenizer, prompt, max_new_tokens, num_draft_tokens):
    sampler = make_sampler(temp=0.0)
    return run_stream_tokens(model, tokenizer, prompt, max_new_tokens, sampler, num_draft_tokens=num_draft_tokens)


def run_two_model_speculative(model, draft_model, tokenizer, prompt, max_new_tokens, num_draft_tokens):
    sampler = make_sampler(temp=0.0)
    return run_stream_tokens(
        model, tokenizer, prompt, max_new_tokens, sampler,
        num_draft_tokens=num_draft_tokens, draft_model=draft_model
    )


# ============================================================
# 8) Correctness metrics
# ============================================================
def token_edit_distance(a, b):
    n = len(a); m = len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
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
    return {m: {"tps": [], "speedup": [], "token_acc": [], "exact": [], "baseline_match": [], "prefix_hit": [], "edit_sim": []}
            for m in methods}


# ============================================================
# 9) Dataset prompt sampling
# ============================================================
def select_text_field(sample):
    for key in ["content", "text", "code"]:
        if key in sample and sample[key]:
            return sample[key]
    return None


def sample_dataset_prompts(tokenizer, dataset_names, split, num_samples, prompt_len, target_len, seed):
    for name in dataset_names:
        try:
            ds = load_dataset(name, split=split, streaming=True).shuffle(seed=seed, buffer_size=1000)
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
                samples.append((tokenizer.decode(prompt_tokens), ref_tokens))
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                return samples, name
        except Exception:
            continue
    raise RuntimeError("Failed to load dataset samples")


# ============================================================
# 10) Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="mlx-community/Qwen2.5-Coder-7B-4bit")
    p.add_argument("--draft-model", default="mlx-community/Qwen2.5-Coder-0.5B-4bit")
    p.add_argument("--medusa-npz", default="mlx_medusa_heads.npz")

    p.add_argument("--apply-norm-before-lm-head", action="store_true", default=True)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--ngram-n", type=int, default=3)
    p.add_argument("--force-lane", choices=["medusa", "ngram", "none"], default="none")

    p.add_argument("--dynamic-mode", choices=["off", "threshold", "cost"], default="cost")
    p.add_argument("--margin-thresh", type=float, default=2.0)
    p.add_argument("--entropy-thresh", type=float, default=4.0)
    p.add_argument("--min-m", type=int, default=1)

    # cost-aware knobs (heuristic, tune per machine)
    p.add_argument("--base-verify-cost", type=float, default=1.0)
    p.add_argument("--per-token-verify-cost", type=float, default=0.25)
    p.add_argument("--benefit-per-accepted-token", type=float, default=1.0)
    p.add_argument("--margin-scale", type=float, default=2.0)
    p.add_argument("--entropy-scale", type=float, default=4.0)

    p.add_argument("--ngram-dynamic-m", action="store_true", default=False)

    # Eval
    p.add_argument("--datasets", nargs="+", default=["bigcode/the-stack-smol", "codeparrot/github-code", "bigcode/starcoderdata"])
    p.add_argument("--split", default="train")
    p.add_argument("--num-samples", type=int, default=25)
    p.add_argument("--prompt-len", type=int, default=64)
    p.add_argument("--target-len", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-draft-tokens", type=int, default=4)
    p.add_argument("--out-json", default="mlx_eval_results.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading base model...")
    model, tokenizer = load(args.base_model)

    print("Loading Medusa heads...")
    hidden_size = model.model.embed_tokens.weight.shape[1]
    medusa = MLXMedusaHeads(hidden_size, num_heads=args.K)
    load_medusa_weights(medusa, args.medusa_npz)

    print("Loading draft model for baseline speculative...")
    draft_model, _ = load(args.draft_model)

    samples, dataset_name = sample_dataset_prompts(
        tokenizer, args.datasets, args.split, args.num_samples,
        args.prompt_len, args.target_len, args.seed
    )

    methods = ["Baseline", "HybridV4", "HybridV4 MedusaOnly", "MLX Lookahead", "Two-Model Speculative"]
    metrics = init_metric_store(methods)
    diag_store = {"HybridV4": [], "HybridV4 MedusaOnly": []}

    print("\n" + "=" * 70)
    print("🧪 DATASET EVALUATION (Speed + Correctness)")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {len(samples)} | prompt_len={args.prompt_len} | target_len={args.target_len}")
    print(f"HybridV4: K={args.K}, dynamic={args.dynamic_mode}, force_lane={args.force_lane}")

    for idx, (prompt, ref_tokens) in enumerate(samples):
        preview = prompt.replace("\n", " ").replace("\r", " ").replace("\t", " ")[:80]
        print(f"\n[Sample {idx+1}] {preview}")

        # Baseline
        base_tokens, base_time = run_base_stream(model, tokenizer, prompt, args.target_len)
        base_eval = base_tokens[:args.target_len]
        base_tps = len(base_eval) / max(base_time, 1e-9)
        base_acc, base_exact, _, base_prefix, base_edit = score_tokens(base_eval, ref_tokens)

        metrics["Baseline"]["tps"].append(base_tps)
        metrics["Baseline"]["speedup"].append(1.0)
        metrics["Baseline"]["token_acc"].append(base_acc)
        metrics["Baseline"]["exact"].append(base_exact)
        metrics["Baseline"]["baseline_match"].append(1.0)
        metrics["Baseline"]["prefix_hit"].append(base_prefix)
        metrics["Baseline"]["edit_sim"].append(base_edit)

        # HybridV4
        lane = None if args.force_lane == "none" else ("medusa" if args.force_lane == "medusa" else "ngram")
        _, hyb_tokens, _, hyb_time, _, hyb_stats = generate_hybrid_engine(
            model=model,
            medusa=medusa,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.target_len,
            K=args.K,
            ngram_n=args.ngram_n,
            apply_norm_before_lm_head=args.apply_norm_before_lm_head,
            force_lane=lane,
            dynamic_mode=args.dynamic_mode,
            margin_thresh=args.margin_thresh,
            entropy_thresh=args.entropy_thresh,
            min_m=args.min_m,
            base_verify_cost=args.base_verify_cost,
            per_token_verify_cost=args.per_token_verify_cost,
            benefit_per_accepted_token=args.benefit_per_accepted_token,
            margin_scale=args.margin_scale,
            entropy_scale=args.entropy_scale,
            ngram_dynamic_m=args.ngram_dynamic_m,
        )
        hyb_eval = hyb_tokens[:args.target_len]
        hyb_tps = len(hyb_eval) / max(hyb_time, 1e-9)
        hyb_speed = hyb_tps / max(base_tps, 1e-9)
        hyb_acc, hyb_exact, hyb_match, hyb_prefix, hyb_edit = score_tokens(hyb_eval, ref_tokens, baseline_tokens=base_eval)

        metrics["HybridV4"]["tps"].append(hyb_tps)
        metrics["HybridV4"]["speedup"].append(hyb_speed)
        metrics["HybridV4"]["token_acc"].append(hyb_acc)
        metrics["HybridV4"]["exact"].append(hyb_exact)
        metrics["HybridV4"]["baseline_match"].append(hyb_match)
        metrics["HybridV4"]["prefix_hit"].append(hyb_prefix)
        metrics["HybridV4"]["edit_sim"].append(hyb_edit)
        diag_store["HybridV4"].append(hyb_stats)

        # HybridV4 MedusaOnly
        _, med_tokens, _, med_time, _, med_stats = generate_hybrid_engine(
            model=model,
            medusa=medusa,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.target_len,
            K=args.K,
            ngram_n=args.ngram_n,
            apply_norm_before_lm_head=args.apply_norm_before_lm_head,
            force_lane="medusa",
            dynamic_mode=args.dynamic_mode,
            margin_thresh=args.margin_thresh,
            entropy_thresh=args.entropy_thresh,
            min_m=args.min_m,
            base_verify_cost=args.base_verify_cost,
            per_token_verify_cost=args.per_token_verify_cost,
            benefit_per_accepted_token=args.benefit_per_accepted_token,
            margin_scale=args.margin_scale,
            entropy_scale=args.entropy_scale,
            ngram_dynamic_m=args.ngram_dynamic_m,
        )
        med_eval = med_tokens[:args.target_len]
        med_tps = len(med_eval) / max(med_time, 1e-9)
        med_speed = med_tps / max(base_tps, 1e-9)
        med_acc, med_exact, med_match, med_prefix, med_edit = score_tokens(med_eval, ref_tokens, baseline_tokens=base_eval)

        metrics["HybridV4 MedusaOnly"]["tps"].append(med_tps)
        metrics["HybridV4 MedusaOnly"]["speedup"].append(med_speed)
        metrics["HybridV4 MedusaOnly"]["token_acc"].append(med_acc)
        metrics["HybridV4 MedusaOnly"]["exact"].append(med_exact)
        metrics["HybridV4 MedusaOnly"]["baseline_match"].append(med_match)
        metrics["HybridV4 MedusaOnly"]["prefix_hit"].append(med_prefix)
        metrics["HybridV4 MedusaOnly"]["edit_sim"].append(med_edit)
        diag_store["HybridV4 MedusaOnly"].append(med_stats)

        # Lookahead baseline
        look_tokens, look_time = run_lookahead_mlx(model, tokenizer, prompt, args.target_len, args.num_draft_tokens)
        look_eval = look_tokens[:args.target_len]
        look_tps = len(look_eval) / max(look_time, 1e-9)
        look_speed = look_tps / max(base_tps, 1e-9)
        look_acc, look_exact, look_match, look_prefix, look_edit = score_tokens(look_eval, ref_tokens, baseline_tokens=base_eval)

        metrics["MLX Lookahead"]["tps"].append(look_tps)
        metrics["MLX Lookahead"]["speedup"].append(look_speed)
        metrics["MLX Lookahead"]["token_acc"].append(look_acc)
        metrics["MLX Lookahead"]["exact"].append(look_exact)
        metrics["MLX Lookahead"]["baseline_match"].append(look_match)
        metrics["MLX Lookahead"]["prefix_hit"].append(look_prefix)
        metrics["MLX Lookahead"]["edit_sim"].append(look_edit)

        # Two-model speculative baseline
        spec_tokens, spec_time = run_two_model_speculative(model, draft_model, tokenizer, prompt, args.target_len, args.num_draft_tokens)
        spec_eval = spec_tokens[:args.target_len]
        spec_tps = len(spec_eval) / max(spec_time, 1e-9)
        spec_speed = spec_tps / max(base_tps, 1e-9)
        spec_acc, spec_exact, spec_match, spec_prefix, spec_edit = score_tokens(spec_eval, ref_tokens, baseline_tokens=base_eval)

        metrics["Two-Model Speculative"]["tps"].append(spec_tps)
        metrics["Two-Model Speculative"]["speedup"].append(spec_speed)
        metrics["Two-Model Speculative"]["token_acc"].append(spec_acc)
        metrics["Two-Model Speculative"]["exact"].append(spec_exact)
        metrics["Two-Model Speculative"]["baseline_match"].append(spec_match)
        metrics["Two-Model Speculative"]["prefix_hit"].append(spec_prefix)
        metrics["Two-Model Speculative"]["edit_sim"].append(spec_edit)

        print(f"Baseline {base_tps:.2f} t/s | HybridV4 {hyb_speed:.2f}x | MedusaOnly {med_speed:.2f}x | Lookahead {look_speed:.2f}x | TwoModel {spec_speed:.2f}x")
        print(f"  HybridV4 diag: avg_m={hyb_stats['avg_m']:.2f}, verify_tokens/gen={hyb_stats['verify_tokens_per_generated']:.3f}, "
              f"accepted_draft/verify={hyb_stats['accepted_draft_per_verify_token']:.3f}, lane(n={hyb_stats['n_gram_used']}, m={hyb_stats['medusa_used']})")

    print("\n" + "=" * 70)
    print("📊 AVERAGED RESULTS")
    print("=" * 70)

    summary = {"dataset": dataset_name, "methods": {}, "args": vars(args)}
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

        summary["methods"][method] = {
            "tps_mean": tps_mean, "tps_std": tps_std,
            "speedup_mean": speed_mean, "speedup_std": speed_std,
            "token_acc_mean": acc_mean, "exact_mean": exact_mean,
            "edit_sim_mean": edit_mean,
            "baseline_match_mean": match_mean, "baseline_match_std": match_std,
            "prefix_hit_mean": prefix_mean, "prefix_hit_std": prefix_std,
        }

        print(
            f"{method}: {tps_mean:.2f}±{tps_std:.2f} tok/s | "
            f"Speedup {speed_mean:.2f}±{speed_std:.2f}x ({improvement:+.1f}%) | "
            f"TokenAcc {acc_mean:.3f} | EditSim@{args.target_len} {edit_mean:.3f} | "
            f"Exact {exact_mean:.3f} | Match@{args.target_len} {match_mean:.3f}±{match_std:.3f} | "
            f"Prefix@{args.target_len} {prefix_mean:.3f}±{prefix_std:.3f}"
        )

    # Dump diagnostics for HybridV4
    summary["diagnostics"] = {
        "HybridV4": diag_store["HybridV4"],
        "HybridV4 MedusaOnly": diag_store["HybridV4 MedusaOnly"],
    }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Saved evaluation summary to: {args.out_json}")

"""
#Use cost-aware dynamic m
python mlx_hybrid_medusa_v4_dynamic_eval.py \
  --base-model mlx-community/Qwen2.5-Coder-7B-4bit \
  --draft-model mlx-community/Qwen2.5-Coder-0.5B-4bit \
  --medusa-npz mlx_medusa_heads.npz \
  --K 3 \
  --dynamic-mode cost \
  --out-json results_cost.json

#Threshold-based dynamic m (closest to previous version):
python mlx_hybrid_medusa_v4_dynamic_eval.py \
  --medusa-npz mlx_medusa_heads.npz \
  --dynamic-mode threshold \
  --margin-thresh 2.0 \
  --entropy-thresh 4.0 \
  --out-json results_thresh.json

#Disable dynamic (fixed K):
python mlx_hybrid_medusa_v4_dynamic_eval.py \
  --medusa-npz mlx_medusa_heads.npz \
  --dynamic-mode off \
  --out-json results_fixed.json
"""
