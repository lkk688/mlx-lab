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
    Load weights saved from your training conversion pipeline into MLX modules.
    Expected key format similar to: blocks.{i}.linear.weight
    """
    weights = np.load(npz_path)
    for k, v in weights.items():
        parts = k.split(".")
        if parts[0] != "blocks":
            continue
        idx = int(parts[1])
        # parts: blocks, {idx}, linear, weight
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
    This avoids O(n) backward scans each step.
    """
    def __init__(self, n: int = 3):
        self.n = n
        self.map = {}  # tuple(tokens[-n:]) -> last position start index

    def build(self, tokens):
        self.map.clear()
        n = self.n
        if len(tokens) < n:
            return
        for i in range(len(tokens) - n):
            key = tuple(tokens[i : i + n])
            self.map[key] = i

    def update_with_new_tokens(self, tokens, new_tokens):
        """
        Update index after appending new tokens to tokens list.
        """
        n = self.n
        # Append already done externally; here we update map for new positions.
        # We only need to add n-grams that end at the new tail.
        # Consider windows that include new tokens: start indices from (len(tokens)-len(new)-n) .. (len(tokens)-n)
        L = len(tokens)
        start_min = max(0, L - len(new_tokens) - n)
        start_max = max(0, L - n)
        for i in range(start_min, start_max + 1):
            if i + n <= L:
                key = tuple(tokens[i : i + n])
                self.map[key] = i

    def try_draft(self, tokens, K: int) -> list | None:
        """
        If last n tokens occurred before, draft the next K tokens from that earlier location.
        """
        n = self.n
        if len(tokens) < n + K:
            return None
        key = tuple(tokens[-n:])
        pos = self.map.get(key, None)
        if pos is None:
            return None

        # Draft tokens after that n-gram occurrence
        start = pos + n
        end = start + K
        if end <= len(tokens):
            draft = tokens[start:end]
            if len(draft) == K:
                return draft
        return None


# ============================================================
# 3) MEDUSA DRAFT (ALIGNED WITH VERIFIER PATH)
# ============================================================
def medusa_draft_tokens(
    model,
    medusa: MLXMedusaHeads,
    hx_last_pre_norm,   # [1,1,H], pre-norm hidden from last layer output
    K: int,
    apply_norm_before_lm_head: bool = True,
):
    """
    Return draft tokens [1,K] from Medusa heads. Optionally apply model.norm() before lm_head().
    """
    feats = medusa(hx_last_pre_norm)  # list of [1,1,H], length K
    toks = []
    for f in feats:
        if apply_norm_before_lm_head:
            logits = model.lm_head(model.model.norm(f))
        else:
            logits = model.lm_head(f)
        toks.append(mx.argmax(logits, axis=-1))  # [1,1]
    return mx.concatenate(toks, axis=1).astype(mx.int32)  # [1,K]


def get_hybrid_draft(
    history_tokens: list,
    ngram: NGramIndex,
    medusa: MLXMedusaHeads,
    model,
    hx_last_pre_norm,
    K: int = 3,
    force_lane: str | None = None,
    apply_norm_before_lm_head: bool = True,
):
    """
    Hybrid proposer:
      1) Try n-gram draft (cheap, longer repeats)
      2) Else Medusa draft (cheap heads)
    """
    if force_lane != "medusa":
        draft = ngram.try_draft(history_tokens, K)
        if draft is not None:
            return mx.array([draft], dtype=mx.int32), "n_gram"

    draft_tensor = medusa_draft_tokens(
        model=model,
        medusa=medusa,
        hx_last_pre_norm=hx_last_pre_norm,
        K=K,
        apply_norm_before_lm_head=apply_norm_before_lm_head,
    )
    return draft_tensor, "medusa"


# ============================================================
# 4) ON-DEVICE PREFIX ACCEPT (NO .tolist(), MIN HOST SYNC)
# ============================================================
def accept_prefix_len(draft_tensor, true_tokens, K: int) -> int:
    """
    draft_tensor: [1,K]
    true_tokens:  [1,K+1] (teacher argmax on verify pass)
    Returns prefix_len in [0..K].
    Only a single scalar host read is performed.
    """
    match = (draft_tensor == true_tokens[:, :K])          # [1,K] bool
    match_i = match.astype(mx.int32)[0]                   # [K]
    prefix = mx.cumprod(match_i, axis=0)                  # [K] stays 1 until first mismatch
    prefix_len = int(mx.sum(prefix).item())               # ONE scalar host read
    return prefix_len


def build_accepted_tensor(draft_tensor, true_tokens, prefix_len: int, K: int):
    """
    Construct accepted tokens:
      - accept draft prefix of length prefix_len
      - if prefix_len < K: accept true token at mismatch position
      - else: accept true token at position K (the next token after the K drafts)
    Returns: accepted [1, <=K+1]
    """
    accepted_prefix = draft_tensor[:, :prefix_len]  # [1,prefix_len]
    if prefix_len < K:
        next_tok = true_tokens[:, prefix_len:prefix_len+1]  # [1,1]
        accepted = mx.concatenate([accepted_prefix, next_tok], axis=1)
    else:
        next_tok = true_tokens[:, K:K+1]  # [1,1]
        accepted = mx.concatenate([accepted_prefix, next_tok], axis=1)
    return accepted.astype(mx.int32)


# ============================================================
# 5) HYBRID GENERATION ENGINE (LOW-OVERHEAD)
# ============================================================
def generate_hybrid_engine_v2(
    model,
    medusa: MLXMedusaHeads,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    K: int = 3,
    force_lane: str | None = None,
    ngram_n: int = 3,
    apply_norm_before_lm_head: bool = True,
):
    """
    Low-overhead hybrid engine:
      - prefill once
      - loop:
          draft (ngram or medusa)
          verify in one base forward call on (1 + K) tokens
          accept prefix length on device
          rollback cache to accepted length
          continue

    Notes:
      - This still runs verification as a separate forward call, but avoids host sync / Python loops.
      - For best speed, keep everything in mx arrays and do minimal conversions.
    """
    prompt_ids = tokenizer.encode(prompt)
    output_tokens = list(prompt_ids)  # Python list to decode at end (batch extend per step)
    ngram = NGramIndex(n=ngram_n)
    ngram.build(output_tokens)

    cache = make_prompt_cache(model)

    start_time = time.time()
    tokens_generated = 0
    forward_calls = 0
    stats = {"n_gram_used": 0, "medusa_used": 0, "draft_tokens_accepted": 0, "verify_tokens": 0}

    # -----------------------
    # PREFILL
    # -----------------------
    x = model.model.embed_tokens(mx.array(prompt_ids, dtype=mx.int32)[None, :])
    for i, layer in enumerate(model.model.layers):
        x = layer(x, mask=None, cache=cache[i])

    # Keep LAST hidden state PRE-NORM for Medusa input
    hx_last = x[:, -1:, :]  # [1,1,H] pre-norm

    # Next token from base model (post-norm -> lm_head)
    logits0 = model.lm_head(model.model.norm(hx_last))
    next_token = mx.argmax(logits0, axis=-1).astype(mx.int32)  # [1,1]
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
        draft_tensor, lane = get_hybrid_draft(
            history_tokens=output_tokens,
            ngram=ngram,
            medusa=medusa,
            model=model,
            hx_last_pre_norm=hx_last,
            K=K,
            force_lane=force_lane,
            apply_norm_before_lm_head=apply_norm_before_lm_head,
        )
        stats[f"{lane}_used"] += 1

        # Verify (next_token + K drafts) in ONE base forward call
        verify_input = mx.concatenate([next_token, draft_tensor], axis=1)  # [1,1+K]
        base_offset = cache[0].offset

        vx = model.model.embed_tokens(verify_input)
        for i, layer in enumerate(model.model.layers):
            vx = layer(vx, mask=None, cache=cache[i])

        # Verifier logits always use post-norm
        slow_logits = model.lm_head(model.model.norm(vx))  # [1,1+K,V]
        true_tokens = mx.argmax(slow_logits, axis=-1).astype(mx.int32)  # [1,1+K]
        mx.eval(true_tokens, draft_tensor)

        forward_calls += 1
        stats["verify_tokens"] += (K + 1)

        # On-device accept prefix length (only 1 scalar host sync)
        p = accept_prefix_len(draft_tensor, true_tokens, K)

        accepted = build_accepted_tensor(draft_tensor, true_tokens, p, K)  # [1,<=K+1]
        mx.eval(accepted)

        # Roll back cache to accepted length
        accepted_len = accepted.shape[1]
        rollback_mlx_cache(cache, base_offset + accepted_len)

        # Append accepted tokens (single host conversion per step)
        accepted_host = accepted[0].tolist()
        output_tokens.extend(accepted_host)
        ngram.update_with_new_tokens(output_tokens, accepted_host)

        tokens_generated += accepted_len
        stats["draft_tokens_accepted"] += max(0, accepted_len - 1)

        # Update next_token (last accepted)
        last_tok = accepted_host[-1]
        next_token = mx.array([[last_tok]], dtype=mx.int32)

        # Update hx_last to last accepted position PRE-NORM:
        # vx corresponds to (next_token + draft) positions; accepted_len-1 is last accepted index in this verify block
        last_idx = accepted_len - 1
        hx_last = vx[:, last_idx:last_idx+1, :]

        if last_tok == tokenizer.eos_token_id:
            break

    wall_time = time.time() - start_time
    gen_tokens = output_tokens[len(prompt_ids):]
    final_text = tokenizer.decode(gen_tokens)
    return final_text, gen_tokens, tokens_generated, wall_time, forward_calls, stats


# ============================================================
# 6) Baselines (unchanged)
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


# ============================================================
# 7) Small helper to sample prompts (unchanged)
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


# ============================================================
# 8) Example usage
# ============================================================
if __name__ == "__main__":
    MODEL_ID = "mlx-community/Qwen2.5-Coder-7B-4bit"
    DRAFT_ID = "mlx-community/Qwen2.5-Coder-0.5B-4bit"

    # IMPORTANT:
    # If you trained Medusa to project logits as lm_head(norm(feat)), set APPLY_NORM_BEFORE_LM_HEAD=True.
    # If you trained Medusa to project logits as lm_head(feat) (no norm), set it False.
    APPLY_NORM_BEFORE_LM_HEAD = True

    print("Loading base model...")
    model, tokenizer = load(MODEL_ID)

    print("Loading Medusa heads...")
    hidden_size = model.model.embed_tokens.weight.shape[1]
    medusa = MLXMedusaHeads(hidden_size, num_heads=3)
    load_medusa_weights(medusa, "mlx_medusa_heads.npz")

    print("Loading draft model (for official speculative baseline)...")
    draft_model, _ = load(DRAFT_ID)

    dataset_names = [
        "bigcode/the-stack-smol",
        "codeparrot/github-code",
        "bigcode/starcoderdata",
    ]
    split = "train"
    num_samples = 10
    prompt_len = 64
    target_len = 64
    seed = 42

    samples, dataset_name = sample_dataset_prompts(
        tokenizer, dataset_names, split, num_samples, prompt_len, target_len, seed
    )

    print(f"\nDataset: {dataset_name} | Samples: {len(samples)}")

    for idx, (prompt, _) in enumerate(samples):
        print(f"\n[Sample {idx+1}]")

        # Baseline
        base_tokens, base_time = run_base_stream(model, tokenizer, prompt, max_new_tokens=target_len)
        base_tps = len(base_tokens) / base_time

        # Hybrid v2
        _, gen_tokens, _, t, forward_calls, stats = generate_hybrid_engine_v2(
            model=model,
            medusa=medusa,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=target_len,
            K=3,
            force_lane=None,
            ngram_n=3,
            apply_norm_before_lm_head=APPLY_NORM_BEFORE_LM_HEAD,
        )
        hybrid_tps = len(gen_tokens[:target_len]) / t

        print(f"Baseline: {base_tps:.2f} tok/s | HybridV2: {hybrid_tps:.2f} tok/s | "
              f"Speedup: {hybrid_tps/base_tps:.2f}x | forward_calls={forward_calls} | stats={stats}")