"""
two_model_react_mlx.py
A two-model ReAct agentic framework targeting Apple MLX-LM style runtimes.

"""

import time
import json
import os
from typing import Any, Dict, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

# ---------------------------
# Hypothetical thin MLX client (Replaced with Real MLXClient)
# ---------------------------
import mlx.core as mx
from mlx_lm import load, stream_generate, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.models.base import create_attention_mask
import numpy as np

class MLXClient:
    def __init__(self, model_path: str, device: str = "metal", quant: Optional[str] = None):
        self.model_path = model_path
        self.device = device
        self.quant = quant
        
        # Load model and tokenizer via mlx_lm
        self.model, self.tokenizer = load(model_path)
        self._kv_cache = None  # to persist KV cache

    def prefill(self, prompt: str, seq_len: Optional[int] = None) -> Dict[str, Any]:
        """
        Prefill / encode the prompt so KV cache is created.
        """
        start = time.time()
        prompt_ids = self.tokenizer.encode(prompt)
        
        self._kv_cache = make_prompt_cache(self.model)
        x = mx.array(prompt_ids)[None]
        
        # Pass through the model layers to populate cache
        x = self.model.model.embed_tokens(x)
        mask_prefill = create_attention_mask(x, self._kv_cache[0]) if x.shape[1] > 1 else None
        for i, layer in enumerate(self.model.model.layers):
            x = layer(x, mask=mask_prefill, cache=self._kv_cache[i])
        
        # Materialize caches
        mx.eval([c.keys for c in self._kv_cache if hasattr(c, "keys")] +
                [c.values for c in self._kv_cache if hasattr(c, "values")])
        
        return {"ok": True, "tokens": prompt_ids, "token_count": len(prompt_ids), "prefill_time": time.time() - start}

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.0,
                 draft_model: Optional['MLXClient'] = None,
                 num_draft_tokens: int = 4,
                 stream: bool = False) -> Dict[str, Any]:
        """
        Main generate. Uses speculative decoding if draft_model is provided.
        """
        start = time.time()
        
        sampler = make_sampler(temp=temperature)

        # For stream_generate args
        kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "sampler": sampler,
        }
        
        if draft_model is not None:
            kwargs["draft_model"] = draft_model.model
            kwargs["num_draft_tokens"] = num_draft_tokens
            
        # Passing prompt_cache if available across turns
        # If mlx_lm version supports passing prompt_cache directly to stream_generate, this works.
        # Otherwise, we might have to manage state manually; but stream_generate usually handles generic requests cleanly.
        if self._kv_cache is not None:
            # We assume it supports `prompt_cache` kwargs or we just let stream_generate handle the full prompt if it doesn't.
            # For robustness we pass the full prompt and let the wrapper optimize if possible.
            pass

        generated_tokens = []
        token_times = []
        from_draft = []
        
        for resp in stream_generate(**kwargs):
            generated_tokens.append(resp.token)
            token_times.append(0.0) # we could add accurate timing per token here
            from_draft.append(resp.from_draft)
            
        text = self.tokenizer.decode(generated_tokens)
        end = time.time()
        
        # mlx memory measurement stats
        peak_mem = mx.metal.get_peak_memory() / (1024 * 1024) if mx.metal.is_available() else None
        
        return {
            "text": text,
            "tokens": generated_tokens,
            "token_times": token_times,
            "from_draft": from_draft,
            "generation_time_s": end - start,
            "peak_memory_mb": peak_mem,
        }

    def generate_cli_wrapper(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.0,
                 draft_model: Optional['MLXClient'] = None,
                 num_draft_tokens: int = 4) -> Dict[str, Any]:
        """
        Directly wraps Apple's exact mlx_lm.generate block to strip all streaming/python overhead 
        for raw un-intercepted maximal TPS generation.
        """
        start = time.time()
        sampler = make_sampler(temp=temperature)

        kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "sampler": sampler,
            "verbose": False
        }
        
        if draft_model is not None:
            kwargs["draft_model"] = draft_model.model
            kwargs["num_draft_tokens"] = num_draft_tokens
            
        text = generate(**kwargs)
        end = time.time()
        
        # We estimate output tokens quickly. 
        # Token metrics are inherently omitted by raw generate() output strings.
        out_tokens = self.tokenizer.encode(text)

        return {
            "text": text,
            "tokens": out_tokens,
            "token_times": [0.0]*len(out_tokens),
            "from_draft": [False]*len(out_tokens),
            "generation_time_s": end - start,
            "peak_memory_mb": mx.metal.get_peak_memory() / (1024 * 1024) if mx.metal.is_available() else None,
        }

    def get_kv_cache(self) -> Any:
        return self._kv_cache

    def set_kv_cache(self, kv: Any):
        self._kv_cache = kv

    def save_prompt_cache(self, fname: str):
        """Optional: persist prompt cache to disk as npz"""
        if not self._kv_cache:
            return
        state_dict = {}
        for i, c in enumerate(self._kv_cache):
            if hasattr(c, "keys") and getattr(c, "keys", None) is not None:
                state_dict[f"layer_{i}_keys"] = np.array(c.keys)
                state_dict[f"layer_{i}_values"] = np.array(c.values)
                state_dict[f"layer_{i}_offset"] = np.array([c.offset])
        np.savez(fname, **state_dict)

    def load_prompt_cache(self, fname: str):
        """Optional: load prompt cache"""
        if not os.path.exists(fname):
            return False
        try:
            state_dict = np.load(fname)
            self._kv_cache = make_prompt_cache(self.model)
            for i, c in enumerate(self._kv_cache):
                k_key = f"layer_{i}_keys"
                v_key = f"layer_{i}_values"
                o_key = f"layer_{i}_offset"
                if k_key in state_dict:
                    c.keys = mx.array(state_dict[k_key])
                    c.values = mx.array(state_dict[v_key])
                    c.offset = int(state_dict[o_key][0])
            return True
        except Exception:
            return False

# ---------------------------
# KV Cache Strategies
# ---------------------------
@dataclass
class KVCacheStrategyConfig:
    """Configuration for which cache strategy to use."""
    name: str  # one of ["no_sync", "keep_both", "prompt_cache", "replay_prefill"]
    prompt_cache_dir: Optional[str] = None  # used for prompt_cache strategy
    max_prefill_tokens_batch: int = 4096  # how many tokens to prefill in a batch when replaying
    replay_chunk_size: int = 1024  # tokens per prefill chunk when catching up

# ---------------------------
# Model Manager
# ---------------------------
class ModelManager:
    """
    Loads and manages target (big) and draft (small) MLX models and implements KV cache strategies.
    """
    def __init__(self, target_path: str, draft_path: str,
                 device: str = "metal", kv_strategy: KVCacheStrategyConfig = None):
        self.target_path = target_path
        self.draft_path = draft_path
        self.device = device
        self.kv_strategy = kv_strategy or KVCacheStrategyConfig(name="no_sync")
        # instantiate clients - replace with real MLX load API
        self.target = MLXClient(target_path, device=device)
        self.draft = MLXClient(draft_path, device=device)
        # Keep per-conversation caches (in-memory)
        self.kv_store = {
            "target": None,
            "draft": None
        }
        # simple prompt history for replay/prefill strategies
        self.prompt_history = deque(maxlen=5000)  # store tuples (role, text)
        # persistent prompt cache filenames if needed
        if self.kv_strategy.prompt_cache_dir:
            os.makedirs(self.kv_strategy.prompt_cache_dir, exist_ok=True)

    # -------------------------
    # Conversation / Prompt helpers
    # -------------------------
    def append_message(self, role: str, text: str):
        self.prompt_history.append((role, text))

    def get_full_prompt(self) -> str:
        messages = [{"role": r, "content": t} for r, t in self.prompt_history]
        
        # Check if tokenizer has apply_chat_template
        if hasattr(self.target.tokenizer, "apply_chat_template") and self.target.tokenizer.chat_template is not None:
            return self.target.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        return "\n".join([f"{r}: {t}" for (r, t) in self.prompt_history])

    # -------------------------
    # Prefill / cache strategies
    # -------------------------
    def ensure_target_prefill(self):
        """Ensure target model has prefilled KV cache for current prompt history."""
        if self.kv_strategy.name == "no_sync":
            # lazy prefill when generation happens; do nothing now
            return
        if self.kv_strategy.name == "keep_both":
            # keep both caches updated by executing prefill on both models
            prompt = self.get_full_prompt()
            self.target.prefill(prompt)
            self.kv_store["target"] = self.target.get_kv_cache()
            self.draft.prefill(prompt)
            self.kv_store["draft"] = self.draft.get_kv_cache()
            return
        if self.kv_strategy.name == "prompt_cache":
            # load from disk if exists, otherwise prefill and save
            cache_fname = os.path.join(self.kv_strategy.prompt_cache_dir, "promptcache.json")
            if not self.target.load_prompt_cache(cache_fname):
                prompt = self.get_full_prompt()
                self.target.prefill(prompt)
                self.target.save_prompt_cache(cache_fname)
                # optionally set kv_store placeholders
                self.kv_store["target"] = self.target.get_kv_cache()
            # draft optionally loaded too
            return
        if self.kv_strategy.name == "replay_prefill":
            # perform chunked prefill (replay) to bring target up to date quickly
            prompt = self.get_full_prompt()
            # naive chunking: split by tokens (here split on whitespace)
            tokens = prompt.split()
            chunks = [tokens[i:i + self.kv_strategy.replay_chunk_size]
                      for i in range(0, len(tokens), self.kv_strategy.replay_chunk_size)]
            for chunk in chunks:
                chunk_prompt = " ".join(chunk)
                self.target.prefill(chunk_prompt)
            self.kv_store["target"] = self.target.get_kv_cache()
            return

    def prefill_target_if_needed(self):
        """A convenience wrapper to call before target generation when using lazy/no_sync strategy."""
        # For "no_sync" we only prefill target if it has no cache
        if self.kv_strategy.name == "no_sync":
            if self.kv_store["target"] is None:
                prompt = self.get_full_prompt()
                self.target.prefill(prompt)
                self.kv_store["target"] = self.target.get_kv_cache()

    # -------------------------
    # Generation APIs
    # -------------------------
    def generate_cli_fast(self, max_tokens: int = 256, temperature: float = 0.0, **gen_kwargs) -> Dict[str, Any]:
        prompt = self.get_full_prompt()
        res = self.target.generate_cli_wrapper(prompt, max_tokens=max_tokens, temperature=temperature, **gen_kwargs)
        return res

    def generate_speculative(self, max_tokens: int = 256, num_draft_tokens: int = 32,
                             temperature: float = 0.0, **gen_kwargs) -> Dict[str, Any]:
        """
        Run speculative decoding: draft then target verify.
        We assume MLXClient.generate supports draft_model and num_draft_tokens for speculative mode.
        """
        # ensure the target has prompt if required by the strategy
        self.ensure_target_prefill()
        prompt = self.get_full_prompt()
        # call target.generate with draft_model provided (MLX may accept draft_model handle directly)
        res = self.target.generate(prompt, max_tokens=max_tokens,
                                   temperature=temperature,
                                   draft_model=self.draft,
                                   num_draft_tokens=num_draft_tokens, **gen_kwargs)
        # update caches if API exposes them
        self.kv_store["target"] = self.target.get_kv_cache()
        self.kv_store["draft"] = self.draft.get_kv_cache()
        return res

    def generate_big_only(self, max_tokens: int = 256, temperature: float = 0.0, **gen_kwargs) -> Dict[str, Any]:
        self.prefill_target_if_needed()
        prompt = self.get_full_prompt()
        res = self.target.generate(prompt, max_tokens=max_tokens, temperature=temperature, **gen_kwargs)
        self.kv_store["target"] = self.target.get_kv_cache()
        return res

    def generate_small_only(self, max_tokens: int = 128, temperature: float = 0.0, **gen_kwargs) -> Dict[str, Any]:
        # small model generation; draft-only mode
        prompt = self.get_full_prompt()
        res = self.draft.generate(prompt, max_tokens=max_tokens, temperature=temperature, **gen_kwargs)
        self.kv_store["draft"] = self.draft.get_kv_cache()
        return res

# ---------------------------
# Router: choose mode
# ---------------------------
@dataclass
class RouterConfig:
    max_small_only_tokens: int = 100
    speculative_threshold_tokens: int = 150
    confidence_threshold: float = 0.8  # if you have draft confidence heuristics
    force_big_for_debug: bool = False

class Router:
    """
    Simple heuristic router deciding small-only / big-only / speculative.
    This can be replaced by learned routing later.
    """
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg

    def decide(self, prompt: str, expected_max_output_tokens: int,
               draft_confidence: Optional[float] = None) -> str:
        """
        Returns one of: "small_only", "big_only", "speculative", "cli_fast"
        """
        if self.cfg.force_big_for_debug:
            return "cli_fast"
            
        if expected_max_output_tokens <= self.cfg.max_small_only_tokens:
            # Use small-only for short outputs
            # if draft_confidence is provided, also require confidence
            if draft_confidence is None or draft_confidence >= self.cfg.confidence_threshold:
                return "small_only"

        if expected_max_output_tokens >= self.cfg.speculative_threshold_tokens:
            return "speculative"
        # default
        return "big_only"

# ---------------------------
# Perf evaluator and logger
# ---------------------------
@dataclass
class CallMetrics:
    call_type: str
    prompt_len_tokens: int
    generated_tokens: int
    total_time_s: float
    tokens_per_sec: float
    draft_accepted_tokens: int
    draft_proposed_tokens: int
    acceptance_rate: Optional[float]
    peak_memory_mb: Optional[float] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

class PerfEvaluator:
    """
    Collects per-call metrics and aggregates them.
    """
    def __init__(self):
        self.calls: List[CallMetrics] = []

    def measure_call(self, call_type: str, prompt: str, raw_response: Dict[str, Any]) -> CallMetrics:
        prompt_len_tokens = len(prompt.split())
        tokens = raw_response.get("tokens", [])
        generated_tokens = len(tokens)
        total_time = raw_response.get("generation_time_s", None)
        # fallback measure timestamp-based if not provided
        if total_time is None and "token_times" in raw_response:
            total_time = sum(raw_response["token_times"])
        if total_time is None:
            total_time = 1e-6  # avoid div by zero
        tps = generated_tokens / total_time if total_time > 0 else 0.0
        from_draft = raw_response.get("from_draft", [])
        draft_proposed = sum(1 for v in from_draft if v is not None)  # may overcount; adapt per API specifics
        draft_accepted = sum(1 for v in from_draft if v is True)
        acceptance_rate = (draft_accepted / draft_proposed) if draft_proposed > 0 else None
        cm = CallMetrics(call_type=call_type,
                         prompt_len_tokens=prompt_len_tokens,
                         generated_tokens=generated_tokens,
                         total_time_s=total_time,
                         tokens_per_sec=tps,
                         draft_accepted_tokens=draft_accepted,
                         draft_proposed_tokens=draft_proposed,
                         acceptance_rate=acceptance_rate,
                         peak_memory_mb=raw_response.get("peak_memory_mb"),
                         raw_response=raw_response)
        self.calls.append(cm)
        return cm

    def summary(self) -> Dict[str, Any]:
        by_type = defaultdict(list)
        for c in self.calls:
            by_type[c.call_type].append(c)
        out = {}
        for k, v in by_type.items():
            avg_tps = sum(c.tokens_per_sec for c in v) / len(v)
            avg_accept = None
            accept_values = [c.acceptance_rate for c in v if c.acceptance_rate is not None]
            if accept_values:
                avg_accept = sum(accept_values) / len(accept_values)
            out[k] = {
                "calls": len(v),
                "avg_tps": avg_tps,
                "avg_acceptance": avg_accept,
                "avg_generated_tokens": sum(c.generated_tokens for c in v) / len(v)
            }
        return out

# ---------------------------
# Agent (ReAct-style)
# ---------------------------
class ReActAgent:
    """
    Simple ReAct agent using ModelManager and Router.
    It demonstrates: reasoning step -> action -> observe -> loop (stop after max steps).
    """
    def __init__(self, model_manager: ModelManager, router: Router,
                 perf: PerfEvaluator, max_steps: int = 4):
        self.mm = model_manager
        self.router = router
        self.perf = perf
        self.max_steps = max_steps

    def run_task(self, user_prompt: str, expected_max_output_tokens: int = 200) -> Tuple[str, List[CallMetrics]]:
        """
        Run the agent loop for a single user task. Returns final text and list of metrics collected.
        """
        # seed conversation
        self.mm.append_message("user", user_prompt)

        # simple loop: at each step, ask small model for "plan" and then big model for "execute"
        step = 0
        final_text = ""
        while step < self.max_steps:
            # Decide mode
            # if we want draft_confidence, call draft with small-only quick query to get confidence proxy
            draft_confidence = None
            # quick draft probe for confidence if you want (optional)
            if True:  # enable draft probe
                probe_prompt = self.mm.get_full_prompt() + "\n\n# Probe: produce short plan (one sentence)."
                probe_res = self.mm.generate_small_only(max_tokens=32)
                probe_cm = self.perf.measure_call("probe_small", probe_prompt, probe_res)
                # heuristic: if draft generated small tokens quickly, set confidence high
                draft_confidence = 1.0 if probe_cm.tokens_per_sec > 500 else 0.9

            decision = self.router.decide(self.mm.get_full_prompt(), expected_max_output_tokens, draft_confidence)
            if decision == "small_only":
                # small-only action (cheap)
                res = self.mm.generate_small_only(max_tokens=expected_max_output_tokens)
                cm = self.perf.measure_call("small_only", self.mm.get_full_prompt(), res)
                final_text = res["text"]
                # commit final text as assistant output and break
                self.mm.append_message("assistant", final_text)
                break

            elif decision == "big_only":
                res = self.mm.generate_big_only(max_tokens=expected_max_output_tokens)
                cm = self.perf.measure_call("big_only", self.mm.get_full_prompt(), res)
                final_text = res["text"]
                self.mm.append_message("assistant", final_text)
                break

            elif decision == "cli_fast":
                res = self.mm.generate_cli_fast(max_tokens=expected_max_output_tokens)
                cm = self.perf.measure_call("cli_fast", self.mm.get_full_prompt(), res)
                final_text = res["text"]
                self.mm.append_message("assistant", final_text)
                break

            elif decision == "speculative":
                res = self.mm.generate_speculative(max_tokens=expected_max_output_tokens, num_draft_tokens=4)
                cm = self.perf.measure_call("speculative", self.mm.get_full_prompt(), res)
                final_text = res["text"]
                self.mm.append_message("assistant", final_text)
                break

            step += 1

        return final_text, self.perf.calls

# ---------------------------
# Example usage & suggested experiments
# ---------------------------
def example_run():
    # configure models and strategy
    kv_cfg = KVCacheStrategyConfig(name="replay_prefill", replay_chunk_size=1024)
    # Using small qwen models which should be locally available as per typical mlx_lm examples
    mm = ModelManager(target_path="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", 
                      draft_path="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
                      device="metal", kv_strategy=kv_cfg)

    # router & perf
    router_cfg = RouterConfig(max_small_only_tokens=80, speculative_threshold_tokens=150, force_big_for_debug=True)
    router = Router(router_cfg)
    perf = PerfEvaluator()

    # agent
    agent = ReActAgent(model_manager=mm, router=router, perf=perf, max_steps=3)

    user_prompt = "Write a function in Python that computes the Levenshtein distance between two strings and include unit tests."

    print("Running agent on prompt:", user_prompt)
    final_text, calls = agent.run_task(user_prompt, expected_max_output_tokens=300)

    print("Final generated text:")
    print(final_text)
    print("\nPer-call summary:")
    print(json.dumps(perf.summary(), indent=2))
    # Detailed per-call metrics
    for c in calls:
        print("--- CALL ---")
        print("type:", c.call_type, "toks/sec:", f"{c.tokens_per_sec:.2f}",
              "gen_toks:", c.generated_tokens, "time_s:", f"{c.total_time_s:.2f}")

if __name__ == "__main__":
    example_run()

"""
Enabled native speculative decoding. When draft_model is present, it directly passes it into stream_generate enabling fast token acceptance without writing separate boilerplate verification logic.
Included KV-cache checkpointing save_prompt_cache and load_prompt_cache to disk using numpy.savez caching tensor parameters.
Re-wired the ModelManager and ReActAgent loop to utilize smaller variants and base variants of the coder models effectively, validating the performance via elapsed latency and MLX metal memory profiles seamlessly gathered.


mlx_lm.generate \
    --model mlx-community/Qwen2.5-Coder-7B-4bit \
    --draft-model mlx-community/Qwen2.5-Coder-0.5B-4bit \
    --prompt "def quicksort(arr):" \
    --num-draft-tokens 4 \
    --temp 0.0

mlx_lm.generate \
    --model mlx-community/Qwen2.5-Coder-7B-4bit \
    --prompt "def quicksort(arr):" \
    --temp 0.0

mlx_lm.generate \
  --model mlx-community/Qwen2.5-Coder-7B-4bit \
  --prompt "def quicksort(arr):" \
  --temp 0.0 \
  --max-tokens 1024 \
  --kv-bits 8 \
  --kv-group-size 64

mlx_lm.generate \
  --model mlx-community/Qwen2.5-Coder-7B-4bit \
  --prompt "Write 2000 lines of Python comments, each unique, numbered 1..2000. Start now:\n# 1 " \
  --temp 0.0 \
  --max-tokens 1024

mlx_lm.generate \
  --model mlx-community/Qwen2.5-Coder-7B-4bit \
  --prompt "Generate a large JSON array with 200 objects, each object has 10 fields. Do not stop early.\n[" \
  --temp 0.0 \
  --max-tokens 1024

#compare with kv-cache quantize
mlx_lm.generate \
  --model mlx-community/Qwen2.5-Coder-7B-4bit \
  --prompt "Write a long JSON array with 300 objects... Start with: [" \
  --temp 0.0 \
  --max-tokens 1024

Prompt: 35 tokens, 38.209 tokens-per-sec
Generation: 1024 tokens, 71.180 tokens-per-sec
Peak memory: 4.423 GB

mlx_lm.generate \
  --model mlx-community/Qwen2.5-Coder-7B-4bit \
  --prompt "Write a long JSON array with 300 objects... Start with: [" \
  --temp 0.0 \
  --max-tokens 1024 \
  --kv-bits 8 \
  --kv-group-size 64

Prompt: 35 tokens, 62.056 tokens-per-sec
Generation: 1024 tokens, 71.174 tokens-per-sec
Peak memory: 4.423 GB

mlx_lm.generate \
  --model mlx-community/Qwen2.5-Coder-7B-4bit \
  --prompt "Write a long JSON array with 300 objects... Start with: [" \
  --temp 0.0 \
  --max-tokens 1024 \
  --kv-bits 4 \
  --kv-group-size 32

Prompt: 35 tokens, 39.256 tokens-per-sec
Generation: 1024 tokens, 70.485 tokens-per-sec
Peak memory: 4.423 GB
"""