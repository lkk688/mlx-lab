from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
import time

model, tok = load("mlx-community/Qwen2.5-Coder-7B-4bit")
prompt = "def levenshtein_distance(s1, s2):"
sampler = make_sampler(temp=0.0)

print("Using generate() for 300 tokens:")
res = generate(model, tok, prompt, sampler=sampler, max_tokens=300, verbose=True)

print("Using stream_generate loop for 300 tokens:")
start = time.perf_counter()
n_toks = 0
for resp in stream_generate(model, tok, prompt, sampler=sampler, max_tokens=300):
    n_toks += 1
end = time.perf_counter()
print(f"TPS: {n_toks / (end - start)}")
