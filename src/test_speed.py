from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
import time

model, tok = load("mlx-community/Qwen2.5-Coder-7B-4bit")
prompt = "def quicksort(arr):"

sampler = make_sampler(temp=0.0)

print("Using generate():")
res = generate(model, tok, prompt, sampler=sampler, verbose=True)

print("Using stream_generate loop:")
start = time.time()
n_toks = 0
for resp in stream_generate(model, tok, prompt, sampler=sampler, max_tokens=100):
    n_toks += 1
end = time.time()
print(f"TPS: {n_toks / (end - start)}")
