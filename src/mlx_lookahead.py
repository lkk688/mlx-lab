from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler  # This is the key!
import time

# Load your 4-bit model
model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-4bit")

prompt = "def quicksort(arr):"

print("\n🚀 Running N-Gram Lookahead (Model-Free Speculation)...")

# Define the sampler for deterministic (greedy) decoding
# Setting temp=0.0 is best for code and speculation accuracy
sampler = make_sampler(temp=0.0)

start = time.time()

# num_draft_tokens: Triggers the model-free N-Gram speculation
response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=100,
    num_draft_tokens=4, 
    sampler=sampler,  # Pass the sampler instead of 'temp'
    verbose=True
)

end = time.time()
print(f"\nTotal Time: {end - start:.2f}s")