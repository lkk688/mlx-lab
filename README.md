# 🚀 MLX Lab

Welcome to **MLX Lab**! This repository is a collection of experiments, optimizations, and benchmarks for running large language models (LLMs) and vision-language models (VLMs) on Apple Silicon using the [MLX framework](https://github.com/ml-explore/mlx).

## ✨ Features & Experiments

This lab focuses on pushing the boundaries of local inference on Mac devices:
- **Speculative Decoding & Medusa**: Implementations of Medusa heads and lookahead decoding to accelerate generation speeds.
- **Hybrid & Elastic Inference**: Custom engines (`mlx_elastic_inference`, `mlx_hybrid_engine`) designed for optimal resource utilization.
- **Vision-Language Models (VLMs)**: Integration and testing of models like Qwen-Vision and CLIP.
- **Quantization & Benchmarking**: Scripts to convert, quantize (e.g., 4-bit), and benchmark models to find the sweet spot between speed and memory.

## 📁 Repository Structure

The project has been organized to keep code, assets, and results clean:

- `src/`: Core Python scripts, including custom inference engines, Medusa implementations, and benchmarking scripts.
- `assets/`: Images and static files used for VLM testing and performance plots.
- `results/`: Output logs, cost metrics (`.json`), and model weights/cache (`.npz`, `.pth`).


## 🛠️ Getting Started

### Prerequisites

Make sure you have an Apple Silicon Mac (M1/M2/M3) and a Python environment set up.

```bash
# Install core MLX packages
pip install mlx-lm mlx-vlm hf_transfer
```

### Running Experiments

To run an example script (e.g., Qwen Vision generation):

```bash
# Example running from the src directory
cd src
python qwen_vision.py
```

## 📖 Qwen MLX Tutorial

This section provides a step-by-step guide to downloading, running, and quantizing Qwen models using Apple's MLX framework.

### 1. Fast Model Downloading with Hugging Face CLI
To speed up model downloads, we recommend installing the Hugging Face CLI and enabling `hf_transfer`.

```bash
# Install the Hugging Face CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Enable fast transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Example: Download a CLIP model to a local directory
hf download --local-dir clip_download mlx-community/clip-vit-base-patch32
```

### 2. Installing MLX Dependencies
Depending on whether you are running text-only or vision-language models, you will need `mlx-lm` or `mlx-vlm`.

```bash
# For standard Language Models (LLMs)
pip install mlx-lm

# For Vision-Language Models (VLMs)
pip install -U mlx-vlm
```

### 3. Text Generation with Qwen LLMs
You can generate text directly from the command line using the `mlx_lm.generate` module.

```bash
# Basic generation with a 4-bit quantized Qwen3.5 9B model
python -m mlx_lm.generate \
  --model mlx-community/Qwen3.5-9B-OptiQ-4bit \
  --prompt "Explain quantum computing in simple terms:" \
  --max-tokens 200

# Chinese prompt example
python -m mlx_lm.generate \
    --model mlx-community/Qwen3.5-9B-OptiQ-4bit \
    --prompt "作为人工智能，你能帮我写一段快排算法的 Python 代码吗？"
```

### 4. Vision Generation with Qwen VLMs
For models that understand images (like Qwen Vision), use the `mlx_vlm.generate` module.

```bash
# Example 1: Basic VLM generation
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-4B-MLX-4bit --max-tokens 100

# Example 2: Analyzing a specific image
python -m mlx_vlm generate \
    --model mlx-community/Qwen3.5-9B-4bit \
    --image assets/hybrid_eval_accuracy.png \
    --prompt "Tell me what you see in this image." \
    --max-tokens 2048 \
    --temp 0.3
```

### 5. Quantizing Models to 4-bit
You can convert standard Hugging Face models to MLX-compatible 4-bit quantized models to save memory and increase inference speed.

```bash
# Convert Qwen3.5-9B to 4-bit
mlx_lm.convert --hf-path Qwen/Qwen3.5-9B -q --q-bits 4

# Other examples:
mlx_lm.convert --hf-path Qwen/Qwen3.5-35B-A3B -q --q-bits 4
mlx_lm.convert --hf-path Qwen/Qwen3.5-27B -q --q-bits 4
```

After conversion, your quantized model will be saved in an `mlx_model` directory. You can test the newly converted model like this:

```bash
mlx_lm.generate \
    --model mlx_model \
    --prompt "Use Python to write a basic neural network from scratch via Apple's MLX framework." \
    --max-tokens 1024
```

## 💎 Gemma 4 MLX Tutorial

This section provides a step-by-step guide to downloading, running, and quantizing Gemma 4 models using Apple's MLX framework.

### 1. Gemma 4 Architecture Overview
All Gemma 4 variants share a common, highly optimized architecture:
- **Sliding + Full Attention**: A repeating pattern of 5 sliding window layers followed by 1 full attention layer.
- **KV Sharing**: Later layers reuse Key/Value states from earlier layers (specifically in 2B/4B models).
- **K-eq-V Attention**: Full attention layers use key states as values (in 26B/31B models).
- **Per-layer Inputs**: Additional per-layer token embeddings (in 2B/4B models).
- **MoE (Mixture of Experts)**: 128 experts with top-8 routing using `gather_mm` (exclusive to the 26B model).
- **Multimodal Encoders**: A shared SigLIP2 vision encoder across all variants, and a Conformer audio encoder (12 blocks) for the 2B/4B models.

### 2. Installation
Since Gemma 4 is very new, it may not be fully supported in the standard release yet. You might need to install the latest development version of `mlx-vlm`.

```bash
# Upgrade core MLX packages
pip install --upgrade mlx mlx-lm mlx-vlm

# Uninstall the stable mlx-vlm release
pip uninstall -y mlx-vlm

# Install the latest main branch from GitHub
pip install git+https://github.com/Blaizzy/mlx-vlm.git
```

### 3. Core Capabilities

#### Vision-Language Generation
Analyze images by passing a local path or URL:
```bash
mlx_vlm.generate \
  --model google/gemma-4-E4B-it \
  --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg \
  --prompt "Describe this image in detail"
```
*(Example Performance in M1Max32G: ~56.8 prompt tokens/sec, ~29.4 generation tokens/sec, Peak memory: 16.75 GB)*

#### Text Generation
Standard text querying:
```bash
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --prompt "What is the capital of China?" \
  --max-tokens 500 \
  --temperature 0
```

#### Audio Understanding (2B/4B Models Only)
Gemma 4 (2B/4B) natively supports audio transcription and reasoning:
```bash
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --audio assets/reply_en.wav \
  --prompt "Transcribe this audio" \
  --max-tokens 500 \
  --temperature 0
```

#### Thinking Mode (Chain-of-Thought)
Enable the model's internal reasoning process before outputting the final answer:
```bash
python -m mlx_vlm.generate \
  --model google/gemma-4-e4b-it \
  --prompt "I want to do a car wash that is 50 meters away, should I walk or drive?" \
  --enable-thinking \
  --thinking-budget 512 \
  --max-tokens 2000 \
  --temperature 0
```

### 4. KV Cache Quantization & Experiments
When running larger models like the 31B variant, quantizing the KV cache is critical for fitting the model into memory. However, extreme quantization can lead to model degradation. Here are the findings from our experiments on an M1 Max (32GB):

#### ❌ The Pitfalls: TurboQuant and Low Bit-rates
Using experimental features like TurboQuant (https://github.com/Blaizzy/mlx-vlm/tree/pc/turbo-quant) or extremely low KV bits (e.g., 3.5 or 4-bit) can cause the model's attention mechanism to collapse.

```bash
# Example of a failing configuration (Attention Collapse / Degradation over time)
# Symptoms: Repeating characters like "額額額額額額....." or "if __- a_ a_ a_"
mlx_vlm.generate \
  --model "mlx-community/gemma-4-31b-it-4bit" \
  --prompt "Help me write a python code to plot a comparison..." \
  --kv-bits 4 \
  --max-tokens 2000 \
  --temperature 0.1
```

#### ✅ Recommended Configurations
To avoid quantization degradation while maintaining efficiency, you have two reliable options:

**Option A: Increase KV Cache Precision (8-bit)**
Using an 8-bit KV cache prevents degradation at the cost of slightly higher memory usage.
```bash
mlx_vlm.generate \
  --model "mlx-community/gemma-4-31b-it-4bit" \
  --prompt "Help me write a python code to plot a comparison of the accuracy and speedup of speculative decoding." \
  --kv-bits 8 \
  --max-tokens 2000 \
  --temperature 0.1
```
*(M1 Max 32G Performance: ~14.78 generation tokens/sec, Peak memory: 19.46 GB)*

**Option B: Increase Temperature (Default Precision)**
If you must use lower precision, slightly increasing the temperature (e.g., `0.4`) can help the model escape repetitive generation loops.
```bash
mlx_vlm.generate \
  --model "mlx-community/gemma-4-31b-it-4bit" \
  --prompt "Help me write a python code to plot a comparison of the accuracy and speedup of speculative decoding." \
  --max-tokens 2000 \
  --temperature 0.4
```
*(M1 Max 32G Performance: ~14.93 generation tokens/sec, Peak memory: 19.44 GB)*

### Local OpenAI API Server

```bash
# 启动本地服务，默认运行在 8080 端口
python -m mlx_vlm.server \
--model "mlx-community/gemma-4-31b-it-4bit" \
--port 8080 \
--host 0.0.0.0
```

Test via curl:
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer not-needed" \
  -d '{
    "model": "mlx-community/gemma-4-31b-it-4bit",
    "messages": [
      {
        "role": "user",
        "content": "Hello! Are you running as an OpenAI compatible API?"
      }
    ],
    "temperature": 0.4,
    "max_tokens": 500
  }'
```

```bash
python -m mlx_vlm.server \
--model "google/gemma-4-E4B-it" \
--port 8010 \
--host 0.0.0.0



curl http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer mymac" \
  -d '{
    "model": "google/gemma-4-E4B-it",
    "messages": [
      {
        "role": "user",
        "content": "You are a helpful AI assistant. Use exactly one sentence to describe what a Paged KV Cache is."
      }
    ],
    "temperature": 0.4,
    "max_tokens": 100
  }'
```

Run our evaluation function:
```bash
python -m mlx_vlm.server --model "google/gemma-4-E4B-it" --port 8010 --host 0.0.0.0 --max-kv-size 8192

python src/evaluator_main.py --action speed --model "google/gemma-4-E4B-it" --base-url "http://0.0.0.0:8010/v1" --base-url-name "gemma-4-E4B-it" --output-json results/gemma4_speed.json

```
ValueError: [broadcast_shapes] Shapes (1,2048,42,256) and (1,4288,42,256) cannot be broadcast. This means the MLX engine hardcoded or implicitly constrained the KV cache context size internally to exactly 2048 tokens for some attention mechanisms inside google/gemma-4-E4B-it . When you pass a 4096 prompt ( 4288 actual tokens), MLX attempts to broadcast the 4288-length query tensor against a 2048-length cache tensor, causing an immediate shape mismatch crash.

Even though added --max-kv-size 8192 , the underlying gemma-4-E4B-it model logic in mlx_vlm 's current version seems to have a bug where the per_layer_inputs or sliding window attention layers are statically fixed or not dynamically respecting the --max-kv-size override.

Only use "--prefill-tokens 1024" to avoid this issue.
```bash
python src/evaluator_main.py --action speed --model "google/gemma-4-E4B-it" --base-url "http://0.0.0.0:8010/v1" --base-url-name "gemma-4-E4B-it" --output-json results/gemma4_e4b_m1maxspeed.json --prefill-tokens 1024
```
Success: Data saved to: mlx-lab/results/gemma4_e4b_m1maxspeed.json

```bash
python -m mlx_vlm.server \
--model "mlx-community/gemma-4-31b-it-4bit" \
--port 8080 \
--host 0.0.0.0

python src/evaluator_main.py --action speed --model "mlx-community/gemma-4-31b-it-4bit" --base-url "http://0.0.0.0:8010/v1" --base-url-name "gemma-4-31b-it-m1max" --output-json results/gemma4_31b_m1maxspeed.json --prefill-tokens 1024
```
Data saved to: mlx-lab/results/gemma4_31b_m1maxspeed.json

The reason MLX server crashes on prompts larger than 1024 tokens, despite gemma-4-E4B-it officially supporting a 128K context window, comes down to a specific architectural feature of Gemma 4 and how mlx_vlm currently implements it. Based on the official Google DeepMind specs from the Hugging Face page you referenced:
 Sliding Window:
- E2B: 512 tokens
- E4B: 512 tokens
- 31B: 1024 tokens
Gemma 4 uses a Hybrid Attention Mechanism that interleaves local sliding window attention (where the model only looks at the last 512 or 1024 tokens) with full global attention layers.

Why the MLX Crash Happens ([broadcast_shapes]): When you send a prompt of 4096 tokens to mlx_vlm :

1. The global attention layers can handle all 4096 tokens just fine (up to 128K).
2. However, the sliding window attention layers are strictly configured to only cache and process the last 512 tokens (for E4B) or 1024 tokens (for 31B).
3. Because mlx_vlm 's current implementation of Gemma 4's hybrid architecture is very new, there is a bug in the tensor broadcasting logic during the prefill phase. The MLX engine tries to perform math (broadcasting) between the full prompt tensor (e.g., shape 4288 ) and the sliding window cache tensor (which is statically clamped to the sliding window size, e.g., 1024 or 2048 ), resulting in: ValueError: [broadcast_shapes] Shapes (1,2048,42,256) and (1,4288,42,256) cannot be broadcast.

The model itself supports 128K context, but the MLX backend implementation of the sliding window cache is currently failing to chunk or process prompts larger than the sliding window boundary in a single prefill pass.

### Do the evaluation for Llama.cpp
Test the gemma 4 in Llama.cpp on RTX 5090:

```bash
CUDA_VISIBLE_DEVICES=1 ./build/bin/llama-server \
  -m ../gemma4/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --mmproj ../gemma4/mmproj-gemma-4-26B-A4B-it-f16.gguf \
  -ngl 99 \
  -c 65536 \
  -np 4 \
  -b 4096 \
  -ub 4096 \
  -fa on \
  --defrag-thold 0.1 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --host 0.0.0.0 --port 8011

curl http://100.65.193.60:8011/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant and an expert in Deep learning. "},
      {"role": "user", "content": "Explain what a Paged KV Cache is."}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
  }'

python src/evaluator_main.py --action speed --model "gemma-4" --base-url "http://100.65.193.60:8011/v1" --base-url-name "gemma-4-llamacpp-rtx5090" --output-json gemma-4-llamacpp-rtx5090.json

```
Batch 1x4096 (different): TTFT 848.6ms, TPOT 8.1ms, pp TPS 5042.2, tg TPS 123.9

## oMLX
Download from https://github.com/jundot/omlx/releases.
```bash
#find the local model
find ~/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots -maxdepth 1 -mindepth 1

/Users/xxx/.cache/huggingface/hub/models--mlx-community--gemma-4-31b-it-4bit/snapshots/535c5606372deb5d5ab7e29280f111ef2a8e084e
```
Testing
```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer mymac" \
  -d '{
    "model": "gemma-4-31b-it-4bit",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful AI assistant."
      },
      {
        "role": "user",
        "content": "Use exactly one sentence to describe what a Paged KV Cache is."
      }
    ],
    "temperature": 0.4,
    "max_tokens": 100
  }'
```
Output these:
```bash
{"id":"chatcmpl-0a71fc2f","object":"chat.completion","created":1775165441,"model":"gemma-4-31b-it-4bit","choices":[{"index":0,"message":{"role":"assistant","content":"ែ ফলে Lately disediakanायचे ফলে ফলে ফলে ফলে Lately Lately額 �ិী ICF額ायचे額額  ायचे บ้าง บ้าง̠額-額額額額額額額ायचे บ้าง บ้าง額額額ायचे บ้างありがとうございましたありがとうございました額額ायचेありがとうございましたありがとうございました̠額 บ้าง candlest額額 額額額額額額額ायचे額額額額額額ायचे額額額額額 及---_---// ಆಗ-�����������","reasoning_content":null,"tool_calls":null},"finish_reason":"length"}],"usage":{"prompt_tokens":39,"completion_tokens":100,"total_tokens":139,"input_tokens":39,"output_tokens":100,"cached_tokens":null,"model_load_duration":null,"time_to_first_token":null,"total_time":null,"prompt_eval_duration":null,"generation_duration":null,"prompt_tokens_per_second":null,"generation_tokens_per_second":null}}
```

```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer mymac" \
  -d '{
    "model": "gemma-4-31b-it-4bit",
    "messages": [
      {
        "role": "user",
        "content": "You are a helpful AI assistant. Use exactly one sentence to describe what a Paged KV Cache is."
      }
    ],
    "temperature": 0.4,
    "max_tokens": 100
  }'
{"id":"chatcmpl-b0c47694","object":"chat.completion","created":1775165717,"model":"gemma-4-31b-it-4bit","choices":[{"index":0,"message":{"role":"assistant","content":"//ងীងীীងងীীងীীងলীীী//額ীীীপ্রী್ರীীী্পরপ্রীীী্পীরীী্পীীরীীীীরীীীররীীীীর্পীপ্রীীী্পীীীীীীীীীীীীীীীীীীীীীীীীীীীীীীীীীী","reasoning_content":null,"tool_calls":null},"finish_reason":"length"}],"usage":{"prompt_tokens":34,"completion_tokens":100,"total_tokens":134,"input_tokens":34,"output_tokens":100,"cached_tokens":null,"model_load_duration":null,"time_to_first_token":null,"total_time":null,"prompt_eval_duration":null,"generation_duration":null,"prompt_tokens_per_second":null,"generation_tokens_per_second":null}}%
```

Change to gemma-4-E4B-it:
```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer mymac" \
  -d '{
    "model": "gemma-4-E4B-it",
    "messages": [
      {
        "role": "user",
        "content": "You are a helpful AI assistant. Use exactly one sentence to describe what a Paged KV Cache is."
      }
    ],
    "temperature": 0.4,
    "max_tokens": 100
  }'
{"error":{"message":"Internal server error","type":"server_error","param":null,"code":null}}
```
## 📈 Benchmarks

Check out the `assets/` folder for visual plots of our experiments, including accuracy vs. speedup comparisons when using Medusa heads and hybrid evaluation engines.

---
*Built with ❤️ for the MLX community.*
