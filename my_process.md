
```bash
Installing collected packages: hf_transfer
Successfully installed hf_transfer-0.1.9
(mypy311) kaikailiu@Kaikais-MacBook-Pro Coder % export HF_HUB_ENABLE_HF_TRANSFER=1
(mypy311) kaikailiu@Kaikais-MacBook-Pro Coder % huggingface-cli download --local-dir clip_download mlx-community/clip-vit-base-patch32
zsh: command not found: huggingface-cli
(mypy311) kaikailiu@Kaikais-MacBook-Pro Coder % curl -LsSf https://hf.co/cli/install.sh | bash
[INFO] Installing Hugging Face CLI...
[INFO] OS: macos
[INFO] Force reinstall: false
[INFO] Install dir: /Users/kaikailiu/.hf-cli
[INFO] Bin dir: /Users/kaikailiu/.local/bin
[INFO] Skip PATH update: false
[INFO] Using Python: Python 3.13.2
[INFO] Creating directories...
[INFO] Creating virtual environment...
[INFO] Virtual environment already exists; reusing (pass --force to recreate)
[INFO] Installing/upgrading Hugging Face CLI (latest)...
[INFO] Installation output suppressed; set HF_CLI_VERBOSE_PIP=1 for full logs
[INFO] Linking hf CLI into /Users/kaikailiu/.local/bin...
[INFO] hf available at /Users/kaikailiu/.local/bin/hf (symlink to venv)
[INFO] Run without touching PATH: env PATH="/Users/kaikailiu/.local/bin:$PATH" hf --help
[INFO] Verifying installation...
[SUCCESS] Hugging Face CLI installed successfully!
[INFO] CLI location: /Users/kaikailiu/.local/bin/hf
[INFO] Installation directory: /Users/kaikailiu/.hf-cli
[INFO] Current version: 1.4.1
[INFO] 
[INFO] To uninstall the Hugging Face CLI, run:
[INFO]   rm -rf /Users/kaikailiu/.hf-cli
[INFO]   rm -f /Users/kaikailiu/.local/bin/hf
[INFO] 
[INFO]   Remove any PATH edits you made manually.
[SUCCESS] hf CLI ready!
[INFO] Binary: /Users/kaikailiu/.local/bin/hf
[INFO] Virtualenv: /Users/kaikailiu/.hf-cli
[INFO] Try it now: env PATH="/Users/kaikailiu/.local/bin:$PATH" hf --help
[INFO] Examples:
[INFO]   hf auth login
[INFO]   hf download deepseek-ai/DeepSeek-R1
[INFO]   hf jobs run python:3.12 python -c 'print("Hello from HF CLI!")'
[INFO] 
(mypy311) kaikailiu@Kaikais-MacBook-Pro Coder % hf download --local-dir clip_download mlx-community/clip-vit-base-patch32
Downloading (incomplete total...): 0.00B [00:00, ?B/s]                            Still waiting to acquire lock on clip_download/.cache/huggingface/.gitignore.lock (elapsed: 0.1 seconds)
Fetching 7 files: 100%|██████████████████████████████| 7/7 [00:14<00:00,  2.04s/it]
Download complete: : 607MB [00:14, 41.9MB/s]              /Users/kaikailiu/Documents/Coder/clip_download
Download complete: : 607MB [00:14, 42.4MB/s]
```

# Qwen MLX
```bash
pip install mlx-lm
# Or for vision models
pip install mlx-vlm

python -m mlx_lm.generate \
  --model mlx-community/Qwen3.5-9B-OptiQ-4bit \
  --prompt "Explain quantum computing in simple terms:" \
  --max-tokens 200
#mlx-community/Qwen3.5-27B-4bit

pip install -U mlx-vlm
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-4B-MLX-4bit --max-tokens 100


python -m mlx_vlm generate \
    --model mlx-community/Qwen3.5-9B-4bit \
    --image hybrid_eval_accuracy.png \
    --prompt "Tell me what you see in this image." \
    --max-tokens 2048 \
    --temp 0.3

python -m mlx_lm.generate \
    --model mlx-community/Qwen3.5-9B-OptiQ-4bit \
    --prompt "作为人工智能，你能帮我写一段快排算法的 Python 代码吗？"
```

Convert to 4bit: https://huggingface.co/Qwen/Qwen3.5-9B
```bash
(mypy311) kaikailiu@Kaikais-MacBook-Pro Coder % mlx_lm.convert --hf-path Qwen/Qwen3.5-9B -q --q-bits 4
[INFO] Loading
Fetching 13 files: 100%|████████████████████████████████████████████████████████████| 13/13 [04:58<00:00, 23.00s/it]
Download complete: : 19.3GB [04:59, 64.6MB/s]                                        | 6/13 [04:58<06:03, 51.97s/it]
[INFO] Quantizing
[INFO] Quantized model with 4.501 bits per weight.
README.md: 77.6kB [00:00, 6.77MB/s]

mlx_lm.generate --model mlx_model --prompt "Use Python to write a basic neural network from scratch via Apple's MLX framework." --max-tokens 1024

Prompt: 27 tokens, 18.570 tokens-per-sec
Generation: 1024 tokens, 45.549 tokens-per-sec
Peak memory: 5.654 GB
```

```bash
mlx_lm.convert --hf-path Qwen/Qwen3.5-35B-A3B -q --q-bits 4

mlx_lm.convert --hf-path Qwen/Qwen3.5-27B -q --q-bits 4
``