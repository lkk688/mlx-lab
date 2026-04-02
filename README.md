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
- `notebooks/`: Directory reserved for Jupyter notebooks (exploratory data analysis & tutorials).
- `my_process.md`: My personal notes on downloading and converting models.

*(Note: Heavy model weights and external cloned repos like `mlx_clip` are intentionally `.gitignore`d to keep the repository lightweight.)*

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

*For more detailed steps on model quantization and usage, please refer to [my_process.md](my_process.md).*

## 📈 Benchmarks

Check out the `assets/` folder for visual plots of our experiments, including accuracy vs. speedup comparisons when using Medusa heads and hybrid evaluation engines.

---
*Built with ❤️ for the MLX community.*
