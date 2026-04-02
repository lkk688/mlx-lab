# mlx_embedding_engine.py

import mlx.core as mx
import numpy as np
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from functools import lru_cache

# Requires: pip install mlx-embeddings
from mlx_embeddings import EmbeddingModel


# -----------------------------
# Config
# -----------------------------

POPULAR_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5-small": "intfloat/e5-small-v2",
    "e5-base": "intfloat/e5-base-v2",
    "nomic": "nomic-ai/nomic-embed-text-v1",
}


@dataclass
class EmbeddingConfig:
    model_name: str = "bge-base"
    batch_size: int = 32
    normalize: bool = True
    use_fp16: bool = True


# -----------------------------
# Embedding Engine
# -----------------------------

class MLXEmbeddingEngine:

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        model_id = POPULAR_MODELS[config.model_name]
        print(f"[Embedding] Loading {model_id}")

        self.model = EmbeddingModel(
            model_id,
            dtype=mx.float16 if config.use_fp16 else mx.float32
        )

        self.dim = self.model.embedding_dim
        print(f"[Embedding] Dimension: {self.dim}")

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def embed(self, texts: List[str]) -> Dict:
        """
        High-throughput batch embedding.
        """
        start = time.time()

        embeddings = self.model.embed(
            texts,
            batch_size=self.config.batch_size
        )

        embeddings = np.array(embeddings)

        if self.config.normalize:
            embeddings = self._normalize(embeddings)

        end = time.time()

        total_tokens = sum(len(t.split()) for t in texts)
        total_time = end - start

        return {
            "embeddings": embeddings,
            "dim": self.dim,
            "texts": len(texts),
            "tokens": total_tokens,
            "time_s": total_time,
            "tokens_per_sec": total_tokens / total_time if total_time > 0 else None,
            "texts_per_sec": len(texts) / total_time if total_time > 0 else None
        }

    def embed_one(self, text: str):
        return self.embed([text])["embeddings"][0]


if __name__ == "__main__":

    cfg = EmbeddingConfig(model_name="bge-base", batch_size=64)
    engine = MLXEmbeddingEngine(cfg)

    texts = ["def quicksort(arr): return sorted(arr)"] * 200

    result = engine.embed(texts)

    print(result)