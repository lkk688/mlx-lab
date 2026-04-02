# bench_embed.py
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Optional
import time
# mlx_text_embed.py
import time
import numpy as np
import mlx.core as mx
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import os
import sys

# Set up mlx_clip package scope
sys.path.insert(0, os.path.abspath("mlx_clip"))
import clip

from mlx_embeddings.utils import load  # <- correct for your installed package
# embed_service.py
import asyncio
from typing import List, Any, Dict


@dataclass
class BatchStats:
    batches: int = 0
    items: int = 0
    total_time_s: float = 0.0

class AsyncBatcher:
    """
    Collects requests for a short window and processes them in one batch.
    Greatly improves throughput for MLX embedding models.
    """

    def __init__(
        self,
        process_batch_fn: Callable[[List[Any]], Any],
        max_batch_size: int = 64,
        flush_interval_ms: int = 5,
        max_queue: int = 10_000,
        name: str = "batcher",
    ):
        self.process_batch_fn = process_batch_fn
        self.max_batch_size = max_batch_size
        self.flush_interval_ms = flush_interval_ms
        self.max_queue = max_queue
        self.name = name

        self._q: asyncio.Queue[Tuple[Any, asyncio.Future]] = asyncio.Queue(maxsize=max_queue)
        self._task: Optional[asyncio.Task] = None
        self.stats = BatchStats()

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def submit(self, item: Any) -> Any:
        fut = asyncio.get_running_loop().create_future()
        await self._q.put((item, fut))
        return await fut

    async def _run_loop(self):
        while True:
            # wait for at least one item
            item, fut = await self._q.get()
            batch_items = [item]
            batch_futs = [fut]

            # small coalescing window
            t_deadline = time.time() + self.flush_interval_ms / 1000.0
            while len(batch_items) < self.max_batch_size:
                timeout = t_deadline - time.time()
                if timeout <= 0:
                    break
                try:
                    item2, fut2 = await asyncio.wait_for(self._q.get(), timeout=timeout)
                    batch_items.append(item2)
                    batch_futs.append(fut2)
                except asyncio.TimeoutError:
                    break

            t0 = time.time()
            try:
                results = self.process_batch_fn(batch_items)
                if len(results) != len(batch_futs):
                    raise RuntimeError(
                        f"[{self.name}] process_batch_fn returned {len(results)} results for {len(batch_futs)} items"
                    )
                for f, r in zip(batch_futs, results):
                    if not f.cancelled():
                        f.set_result(r)
            except Exception as e:
                for f in batch_futs:
                    if not f.cancelled():
                        f.set_exception(e)
            finally:
                dt = time.time() - t0
                self.stats.batches += 1
                self.stats.items += len(batch_items)
                self.stats.total_time_s += dt



@dataclass
class EmbedPerf:
    items: int
    tokens: int
    time_s: float
    items_per_s: float
    tokens_per_s: float

class MLXTextEmbedder:
    """
    Multilingual text embedding using mlx-embeddings.
    Default model: mlx-community/multilingual-e5-base-mlx (EN+ZH strong).
    """
    def __init__(
        self,
        model_name: str = "mlx-community/multilingual-e5-base-mlx",
        max_length: int = 512,
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.model, self.tokenizer = load(model_name)
        # warmup: triggers Metal compilation
        self.embed(["warmup"], with_perf=False)

    def embed(self, texts: List[str], with_perf: bool = True) -> Dict[str, Any]:
        t0 = time.time()

        # Token count proxy (use tokenizer if you want exact token counts)
        approx_tokens = sum(len(t.split()) for t in texts)

        # Tokenize batch
        hf_tok = self.tokenizer._tokenizer
        batch_res = hf_tok(
            texts,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: mx.array(v) for k, v in batch_res.items()}

        # Forward pass
        outputs = self.model(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
        )

        # Per docs: outputs.text_embeds is mean pooled + normalized embeddings.  [oai_citation:6‡PyPI](https://pypi.org/project/mlx-embeddings/)
        embeds = np.array(outputs.text_embeds)

        dt = time.time() - t0
        perf = None
        if with_perf:
            perf = EmbedPerf(
                items=len(texts),
                tokens=approx_tokens,
                time_s=dt,
                items_per_s=(len(texts) / dt) if dt > 0 else 0.0,
                tokens_per_s=(approx_tokens / dt) if dt > 0 else 0.0,
            )

        return {"embeddings": embeds, "perf": perf}

class MLXCLIPEmbedder:
    """
    CLIP image/text embedding using mlx_clip.
    """
    def __init__(
        self,
        mlx_model_dir: str = "./clip-vit-base-patch32",
        batch_size: int = 32,
    ):
        self.mlx_model_dir = mlx_model_dir
        self.batch_size = batch_size
        self.model, _, self.img_processor = clip.load(mlx_model_dir)
        
        from transformers import CLIPTokenizer
        # Use HuggingFace robust tokenizer for CLIP instead of the naive one from mlx-examples
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # partial warmup
        self.embed_text(["warmup"], with_perf=False)

    def embed_text(self, texts: List[str], with_perf: bool = True) -> Dict[str, Any]:
        t0 = time.time()
        approx_tokens = sum(len(t.split()) for t in texts)

        # Tokenize with padding and truncation to avoid ragged array errors
        batch_res = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=77, 
            return_tensors="np"
        )
        input_ids = mx.array(batch_res["input_ids"])
        
        # Forward pass
        output = self.model(input_ids=input_ids)
        embeds = np.array(output.text_embeds)

        dt = time.time() - t0
        perf = None
        if with_perf:
            perf = EmbedPerf(
                items=len(texts),
                tokens=approx_tokens,
                time_s=dt,
                items_per_s=(len(texts) / dt) if dt > 0 else 0.0,
                tokens_per_s=(approx_tokens / dt) if dt > 0 else 0.0,
            )
        return {"embeddings": embeds, "perf": perf}

class EmbeddingService:
    """
    - text_embedder: multilingual text->text retrieval
    - clip_embedder: text->image retrieval (uses CLIP text encoder)
    """
    def __init__(
        self,
        text_model: str = "mlx-community/multilingual-e5-base-mlx",
        text_batch_size: int = 64,
        clip_mlx_model_dir: str = "./clip-vit-base-patch32",
        clip_batch_size: int = 32,
        flush_ms: int = 5,
    ):
        self.text = MLXTextEmbedder(model_name=text_model, batch_size=text_batch_size)
        self.clip = MLXCLIPEmbedder(mlx_model_dir=clip_mlx_model_dir, batch_size=clip_batch_size)

        # Each batcher returns ONE embedding per input item
        self.text_batcher = AsyncBatcher(
            process_batch_fn=self._process_text_batch,
            max_batch_size=text_batch_size,
            flush_interval_ms=flush_ms,
            name="text_batcher",
        )
        self.clip_text_batcher = AsyncBatcher(
            process_batch_fn=self._process_clip_text_batch,
            max_batch_size=clip_batch_size,
            flush_interval_ms=flush_ms,
            name="clip_text_batcher",
        )

    async def start(self):
        await self.text_batcher.start()
        await self.clip_text_batcher.start()

    def _process_text_batch(self, texts: List[str]):
        out = self.text.embed(texts, with_perf=True)
        embs = out["embeddings"]
        # Return per-item embeddings
        return [embs[i] for i in range(embs.shape[0])]

    def _process_clip_text_batch(self, texts: List[str]):
        out = self.clip.embed_text(texts, with_perf=True)
        embs = out["embeddings"]
        return [embs[i] for i in range(embs.shape[0])]

    async def embed_text(self, text: str):
        return await self.text_batcher.submit(text)

    async def embed_for_image_search(self, text: str):
        return await self.clip_text_batcher.submit(text)

    def stats(self) -> Dict[str, Any]:
        return {
            "text_batcher": self.text_batcher.stats,
            "clip_text_batcher": self.clip_text_batcher.stats,
        }

async def main():
    svc = EmbeddingService(
        text_model="mlx-community/multilingual-e5-base-mlx",
        text_batch_size=64,
        clip_mlx_model_dir="./clip_download",
        clip_batch_size=32,
        flush_ms=5,
    )
    await svc.start()

    # Mixed EN + ZH queries
    queries = [
        "How to implement speculative decoding?",
        "给我总结一下这段代码的功能",
        "Find the bug in this Python function",
        "这张图片里有什么？",
    ] * 200  # scale up

    # Benchmark text embeddings
    t0 = time.time()
    _ = await asyncio.gather(*[svc.embed_text(q) for q in queries])
    dt = time.time() - t0
    print(f"[TEXT] {len(queries)} items in {dt:.3f}s => {len(queries)/dt:.1f} items/s")

    # Benchmark CLIP-text embeddings (for image search)
    t0 = time.time()
    _ = await asyncio.gather(*[svc.embed_for_image_search(q) for q in queries])
    dt = time.time() - t0
    print(f"[CLIP-TEXT] {len(queries)} items in {dt:.3f}s => {len(queries)/dt:.1f} items/s")

    print("Batcher stats:", svc.stats())

if __name__ == "__main__":
    asyncio.run(main())