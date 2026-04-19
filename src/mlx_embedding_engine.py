# mlx_embedding_engine.py

import asyncio
import time
import numpy as np
import mlx.core as mx
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# Requires: pip install mlx-embeddings
from mlx_embeddings import EmbeddingModel

# -----------------------------
# OpenAI API Schemas
# -----------------------------
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "bge-base"
    encoding_format: str = "float"

class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage

# -----------------------------
# Async Batcher
# -----------------------------
class AsyncBatcher:
    """
    Collects concurrent requests for a short window and processes them in one batch.
    Greatly improves throughput for MLX embedding models.
    """
    def __init__(
        self,
        process_batch_fn,
        max_batch_size: int = 64,
        flush_interval_ms: int = 5,
        max_queue: int = 10000,
    ):
        self.process_batch_fn = process_batch_fn
        self.max_batch_size = max_batch_size
        self.flush_interval_ms = flush_interval_ms
        self._q: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._task: Optional[asyncio.Task] = None

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
            item, fut = await self._q.get()
            batch_items = [item]
            batch_futs = [fut]

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

            try:
                # process_batch_fn should return a list of results matching batch_items length
                results = self.process_batch_fn(batch_items)
                for f, r in zip(batch_futs, results):
                    if not f.cancelled():
                        f.set_result(r)
            except Exception as e:
                for f in batch_futs:
                    if not f.cancelled():
                        f.set_exception(e)

# -----------------------------
# Config & Engine
# -----------------------------
POPULAR_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5-small": "intfloat/e5-small-v2",
    "e5-base": "intfloat/e5-base-v2",
    "nomic": "nomic-ai/nomic-embed-text-v1",
}

class MLXEmbeddingEngine:
    def __init__(self, model_name: str = "bge-base", batch_size: int = 64, use_fp16: bool = True):
        self.model_name = model_name
        self.batch_size = batch_size
        model_id = POPULAR_MODELS.get(model_name, model_name)
        print(f"[Embedding] Loading {model_id}...")

        self.model = EmbeddingModel(
            model_id,
            dtype=mx.float16 if use_fp16 else mx.float32
        )
        self.dim = self.model.embedding_dim
        print(f"[Embedding] Dimension: {self.dim}, ready.")

        self.batcher = AsyncBatcher(
            process_batch_fn=self._process_batch,
            max_batch_size=batch_size,
            flush_interval_ms=10
        )

    async def start(self):
        await self.batcher.start()

    async def stop(self):
        await self.batcher.stop()

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def _process_batch(self, texts: List[str]) -> List[np.ndarray]:
        # mlx-embeddings takes a list of strings
        embeddings = self.model.embed(texts, batch_size=len(texts))
        
        # Ensure evaluation is complete before numpy conversion
        mx.eval(embeddings)
        embeddings_np = np.array(embeddings)
        embeddings_np = self._normalize(embeddings_np)
        
        return [embeddings_np[i] for i in range(embeddings_np.shape[0])]

    async def embed(self, text: str) -> np.ndarray:
        return await self.batcher.submit(text)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="MLX Embedding Server (OpenAI Compatible)")
engine: Optional[MLXEmbeddingEngine] = None

@app.on_event("startup")
async def startup_event():
    global engine
    # Configure your default model here
    engine = MLXEmbeddingEngine(model_name="bge-base", batch_size=64)
    await engine.start()

@app.on_event("shutdown")
async def shutdown_event():
    global engine
    if engine:
        await engine.stop()

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(req: EmbeddingRequest):
    global engine
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    inputs = req.input if isinstance(req.input, list) else [req.input]
    
    if not inputs:
        raise HTTPException(status_code=400, detail="No input provided")

    # Count tokens approximately
    approx_tokens = sum(len(t.split()) for t in inputs)

    # Process concurrently using the batcher
    tasks = [engine.embed(text) for text in inputs]
    try:
        results = await asyncio.gather(*tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    data = []
    for i, emb in enumerate(results):
        data.append(
            EmbeddingData(
                index=i,
                embedding=emb.tolist(),
            )
        )

    return EmbeddingResponse(
        data=data,
        model=req.model,
        usage=Usage(prompt_tokens=approx_tokens, total_tokens=approx_tokens)
    )

if __name__ == "__main__":
    uvicorn.run("mlx_embedding_engine:app", host="0.0.0.0", port=8000, reload=False)