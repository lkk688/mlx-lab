# bench_embed.py
import asyncio
import time
import argparse
import aiohttp
from typing import List

async def fetch_embeddings(session: aiohttp.ClientSession, url: str, texts: List[str], model: str):
    payload = {
        "input": texts,
        "model": model,
        "encoding_format": "float"
    }
    async with session.post(url, json=payload) as response:
        if response.status != 200:
            text = await response.text()
            raise RuntimeError(f"API Error {response.status}: {text}")
        return await response.json()

async def benchmark_api(url: str, model: str, concurrency: int, requests_per_worker: int, batch_size: int):
    print(f"Starting benchmark against {url}")
    print(f"Concurrency: {concurrency}, Requests/worker: {requests_per_worker}, Batch size: {batch_size}")
    
    # Sample queries
    sample_queries = [
        "How to implement speculative decoding?",
        "给我总结一下这段代码的功能",
        "Find the bug in this Python function",
        "这张图片里有什么？",
        "Machine learning on Apple Silicon is extremely fast.",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a SQL query to find the second highest salary."
    ]

    async def worker(worker_id: int, session: aiohttp.ClientSession):
        tokens_processed = 0
        items_processed = 0
        
        for i in range(requests_per_worker):
            # Pick a slice of samples
            start_idx = (worker_id + i) % len(sample_queries)
            batch = [sample_queries[(start_idx + j) % len(sample_queries)] for j in range(batch_size)]
            
            try:
                res = await fetch_embeddings(session, url, batch, model)
                tokens_processed += res["usage"]["total_tokens"]
                items_processed += len(batch)
            except Exception as e:
                print(f"Worker {worker_id} failed: {e}")
        
        return items_processed, tokens_processed

    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [worker(i, session) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)

    total_time = time.time() - t0
    total_items = sum(r[0] for r in results)
    total_tokens = sum(r[1] for r in results)

    print("\n--- Benchmark Results ---")
    print(f"Total time       : {total_time:.3f} s")
    print(f"Total requests   : {total_items} items")
    print(f"Total tokens     : {total_tokens} tokens")
    print(f"Throughput       : {total_items / total_time:.2f} items/s")
    print(f"Token Throughput : {total_tokens / total_time:.2f} tokens/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OpenAI-compatible Embedding API")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/v1/embeddings", help="API Endpoint")
    parser.add_argument("--model", type=str, default="bge-base", help="Model name to request")
    parser.add_argument("-c", "--concurrency", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("-n", "--requests", type=int, default=20, help="Number of requests per worker")
    parser.add_argument("-b", "--batch-size", type=int, default=4, help="Number of texts per request")
    
    args = parser.parse_args()
    asyncio.run(benchmark_api(args.url, args.model, args.concurrency, args.requests, args.batch_size))