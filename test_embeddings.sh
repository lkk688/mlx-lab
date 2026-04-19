#!/bin/bash
set -e

echo "Starting MLX Embedding Server..."
# Run the FastAPI server in the background
python src/mlx_embedding_engine.py &
SERVER_PID=$!

# Wait for the server to start
echo "Waiting for server to initialize..."
sleep 5

echo "-------------------------------------"
echo "Running Basic cURL test (OpenAI format):"
echo "-------------------------------------"
curl -s -X POST http://127.0.0.1:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a simple test.",
    "model": "bge-base"
  }' | head -c 300
echo "..."

echo -e "\n\n-------------------------------------"
echo "Running Benchmark Script:"
echo "-------------------------------------"
# Test the server using the benchmark script
python src/bench_embedding.py --url http://127.0.0.1:8000/v1/embeddings --model bge-base -c 10 -n 10 -b 4

echo "-------------------------------------"
echo "Cleaning up..."
kill $SERVER_PID
echo "Done!"
