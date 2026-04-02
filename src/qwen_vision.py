import mlx.core as mx
from mlx_vlm import load, generate

# 1. 配置基础参数
model_path = "mlx-community/Qwen3.5-9B-4bit"
image_paths = ["hybrid_eval_accuracy.png"]  # 注意：Python API 中这里最好传入列表
prompt_text = "Tell me what you see in this image."
max_tokens = 2048  # 调大 token 限制，让模型有充足空间完成 <think> 推理
temperature = 0.3

# 2. 加载模型与处理器 (Processor 负责处理图片特征和文本分词)
print(f"Loading model: {model_path} ...")
model, processor = load(model_path)

# 3. 构造符合 Qwen3.5 原生多模态的对话模板 (Chat Template)
# 这一步非常关键，它会自动帮你把 <|im_start|>, <|vision_start|> 等特殊标记拼好
messages = [
    {
        "role": "user", 
        "content": [
            {"type": "image"}, 
            {"type": "text", "text": prompt_text}
        ]
    }
]
formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# 4. 执行生成并打印测速报告
print("Starting generation...\n")

# 调用 generate 函数
output = generate(
    model=model,
    processor=processor,
    prompt=formatted_prompt,
    image=image_paths,
    max_tokens=max_tokens,
    temperature=temperature,
    verbose=True  # 💡 核心开关：设为 True 即可自动流式输出并打印 Token speed 和内存统计
)

"""
#command line version
python -m mlx_vlm generate \
    --model mlx-community/Qwen3.5-9B-4bit \
    --image hybrid_eval_accuracy.png \
    --prompt "Tell me what you see in this image." \
    --max-tokens 2048 \
    --temp 0.3

python qwen_vision.py

Prompt: 4439 tokens, 140.590 tokens-per-sec
Generation: 2048 tokens, 53.521 tokens-per-sec
Peak memory: 17.156 GB


python -m mlx_vlm.server --host 127.0.0.1 --port 8080

curl http://127.0.0.1:8080/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-anything" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Tell me what you see in this image."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "/Users/kaikailiu/Documents/Coder/hybrid_eval_accuracy.png"
            }
          }
        ]
      }
    ],
    "max_tokens": 1024,
    "temperature": 0.3
  }'
#"url": "data:image/jpeg;base64,<这里是一大串Base64代码>"

curl http://127.0.0.1:8080/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-anything" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is MLX framework in Mac platform?"
          }
        ]
      }
    ],
    "max_tokens": 1024,
    "temperature": 0.3
  }'

"""