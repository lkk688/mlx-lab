import os
import re
import json
import asyncio
import csv
import math
import statistics
import time
import argparse
import sys
import random
import subprocess
import shutil
import importlib.util
import select
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from rich.console import Console

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["LM_EVAL_ALLOW_CODE_EXECUTION"] = "1"

console = Console()

from typing import Tuple, Callable, Awaitable
from collections import Counter

try:
    import tiktoken
except ImportError:
    tiktoken = None

def now_stamp() -> str:
    return time.strftime('%Y-%m-%d_%H%M%S')

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken:
        try:
            enc = tiktoken.get_encoding('cl100k_base')
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text) // 4

def compute_safe_max_tokens(input_tokens: int, model_max_context: int, desired_max_output: int, safety_margin: int=1000, min_output: int=1024) -> int:
    """
    Compute the largest safe max_tokens value that won't exceed the model's context limit.
    
    Args:
        input_tokens: Estimated token count of the input (system + user messages)
        model_max_context: Model's maximum context window
        desired_max_output: The user's requested max output tokens
        safety_margin: Extra buffer for tokenizer estimation errors
        min_output: Minimum output tokens; below this, signal an error condition
    
    Returns:
        Clamped max_tokens value without blindly asserting min_output if budget is tight.
    """
    adjusted_input = int(input_tokens * 1.1)
    available = model_max_context - adjusted_input - safety_margin
    if available < min_output:
        console.print(f'[red]Context budget very tight: {available} tokens available (est_input={input_tokens} -> {adjusted_input}, limit={model_max_context}). Returning available tokens to avoid exceeding context window.[/red]')
        return max(1, available)
    safe = min(desired_max_output, available)
    return safe

def compress_messages(messages: List[Dict[str, str]], max_allowed_tokens: int) -> List[Dict[str, str]]:
    """
    Compress the messages list to fit within max_allowed_tokens by truncating 
    the longest text blocks in 'user' or 'assistant' messages.
    """
    import copy
    msgs = copy.deepcopy(messages)
    while True:
        current_tokens = sum((estimate_tokens(m.get('content', '')) for m in msgs))
        if current_tokens <= max_allowed_tokens:
            break
        longest_idx = -1
        longest_len = 0
        for i, m in enumerate(msgs):
            if i in (0, 1):
                continue
            content_len = len(m.get('content', ''))
            if content_len > longest_len:
                longest_len = content_len
                longest_idx = i
        if longest_idx == -1:
            for i, m in enumerate(msgs):
                if i == 0:
                    continue
                content_len = len(m.get('content', ''))
                if content_len > longest_len:
                    longest_len = content_len
                    longest_idx = i
        if longest_idx == -1 or longest_len < 400:
            break
        content = msgs[longest_idx]['content']
        keep_chars = int(longest_len * 0.45)
        if keep_chars * 2 + 35 >= longest_len:
            break
        msgs[longest_idx]['content'] = content[:keep_chars] + '\n...[TRUNCATED TO FIT CONTEXT]...\n' + content[-keep_chars:]
    return msgs

def _detect_repetition(text: str, window_size: int=50, threshold: int=4) -> bool:
    """
    Advanced Repetition Detector.
    Catches exact looping phrases AND structural loops (like incrementing numbers in the same sentence).
    """
    if len(text) < 500:
        return False
    tail_lines = text[-500:].strip().split('\n')
    if len(tail_lines) >= 4:
        if len(set(tail_lines[-4:])) == 1 and len(tail_lines[-1]) > 5:
            return True
    normalized_text = re.sub('\\d+', '<NUM>', text)
    tokens = normalized_text.split()
    if len(tokens) < window_size * 2:
        return False
    n = 15
    if len(tokens) < n:
        return False
    ngrams = [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n)]
    ngram_counts = Counter(ngrams)
    most_common = ngram_counts.most_common(1)
    if most_common and most_common[0][1] > 5:
        print(f'\n[bold red]⚠️ Repetition Circuit Breaker Fused![/bold red]')
        print(f"[dim]Detected repeating pattern: '{most_common[0][0]}' ({most_common[0][1]} times)[/dim]")
        return True
    return False

def compute_stream_speed_metrics(prompt_tokens: int, completion_tokens: int, elapsed_seconds: float, ttft_seconds: Optional[float]=None) -> Dict[str, float]:
    ttft = float(ttft_seconds) if ttft_seconds is not None else float(elapsed_seconds)
    ttft = max(0.0, ttft)
    elapsed = max(0.0, float(elapsed_seconds))
    decode_latency = max(0.0, elapsed - ttft)
    e2e_tps = float(completion_tokens) / elapsed if elapsed > 0 else 0.0
    decode_tps = float(completion_tokens) / decode_latency if decode_latency > 0 and completion_tokens > 0 else 0.0
    prefill_tps = float(prompt_tokens) / ttft if ttft > 0 and prompt_tokens > 0 else 0.0
    per_token_decode_latency_ms = decode_latency * 1000.0 / float(completion_tokens) if completion_tokens > 0 and decode_latency > 0 else 0.0
    return {'ttft_seconds': ttft, 'decode_latency_seconds': decode_latency, 'e2e_tokens_per_second': e2e_tps, 'decode_tokens_per_second': decode_tps, 'prefill_tokens_per_second': prefill_tps, 'per_token_decode_latency_ms': per_token_decode_latency_ms}

async def _execute_openai_async(client: Any, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, stream: bool, tools: Optional[List[Dict[str, Any]]]=None, verbose: bool=False, on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]]=None, backend: str='openai', enable_thinking: bool=True) -> Tuple[str, str, Dict[str, int], List[Dict[str, Any]]]:
    kwargs = {'model': model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens, 'stream': stream}
    if backend in ['llama.cpp', 'vllm'] and enable_thinking is not None:
        kwargs['extra_body'] = {'chat_template_kwargs': {'enable_thinking': enable_thinking}}
        kwargs['stop'] = ['<|im_end|>', '<|im_start|>', '<|endoftext|>']
    if tools:
        kwargs['tools'] = tools
    if stream:
        kwargs['stream_options'] = {'include_usage': True}
    resp = await client.chat.completions.create(**kwargs)
    content = ''
    finish_reason = 'stop'
    usage_info = {}
    native_tool_calls = []
    tc_dict = {}
    if stream:
        chunk_counter = 0
        in_think = False
        in_tool = False
        buffer = ''
        tool_args_buffer = ''
        tool_name_buffer = ''
        async for chunk in resp:
            chunk_counter += 1
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = {'prompt_tokens': chunk.usage.prompt_tokens, 'completion_tokens': chunk.usage.completion_tokens}
                continue
            delta = chunk.choices[0].delta
            reasoning = delta.model_dump().get('reasoning_content')
            if reasoning:
                if verbose:
                    console.print(reasoning, end='', style='dim', highlight=False, markup=False)
                if on_event:
                    await on_event({'type': 'think', 'data': reasoning})
            if delta.content:
                text_chunk = delta.content
                content += text_chunk
                buffer += text_chunk
                if not in_think and '<think>' in buffer:
                    in_think = True
                    buffer = buffer.split('<think>')[-1]
                if in_think and '</think>' in buffer:
                    in_think = False
                    buffer = buffer.split('</think>')[-1]
                if not in_tool and '<tool_call>' in buffer:
                    in_tool = True
                    buffer = buffer.split('<tool_call>')[-1]
                if in_tool and '</tool_call>' in buffer:
                    in_tool = False
                    buffer = ''
                    tool_args_buffer = ''
                    tool_name_buffer = ''
                clean_chunk = re.sub('</?think>|</?tool_call>', '', text_chunk)
                if in_think:
                    if verbose:
                        console.print(clean_chunk, end='', style='dim', highlight=False, markup=False)
                    if on_event and clean_chunk:
                        await on_event({'type': 'think', 'data': clean_chunk})
                elif in_tool:
                    tool_args_buffer += text_chunk
                    if not tool_name_buffer and '>' in tool_args_buffer:
                        match = re.search('<([a-zA-Z0-9_]+)>', tool_args_buffer)
                        if match:
                            tool_name_buffer = match.group(1)
                            if verbose:
                                console.print(f'\n[bold magenta]🛠️ Parsing Tool: {tool_name_buffer}...[/bold magenta]')
                    if on_event:
                        await on_event({'type': 'tool', 'status': 'streaming', 'data': text_chunk})
                elif clean_chunk:
                    if verbose:
                        console.print(clean_chunk, end='', highlight=False, markup=False)
                    if on_event:
                        await on_event({'type': 'message', 'data': clean_chunk})
                if chunk_counter % 40 == 0 and len(content) > 500:
                    if _detect_repetition(content):
                        if verbose:
                            print('\n\n[bold red]⚠️ [Stream Interrupted] Repetition Loop Detected! Connection severed.[/bold red]')
                        finish_reason = 'repetition'
                        break
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_dict:
                        func_name = tc.function.name if tc.function and tc.function.name else 'unknown_tool'
                        tc_dict[idx] = {'name': func_name, 'arguments': ''}
                        if on_event:
                            await on_event({'type': 'tool', 'name': func_name, 'status': 'started'})
                        if verbose:
                            console.print(f'\n[bold magenta]🛠️ Calling Tool: {func_name}...[/bold magenta]')
                    if tc.function and tc.function.arguments:
                        chunk_arg = tc.function.arguments
                        tc_dict[idx]['arguments'] += chunk_arg
                        if on_event:
                            await on_event({'type': 'tool', 'name': tc_dict[idx]['name'], 'args_delta': chunk_arg, 'status': 'streaming'})
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        if verbose and content and (finish_reason != 'repetition'):
            print()
        native_tool_calls = list(tc_dict.values())
    else:
        msg = resp.choices[0].message
        content = msg.content or ''
        finish_reason = resp.choices[0].finish_reason or 'stop'
        if hasattr(resp, 'usage') and resp.usage:
            usage_info = {'prompt_tokens': resp.usage.prompt_tokens, 'completion_tokens': resp.usage.completion_tokens}
        if msg.tool_calls:
            for tc in msg.tool_calls:
                native_tool_calls.append({'name': tc.function.name, 'arguments': tc.function.arguments})
    return (content, finish_reason, usage_info, native_tool_calls)

async def _execute_anthropic_async(client: Any, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, tools: Optional[List[Dict[str, Any]]]=None, verbose: bool=False) -> Tuple[str, str, Dict[str, int], List[Dict[str, Any]]]:
    """
    Handles Anthropic specific API formatting with native tool calls.
    Utilizes Ephemeral Caching for massive cost reduction on large contexts.
    """
    sys_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
    usr_msgs = []
    for m in messages:
        if m['role'] != 'system':
            role = 'assistant' if m['role'] == 'assistant' else 'user'
            usr_msgs.append({'role': role, 'content': [{'type': 'text', 'text': m['content']}]})
    if usr_msgs and usr_msgs[-1]['role'] == 'user':
        usr_msgs[-1]['content'][-1]['cache_control'] = {'type': 'ephemeral'}
    sys_msg_blocks = [{'type': 'text', 'text': sys_msg, 'cache_control': {'type': 'ephemeral'}}]
    config_kwargs = {'model': model, 'system': sys_msg_blocks, 'messages': usr_msgs, 'temperature': temperature, 'max_tokens': max_tokens}
    if tools:
        config_kwargs['tools'] = tools
    resp = await client.messages.create(**config_kwargs)
    content = ''
    native_tool_calls = []
    for block in resp.content:
        if block.type == 'text':
            content += block.text
        elif block.type == 'tool_use':
            native_tool_calls.append({'name': block.name, 'arguments': json.dumps(block.input)})
    if resp.stop_reason == 'max_tokens':
        finish_reason = 'length'
    elif resp.stop_reason == 'tool_use':
        finish_reason = 'tool_calls'
    else:
        finish_reason = 'stop'
    usage_info = {'prompt_tokens': getattr(resp.usage, 'input_tokens', 0), 'completion_tokens': getattr(resp.usage, 'output_tokens', 0)} if hasattr(resp, 'usage') else {}
    if verbose and content:
        print(content)
    return (content, finish_reason, usage_info, native_tool_calls)

async def complete_with_async(client: Any, model: str, messages: List[Dict[str, str]], temperature: float=0.2, max_output_tokens: int=4096, model_max_context: int=16384, provider: str='openai', stream: bool=True, verbose: bool=False, on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]]=None, backend: str='openai', enable_thinking: bool=True) -> Tuple[str, Dict[str, Any]]:
    """
    Simple single-shot async wrapper for LLM API calls.

    Unlike complete_with_continuation_async, this function:
    - Does NOT loop on finish_reason == 'length' (no auto-continuation)
    - Does NOT parse tool calls or agent actions
    - Returns (content: str, usage_info: dict)

    The optional `on_token` async callback receives each streamed token as it
    arrives, enabling FastAPI SSE endpoints to forward tokens to clients in
    real-time without any extra buffering.

    Args:
        client:            Async OpenAI-compatible client (or Anthropic client)
        model:             Model name
        messages:          List of chat messages (role/content dicts)
        temperature:       Sampling temperature
        max_output_tokens: Maximum completion tokens to request
        model_max_context: Total context window size (for token budget calc)
        provider:          'openai' or 'anthropic'
        stream:            Enable streaming API (token-by-token)
        verbose:           Print tokens to terminal as they arrive
        on_token:          Optional async callback called for each streamed token.
                           Signature: async def on_token(token: str) -> None

    Returns:
        (content, usage_info)  where usage_info has 'prompt_tokens',
        'completion_tokens', 'elapsed_seconds', 'tokens_per_second',
        and 'finish_reason' keys.
    """
    input_text = '\n'.join((m.get('content', '') for m in messages))
    input_est = estimate_tokens(input_text)
    min_output = 256
    max_allowed_input = model_max_context - 1000 - min_output
    if int(input_est * 1.1) > max_allowed_input > 0:
        console.print(f'[yellow]Compressing messages (est {input_est} > limit).[/yellow]')
        messages = compress_messages(messages, max_allowed_tokens=int(max_allowed_input / 1.1))
        input_est = estimate_tokens('\n'.join((m.get('content', '') for m in messages)))
    safe_tokens = compute_safe_max_tokens(input_est, model_max_context, max_output_tokens, min_output)
    content: str = ''
    finish_reason: str = 'stop'
    usage_info: Dict[str, Any] = {}
    start_time = time.time()
    for attempt in range(3):
        try:
            if provider == 'anthropic':
                content, finish_reason, usage_info, _ = await _execute_anthropic_async(client, model, messages, temperature, safe_tokens, tools=None, verbose=verbose)
            else:
                content, finish_reason, usage_info, _ = await _execute_openai_async(client, model, messages, temperature, safe_tokens, stream=stream, tools=None, verbose=verbose, on_event=on_event, backend=backend, enable_thinking=enable_thinking)
            break
        except Exception as e:
            err_str = str(e)
            if 'max_tokens' in err_str or 'context length' in err_str:
                safe_tokens = max(1024, safe_tokens // 2)
                console.print(f'[red]Context overflow. Retrying max_tokens={safe_tokens}[/red]')
                await asyncio.sleep(1)
                continue
            
            console.print(f'[red]LLM Call failed (attempt {attempt + 1}): {e}[/red]')
            
            # Immediately abort on fatal server/engine errors to prevent endless retry loops 
            # and to allow the evaluator to fail fast.
            if "broadcast_shapes" in err_str or "streaming" in err_str or "Internal server error" in err_str:
                console.print('[bold red]Fatal server error detected (likely KV cache exhaustion or OOM). Aborting retries for this batch.[/bold red]')
                finish_reason = "error"
                break
                
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            break
    elapsed = time.time() - start_time
    if not usage_info:
        usage_info = {'prompt_tokens': input_est, 'completion_tokens': estimate_tokens(content)}
    speed_metrics = compute_stream_speed_metrics(prompt_tokens=int(usage_info.get('prompt_tokens', input_est) or 0), completion_tokens=int(usage_info.get('completion_tokens', estimate_tokens(content)) or 0), elapsed_seconds=elapsed)
    console.print(f"[bold blue][LLM][/bold blue] [dim]{usage_info['prompt_tokens']}P, {usage_info['completion_tokens']}C | {speed_metrics['e2e_tokens_per_second']:.1f} T/s | {elapsed:.1f}s | finish={finish_reason}[/dim]")
    usage_info['elapsed_seconds'] = round(elapsed, 2)
    usage_info['tokens_per_second'] = round(speed_metrics['e2e_tokens_per_second'], 1)
    usage_info['prefill_tokens_per_second'] = round(speed_metrics['prefill_tokens_per_second'], 3)
    usage_info['decode_tokens_per_second'] = round(speed_metrics['decode_tokens_per_second'], 3)
    usage_info['per_token_decode_latency_ms'] = round(speed_metrics['per_token_decode_latency_ms'], 3)
    usage_info['finish_reason'] = finish_reason
    return (content, usage_info)


try:
    from openai import AsyncOpenAI as _AsyncOpenAI
except ImportError:
    _AsyncOpenAI = None

LLM_REQUEST_ERRORS = (
    httpx.HTTPError,
    asyncio.TimeoutError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
)
SUBPROCESS_ERRORS = (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError)


@dataclass
class APICase:
    name: str
    base_url: str
    model: str
    api_key: str
    backend: str
    provider: str = "openai"
    enable_thinking: bool = False
    tokenizer: str = ""


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _require_async_openai() -> Any:
    if _AsyncOpenAI is None:
        raise ModuleNotFoundError("openai package is required for LLM evaluation. Install with: pip install openai")
    return _AsyncOpenAI


def _parse_csv_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in _parse_csv_list(raw):
        try:
            value = int(item)
            if value > 0:
                values.append(value)
        except ValueError:
            continue
    return values


def _parse_case_entry(raw: str, defaults: Dict[str, str]) -> APICase:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    kv: Dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        kv[key.strip().lower()] = value.strip()
    name = kv.get("name") or kv.get("label")
    base_url = kv.get("url") or kv.get("base_url")
    if not name or not base_url:
        raise ValueError(f"Invalid --llm-case '{raw}'. Required keys: name,url")
    return APICase(
        name=name,
        base_url=base_url,
        model=kv.get("model", defaults["model"]),
        api_key=kv.get("api_key", defaults["api_key"]),
        backend=kv.get("backend", defaults["backend"]),
        provider=kv.get("provider", defaults["provider"]),
        enable_thinking=kv.get("enable_thinking", "false").lower() in {"1", "true", "yes", "y"},
        tokenizer=kv.get("tokenizer", defaults.get("tokenizer", "")),
    )


def _build_llm_cases(args: argparse.Namespace) -> List[APICase]:
    return [
        APICase(
            name=args.base_url_name,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            backend=args.backend,
            provider=args.provider,
            enable_thinking=args.enable_thinking,
            tokenizer=args.tokenizer,
        )
    ]


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _build_prefill_text(target_tokens: int) -> str:
    seed = (
        "Autoregressive decoding benchmark context. "
        "Measure prompt ingestion speed, decode speed, and latency distributions. "
        "Keep semantic coherence while expanding token count for prefill stress testing."
    )
    chunks: List[str] = []
    while estimate_tokens("\n".join(chunks)) < target_tokens:
        chunks.append(seed)
    return "\n".join(chunks)


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _summarize_speed_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_runs = len(rows)
    ok = [r for r in rows if r.get("success")]
    if not ok:
        return {
            "runs": total_runs,
            "successful_runs": 0,
            "success_rate": 0.0,
            "ttft_p50_sec": 0.0,
            "ttft_p95_sec": 0.0,
            "ttft_p99_sec": 0.0,
            "e2e_p50_sec": 0.0,
            "e2e_p95_sec": 0.0,
            "e2e_p99_sec": 0.0,
            "e2e_speed_p50_tokens_per_sec": 0.0,
            "decode_p50_tokens_per_sec": 0.0,
            "decode_p95_tokens_per_sec": 0.0,
            "decode_p99_tokens_per_sec": 0.0,
            "prefill_mean_tokens_per_sec": 0.0,
            "per_token_decode_latency_p50_ms": 0.0,
            "per_token_decode_latency_p95_ms": 0.0,
            "per_token_decode_latency_p99_ms": 0.0,
            "throughput_tokens_per_sec": 0.0,
        }
    ttft_vals = [float(r["ttft_sec"]) for r in ok]
    e2e_vals = [float(r["e2e_latency_sec"]) for r in ok]
    e2e_tps_vals = [float(r["e2e_tokens_per_sec"]) for r in ok]
    decode_tps_vals = [float(r["decode_tokens_per_sec"]) for r in ok]
    prefill_tps_vals = [float(r["prefill_tokens_per_sec"]) for r in ok]
    per_token_ms_vals = [float(r["per_token_decode_latency_ms"]) for r in ok if r.get("per_token_decode_latency_ms") is not None]
    total_completion_tokens = sum(int(r["completion_tokens"]) for r in ok)
    total_elapsed = sum(float(r["e2e_latency_sec"]) for r in ok)
    return {
        "runs": total_runs,
        "successful_runs": len(ok),
        "success_rate": round(len(ok) / total_runs, 4) if total_runs else 0.0,
        "ttft_p50_sec": round(_percentile(ttft_vals, 50), 4),
        "ttft_p95_sec": round(_percentile(ttft_vals, 95), 4),
        "ttft_p99_sec": round(_percentile(ttft_vals, 99), 4),
        "e2e_p50_sec": round(_percentile(e2e_vals, 50), 4),
        "e2e_p95_sec": round(_percentile(e2e_vals, 95), 4),
        "e2e_p99_sec": round(_percentile(e2e_vals, 99), 4),
        "e2e_speed_p50_tokens_per_sec": round(_percentile(e2e_tps_vals, 50), 3),
        "decode_p50_tokens_per_sec": round(_percentile(decode_tps_vals, 50), 3),
        "decode_p95_tokens_per_sec": round(_percentile(decode_tps_vals, 95), 3),
        "decode_p99_tokens_per_sec": round(_percentile(decode_tps_vals, 99), 3),
        "prefill_mean_tokens_per_sec": round(statistics.fmean(prefill_tps_vals), 3),
        "per_token_decode_latency_p50_ms": round(_percentile(per_token_ms_vals, 50), 3) if per_token_ms_vals else 0.0,
        "per_token_decode_latency_p95_ms": round(_percentile(per_token_ms_vals, 95), 3) if per_token_ms_vals else 0.0,
        "per_token_decode_latency_p99_ms": round(_percentile(per_token_ms_vals, 99), 3) if per_token_ms_vals else 0.0,
        "throughput_tokens_per_sec": round((total_completion_tokens / total_elapsed) if total_elapsed > 0 else 0.0, 3),
    }


def _build_speed_report_html(summary_by_case: Dict[str, Any], summary_by_prefill: Dict[str, Any], raw_rows: List[Dict[str, Any]]) -> str:
    summary_json = json.dumps(summary_by_case)
    summary_prefill_json = json.dumps(summary_by_prefill)
    raw_rows_json = json.dumps(raw_rows)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM Speed Benchmark Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: #0b1020;
      color: #e5e7eb;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1, h2 {{
      margin: 8px 0 12px 0;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 12px;
    }}
    .full {{
      grid-column: 1 / -1;
    }}
    .meta {{
      color: #9ca3af;
      font-size: 13px;
      margin-bottom: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #1f2937;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>LLM Inference Speed Benchmark</h1>
    <div class="meta">Metrics: TTFT, End-to-End Latency, E2E Speed, Decode Speed, Prefill Speed, Per-token Decode Latency, P95/P99, Throughput</div>
    <div class="grid">
      <div class="card"><div id="ttft_chart"></div></div>
      <div class="card"><div id="e2e_chart"></div></div>
      <div class="card"><div id="decode_chart"></div></div>
      <div class="card"><div id="prefill_chart"></div></div>
      <div class="card"><div id="throughput_chart"></div></div>
      <div class="card"><div id="e2e_speed_chart"></div></div>
      <div class="card full"><div id="prefill_curve"></div></div>
      <div class="card full">
        <h2>Case Summary</h2>
        <table id="summary_table"></table>
      </div>
    </div>
  </div>
  <script>
    const summaryByCase = {summary_json};
    const summaryByPrefill = {summary_prefill_json};
    const rawRows = {raw_rows_json};
    const labels = Object.keys(summaryByCase);
    const ttftP95 = labels.map(k => summaryByCase[k].ttft_p95_sec);
    const e2eP95 = labels.map(k => summaryByCase[k].e2e_p95_sec);
    const decodeP50 = labels.map(k => summaryByCase[k].decode_p50_tokens_per_sec);
    const prefillMean = labels.map(k => summaryByCase[k].prefill_mean_tokens_per_sec);
    const throughput = labels.map(k => summaryByCase[k].throughput_tokens_per_sec);
    const e2eSpeed = labels.map(k => summaryByCase[k].e2e_speed_p50_tokens_per_sec);
    const darkLayout = {{
      paper_bgcolor: '#111827',
      plot_bgcolor: '#111827',
      font: {{color: '#e5e7eb'}},
      margin: {{l: 60, r: 20, t: 40, b: 60}},
    }};
    Plotly.newPlot('ttft_chart', [{{ x: labels, y: ttftP95, type: 'bar', marker: {{color: '#60a5fa'}} }}], {{...darkLayout, title: 'TTFT P95', yaxis: {{title: 'Seconds'}}}});
    Plotly.newPlot('e2e_chart', [{{ x: labels, y: e2eP95, type: 'bar', marker: {{color: '#f97316'}} }}], {{...darkLayout, title: 'End-to-End Latency P95', yaxis: {{title: 'Seconds'}}}});
    Plotly.newPlot('decode_chart', [{{ x: labels, y: decodeP50, type: 'bar', marker: {{color: '#34d399'}} }}], {{...darkLayout, title: 'Decode Speed P50', yaxis: {{title: 'Tokens/s'}}}});
    Plotly.newPlot('prefill_chart', [{{ x: labels, y: prefillMean, type: 'bar', marker: {{color: '#22d3ee'}} }}], {{...darkLayout, title: 'Prefill Speed Mean', yaxis: {{title: 'Tokens/s'}}}});
    Plotly.newPlot('throughput_chart', [{{ x: labels, y: throughput, type: 'bar', marker: {{color: '#a78bfa'}} }}], {{...darkLayout, title: 'Throughput', yaxis: {{title: 'Tokens/s'}}}});
    Plotly.newPlot('e2e_speed_chart', [{{ x: labels, y: e2eSpeed, type: 'bar', marker: {{color: '#facc15'}} }}], {{...darkLayout, title: 'End-to-End Speed P50', yaxis: {{title: 'Tokens/s'}}}});
    const grouped = {{}};
    Object.entries(summaryByPrefill).forEach(([k, v]) => {{
      const sep = k.lastIndexOf('::');
      if (sep < 0) return;
      const name = k.slice(0, sep);
      const prefill = Number(k.slice(sep + 2));
      if (!grouped[name]) grouped[name] = [];
      grouped[name].push({{prefill, ttft: v.ttft_p50_sec, decode: v.decode_p50_tokens_per_sec}});
    }});
    const traces = Object.entries(grouped).map(([name, arr]) => {{
      arr.sort((a, b) => a.prefill - b.prefill);
      return {{ x: arr.map(x => x.prefill), y: arr.map(x => x.ttft), mode: 'lines+markers', name: name + ' TTFT P50' }};
    }});
    Plotly.newPlot('prefill_curve', traces, {{...darkLayout, title: 'TTFT vs Prefill Tokens', xaxis: {{title: 'Prefill Tokens'}}, yaxis: {{title: 'TTFT P50 (s)'}}}});
    const table = document.getElementById('summary_table');
    const headers = ['Case', 'Success', 'TTFT P95', 'E2E P95', 'E2E Speed P50', 'Prefill Mean', 'Decode P50', 'Per-token P95 ms', 'Throughput'];
    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    headers.forEach(h => {{ const th = document.createElement('th'); th.textContent = h; trh.appendChild(th); }});
    thead.appendChild(trh);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    labels.forEach(k => {{
      const s = summaryByCase[k];
      const tr = document.createElement('tr');
      [k, `${{s.successful_runs}}/${{s.runs}}`, s.ttft_p95_sec, s.e2e_p95_sec, s.e2e_speed_p50_tokens_per_sec, s.prefill_mean_tokens_per_sec, s.decode_p50_tokens_per_sec, s.per_token_decode_latency_p95_ms, s.throughput_tokens_per_sec].forEach((v, idx) => {{
        const td = document.createElement('td');
        td.textContent = String(v);
        if (idx === 0) td.style.textAlign = 'left';
        tr.appendChild(td);
      }});
      tbody.appendChild(tr);
    }});
    table.appendChild(tbody);
  </script>
</body>
</html>"""


async def _run_single_speed_request(client: Any, case: APICase, messages: List[Dict[str, str]], max_output_tokens: int, model_max_context: int) -> Dict[str, Any]:
    first_token_ts: Optional[float] = None
    started = time.perf_counter()
    
    async def on_event(evt: Dict[str, Any]) -> None:
        nonlocal first_token_ts
        evt_type = str(evt.get("type", ""))
        if evt_type not in {"message", "think"}:
            return
        if not str(evt.get("data", "")):
            return
        if first_token_ts is None:
            first_token_ts = time.perf_counter()

    error_text = ""
    content = ""
    usage_info: Dict[str, Any] = {}
    try:
        content, usage_info = await complete_with_async(
            client=client,
            model=case.model,
            messages=messages,
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            model_max_context=model_max_context,
            provider=case.provider,
            stream=True,
            verbose=False,
            on_event=on_event,
            backend=case.backend,
            enable_thinking=case.enable_thinking,
        )
    except Exception as e:
        error_text = str(e)
        
    elapsed = time.perf_counter() - started
    prompt_tokens = int(usage_info.get("prompt_tokens", estimate_tokens("\n".join(m["content"] for m in messages))))
    completion_tokens = int(usage_info.get("completion_tokens", estimate_tokens(content)))
    ttft_sec = (first_token_ts - started) if first_token_ts is not None else elapsed
    
    speed_metrics = compute_stream_speed_metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        elapsed_seconds=elapsed,
        ttft_seconds=ttft_sec,
    )
    
    success = (error_text == "") and (completion_tokens > 0 or len(content.strip()) > 0)
    
    return {
        "success": success,
        "error": error_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_sec": speed_metrics["ttft_seconds"],
        "e2e_latency_sec": elapsed,
        "per_token_decode_latency_ms": speed_metrics["per_token_decode_latency_ms"],
        "usage_info": usage_info
    }


async def run_llm_speed_evaluation(args: argparse.Namespace, report_dir: Path, case: APICase) -> Dict[str, Any]:
    batch_sizes = _parse_csv_int_list(args.batch_sizes)
    prefill_targets = _parse_csv_int_list(args.prefill_tokens)
    max_output_tokens = args.max_output_tokens
    
    if not batch_sizes:
        batch_sizes = [1, 2, 4]
    if not prefill_targets:
        prefill_targets = [1024, 4096]
        
    speed_dir = report_dir / "speed"
    speed_dir.mkdir(parents=True, exist_ok=True)
    raw_json_path = speed_dir / "raw_runs.json"
    
    rows: List[Dict[str, Any]] = []
    
    console.print(f"[bold cyan]Speed benchmark case:[/bold cyan] {case.name} ({case.base_url})")
    http_client = httpx.AsyncClient(timeout=float(args.timeout_seconds))
    async_openai_cls = _require_async_openai()
    client = async_openai_cls(base_url=case.base_url, api_key=case.api_key, http_client=http_client)
    
    try:
        for batch_size in batch_sizes:
            for prefill_target in prefill_targets:
                for prompt_type in ["same", "different"]:
                    console.print(f"Running batch_size={batch_size}, prefill={prefill_target}, prompt={prompt_type}...")
                    
                    tasks = []
                    base_context = _build_prefill_text(prefill_target)
                    
                    for i in range(batch_size):
                        if prompt_type == "different":
                            # Make the prompt slightly different but same length
                            prefix = f"Ignore this salt {i}: {random.randint(100000, 999999)}. "
                            adjusted_context = prefix + base_context[len(prefix):] if len(base_context) > len(prefix) else base_context + prefix
                        else:
                            adjusted_context = base_context
                            
                        messages = [
                            {"role": "system", "content": "You are a precise and concise assistant."},
                            {"role": "user", "content": f"{adjusted_context}\n\nTask: Summarize the technical context in five concise bullets. Do not use markdown tables."},
                        ]
                        
                        tasks.append(_run_single_speed_request(client, case, messages, max_output_tokens, args.model_max_context))
                        
                    batch_started = time.perf_counter()
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    batch_elapsed = time.perf_counter() - batch_started
                    
                    valid_results = [r for r in results if isinstance(r, dict) and r.get("success")]
                    
                    if not valid_results:
                        console.print(f"[red]Batch completely failed![/red]")
                        if any(isinstance(r, Exception) for r in results):
                            console.print(f"[red]Exception: {next(r for r in results if isinstance(r, Exception))}[/red]")
                        continue
                        
                    # Calculate metrics
                    ttft_vals = [r["ttft_sec"] for r in valid_results if r["ttft_sec"] > 0]
                    tpot_vals = [r["per_token_decode_latency_ms"] for r in valid_results if r["per_token_decode_latency_ms"] > 0]
                    
                    mean_ttft_ms = (sum(ttft_vals) / len(ttft_vals) * 1000) if ttft_vals else 0
                    mean_tpot_ms = (sum(tpot_vals) / len(tpot_vals)) if tpot_vals else 0
                    
                    total_prompt_tokens = sum(r["prompt_tokens"] for r in valid_results)
                    total_completion_tokens = sum(r["completion_tokens"] for r in valid_results)
                    
                    max_ttft_sec = max(ttft_vals) if ttft_vals else 0.001
                    max_e2e_sec = max([r["e2e_latency_sec"] for r in valid_results]) if valid_results else 0.001
                    
                    pp_tps = total_prompt_tokens / max_ttft_sec
                    # Prevent division by zero if all tokens come in the first chunk
                    decode_time_sec = (max_e2e_sec - max_ttft_sec) if (max_e2e_sec > max_ttft_sec) else 0.001
                    tg_tps = total_completion_tokens / decode_time_sec
                    
                    throughput = total_completion_tokens / max_e2e_sec
                    
                    # Try to get peak mem from the first valid usage info if available
                    peak_mem = "N/A"
                    for r in valid_results:
                        usage = r.get("usage_info", {})
                        if isinstance(usage, dict) and "peak_mem" in usage:
                            peak_mem_val = usage["peak_mem"]
                            if peak_mem_val:
                                peak_mem = str(peak_mem_val)
                                break
                    
                    row = {
                        "case_name": case.name,
                        "batch_size": batch_size,
                        "prefill_target": prefill_target,
                        "prompt_type": prompt_type,
                        "success_count": len(valid_results),
                        "total_requests": batch_size,
                        "mean_ttft_ms": round(mean_ttft_ms, 2),
                        "mean_tpot_ms": round(mean_tpot_ms, 2),
                        "pp_tps": round(pp_tps, 2),
                        "tg_tps": round(tg_tps, 2),
                        "e2e_latency_sec": round(max_e2e_sec, 3),
                        "throughput": round(throughput, 2),
                        "peak_mem": peak_mem,
                    }
                    rows.append(row)
                    
                    console.print(f"[dim]Batch {batch_size}x{prefill_target} ({prompt_type}): TTFT {mean_ttft_ms:.1f}ms, TPOT {mean_tpot_ms:.1f}ms, pp TPS {pp_tps:.1f}, tg TPS {tg_tps:.1f}, Peak Mem: {peak_mem}[/dim]")
    finally:
        await http_client.aclose()
        
    raw_json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return {
        "suite": "speed",
        "case_name": case.name,
        "records": rows
    }


def _extract_digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


def _build_passkey_context(target_tokens: int, passkey: int, depth: float) -> str:
    filler_sentence = "The sun sets in the west. "
    filler = []
    while estimate_tokens("".join(filler)) < max(64, target_tokens):
        filler.append(filler_sentence)
    filler_text = "".join(filler)
    needle = f"\nThe secret passkey is {passkey}.\n"
    insert_at = int(len(filler_text) * max(0.0, min(depth, 1.0)))
    return filler_text[:insert_at] + needle + filler_text[insert_at:]


async def _run_passkey_for_case(args: argparse.Namespace, case: APICase) -> Dict[str, Any]:
    hits = 0
    trials: List[Dict[str, Any]] = []
    http_client = httpx.AsyncClient(timeout=float(args.timeout_seconds))
    async_openai_cls = _require_async_openai()
    client = async_openai_cls(base_url=case.base_url, api_key=case.api_key, http_client=http_client)
    try:
        for idx in range(args.passkey_trials):
            passkey = random.randint(10000, 99999)
            context_text = _build_passkey_context(args.passkey_ctx, passkey, args.passkey_depth)
            prompt = f"{context_text}\nWhat is the secret passkey? Answer with ONLY the number."
            content = ""
            error_text = ""
            try:
                content, _ = await complete_with_async(
                    client=client,
                    model=case.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_output_tokens=32,
                    model_max_context=args.model_max_context,
                    provider=case.provider,
                    stream=False,
                    verbose=False,
                    on_event=None,
                    backend=case.backend,
                    enable_thinking=case.enable_thinking,
                )
            except LLM_REQUEST_ERRORS as e:
                error_text = str(e)
            predicted_digits = _extract_digits(content)
            hit = str(passkey) in predicted_digits and error_text == ""
            if hit:
                hits += 1
            trials.append(
                {
                    "case_name": case.name,
                    "trial": idx + 1,
                    "expected_passkey": passkey,
                    "prediction": content.strip(),
                    "prediction_digits": predicted_digits,
                    "hit": hit,
                    "error": error_text,
                }
            )
    finally:
        await http_client.aclose()
    accuracy = (100.0 * hits / args.passkey_trials) if args.passkey_trials > 0 else 0.0
    return {"benchmark": "passkey", "case_name": case.name, "trials": trials, "passkey_accuracy": round(accuracy, 4)}


def _which_or_none(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run_subprocess(cmd: List[str], env: Dict[str, str], log_path: Path, timeout_sec: int) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_chunks: List[str] = []
    start_time = time.monotonic()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    if process.stdout is None:
        raise OSError("Failed to capture subprocess output")
    fd = process.stdout.fileno()
    with log_path.open("w", encoding="utf-8") as log_file:
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_sec:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(cmd, timeout_sec, output="".join(output_chunks))
            ready, _, _ = select.select([fd], [], [], 1.0)
            if ready:
                raw_chunk = os.read(fd, 4096)
                if raw_chunk:
                    text_chunk = raw_chunk.decode("utf-8", errors="replace")
                    output_chunks.append(text_chunk)
                    log_file.write(text_chunk)
                    log_file.flush()
                    for line in text_chunk.splitlines():
                        if line.strip():
                            console.print(f"[dim]{line}[/dim]")
                elif process.poll() is not None:
                    break
            elif process.poll() is not None:
                break
    return_code = process.wait()
    return subprocess.CompletedProcess(cmd, return_code, "".join(output_chunks), None)


def _resolve_command(binary_name: str, module_name: str) -> Optional[List[str]]:
    cmd_path = _which_or_none(binary_name)
    if cmd_path:
        return [cmd_path]
    try:
        if importlib.util.find_spec(module_name) is not None:
            return [sys.executable, "-m", module_name]
    except (ModuleNotFoundError, ImportError, ValueError):
        return None
    return None


def _run_evalplus_for_case(case: APICase, dataset: str, parallel: int, out_dir: Path, timeout_sec: int) -> Dict[str, Any]:
    cmd_prefix = _resolve_command("evalplus.evaluate", "evalplus.evaluate")
    if not cmd_prefix:
        return {"benchmark": f"evalplus_{dataset}", "case_name": case.name, "status": "unavailable", "reason": "evalplus package not found"}
    log_path = out_dir / f"evalplus_{case.name}_{dataset}.log"
    evalplus_root = out_dir / f"evalplus_results_{case.name}_{dataset}"
    cmd = cmd_prefix + [
        "--dataset",
        dataset,
        "--greedy",
        "--model",
        case.model,
        "--backend",
        "openai",
        "--base-url",
        case.base_url,
        "--root",
        str(evalplus_root),
        "--resume",
        str(getattr(args, "resume", False)),
    ]
    if parallel > 0:
        cmd.extend(["--parallel", str(parallel)])
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = case.api_key or "EMPTY"
    console.print(f"{case.name} evalplus_{dataset}: running (timeout={timeout_sec}s)")
    try:
        proc = _run_subprocess(cmd, env, log_path, timeout_sec)
    except SUBPROCESS_ERRORS as e:
        return {"benchmark": f"evalplus_{dataset}", "case_name": case.name, "status": "failed", "error": str(e), "log": str(log_path)}
    if proc.returncode != 0:
        return {"benchmark": f"evalplus_{dataset}", "case_name": case.name, "status": "failed", "code": proc.returncode, "log": str(log_path)}
    matched = re.search(r"pass@1\s*:\s*([0-9.]+)", proc.stdout)
    score = float(matched.group(1)) * 100.0 if matched else 0.0
    return {"benchmark": f"evalplus_{dataset}", "case_name": case.name, "status": "ok", "score_pass_at_1": round(score, 4), "log": str(log_path)}


def _run_lm_eval_for_case(case: APICase, tasks: str, batch_size: int, out_dir: Path, timeout_sec: int) -> Dict[str, Any]:
    cmd_prefix = _resolve_command("lm_eval", "lm_eval")
    if not cmd_prefix:
        return {"benchmark": "lm_eval", "case_name": case.name, "status": "unavailable", "reason": "lm_eval package not found"}
    out_json_path = out_dir / f"lm_eval_{case.name}.json"
    log_path = out_dir / f"lm_eval_{case.name}.log"
    api_base = case.base_url.rstrip("/")
    completions_url = f"{api_base}/completions" if api_base.endswith("/v1") else f"{api_base}/v1/completions"
    model_args = f"model={case.model},base_url={completions_url},num_concurrent=8"
    if case.tokenizer:
        model_args += f",tokenizer={case.tokenizer}"
    cmd = cmd_prefix + [
        "--tasks",
        tasks,
        "--output_path",
        str(out_json_path),
        "--confirm_run_unsafe_code",
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--batch_size",
        str(batch_size),
    ]
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = case.api_key or "EMPTY"
    env["LM_EVAL_ALLOW_CODE_EXECUTION"] = "1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    console.print(f"{case.name} lm_eval: running (timeout={timeout_sec}s)")
    try:
        proc = _run_subprocess(cmd, env, log_path, timeout_sec)
    except SUBPROCESS_ERRORS as e:
        return {"benchmark": "lm_eval", "case_name": case.name, "status": "failed", "error": str(e), "log": str(log_path)}
        
    actual_json_path = out_json_path
    if not actual_json_path.exists():
        candidate_jsons = list(out_dir.glob(f"lm_eval_{case.name}*.json"))
        if candidate_jsons:
            candidate_jsons.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            actual_json_path = candidate_jsons[0]
            
    if proc.returncode != 0 and not actual_json_path.exists():
        return {"benchmark": "lm_eval", "case_name": case.name, "status": "failed", "code": proc.returncode, "log": str(log_path)}
    if not actual_json_path.exists():
        return {"benchmark": "lm_eval", "case_name": case.name, "status": "failed", "reason": "No JSON output", "log": str(log_path)}
    try:
        payload = json.loads(actual_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return {"benchmark": "lm_eval", "case_name": case.name, "status": "failed", "error": str(e), "log": str(log_path)}
    metrics: Dict[str, float] = {}
    for task_name, task_result in payload.get("results", {}).items():
        best_metric = None
        for key, value in task_result.items():
            if any(x in key for x in ["acc", "exact_match", "pass"]) and isinstance(value, (int, float)):
                best_metric = float(value)
                break
        if best_metric is not None:
            metrics[task_name] = round(best_metric * 100.0, 4)
    return {"benchmark": "lm_eval", "case_name": case.name, "status": "ok", "tasks": tasks, "metrics": metrics, "output_json": str(out_json_path), "log": str(log_path)}


def _build_accuracy_report_html(records: List[Dict[str, Any]]) -> str:
    records_json = json.dumps(records)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM Accuracy Benchmark Report</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: #0b1020;
      color: #e5e7eb;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 12px;
      margin-bottom: 16px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #1f2937;
      text-align: left;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>LLM Accuracy Benchmark</h1>
    <div class="card"><div id="accuracy_chart"></div></div>
    <div class="card">
      <h2>Detailed Records</h2>
      <table id="records_table"></table>
    </div>
  </div>
  <script>
    const records = {records_json};
    const numericRows = records.filter(r => r.status === 'ok' && r.score !== '' && !Number.isNaN(Number(r.score)));
    const grouped = {{}};
    numericRows.forEach(r => {{
      if (!grouped[r.benchmark]) grouped[r.benchmark] = {{}};
      grouped[r.benchmark][r.case_name] = Number(r.score);
    }});
    const benchmarkNames = Object.keys(grouped).sort();
    const caseNames = [...new Set(numericRows.map(r => r.case_name))].sort();
    const traces = caseNames.map(caseName => {{
      return {{
        x: benchmarkNames,
        y: benchmarkNames.map(b => grouped[b][caseName] ?? null),
        type: 'bar',
        name: caseName
      }};
    }});
    Plotly.newPlot(
      'accuracy_chart',
      traces,
      {{
        barmode: 'group',
        title: 'Accuracy Scores by Benchmark',
        paper_bgcolor: '#111827',
        plot_bgcolor: '#111827',
        font: {{color: '#e5e7eb'}},
        yaxis: {{title: 'Score (%)'}}
      }}
    );
    const table = document.getElementById('records_table');
    const headers = ['Case', 'Benchmark', 'Status', 'Score', 'Meta'];
    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    headers.forEach(h => {{
      const th = document.createElement('th');
      th.textContent = h;
      trh.appendChild(th);
    }});
    thead.appendChild(trh);
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    records.forEach(r => {{
      const tr = document.createElement('tr');
      [r.case_name, r.benchmark, r.status, String(r.score), r.meta].forEach(value => {{
        const td = document.createElement('td');
        td.textContent = String(value);
        tr.appendChild(td);
      }});
      tbody.appendChild(tr);
    }});
    table.appendChild(tbody);
  </script>
</body>
</html>"""


async def _collect_accuracy_for_case(
    args: argparse.Namespace,
    case: APICase,
    benchmarks: set[str],
    raw_dir: Path,
    completed_benchmarks: set[str] = None
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    completed_benchmarks = completed_benchmarks or set()
    records: List[Dict[str, Any]] = []
    passkey_trials_rows: List[Dict[str, Any]] = []
    if "passkey" in benchmarks:
        if "passkey" in completed_benchmarks:
            console.print(f"[dim]Skipping finished passkey for {case.name}[/dim]")
        else:
            passkey_result = await _run_passkey_for_case(args, case)
            records.append(
                {
                    "case_name": case.name,
                    "benchmark": "passkey",
                    "status": "ok",
                    "score": passkey_result["passkey_accuracy"],
                    "meta": "",
                }
            )
            passkey_trials_rows.extend(passkey_result["trials"])
            console.print(f"[dim]{case.name} passkey: {passkey_result['passkey_accuracy']:.2f}%[/dim]")
    if "ppl" in benchmarks:
        if "ppl" in completed_benchmarks:
            pass
        else:
            records.append(
                {
                    "case_name": case.name,
                    "benchmark": "ppl",
                    "status": "not_supported_for_api",
                    "score": "",
                    "meta": "Use local/HF model pipeline from qwen_coder_evalv4.py",
                }
            )
            console.print(f"[yellow]{case.name} ppl: not supported for OpenAI-compatible API mode[/yellow]")
    if "humaneval" in benchmarks or "evalplus_humaneval" in benchmarks:
        if "evalplus_humaneval" in completed_benchmarks:
            console.print(f"[dim]Skipping finished evalplus_humaneval for {case.name}[/dim]")
        else:
            res = _run_evalplus_for_case(case, "humaneval", args.evalplus_parallel, raw_dir, args.command_timeout_seconds)
            records.append(
                {
                    "case_name": case.name,
                    "benchmark": "evalplus_humaneval",
                    "status": res.get("status", "failed"),
                    "score": res.get("score_pass_at_1", ""),
                    "meta": res.get("log", res.get("reason", "")),
                }
            )
            console.print(f"[dim]{case.name} evalplus_humaneval: {res.get('status', 'failed')}[/dim]")
    if "mbpp" in benchmarks or "evalplus_mbpp" in benchmarks:
        if "evalplus_mbpp" in completed_benchmarks:
            console.print(f"[dim]Skipping finished evalplus_mbpp for {case.name}[/dim]")
        else:
            res = _run_evalplus_for_case(case, "mbpp", args.evalplus_parallel, raw_dir, args.command_timeout_seconds)
            records.append(
                {
                    "case_name": case.name,
                    "benchmark": "evalplus_mbpp",
                    "status": res.get("status", "failed"),
                    "score": res.get("score_pass_at_1", ""),
                    "meta": res.get("log", res.get("reason", "")),
                }
            )
            console.print(f"[dim]{case.name} evalplus_mbpp: {res.get('status', 'failed')}[/dim]")
    if "lm_eval" in benchmarks:
        if "lm_eval" in completed_benchmarks:
            console.print(f"[dim]Skipping finished lm_eval for {case.name}[/dim]")
        else:
            res = _run_lm_eval_for_case(case, args.lm_eval_tasks, args.lm_eval_batch_size, raw_dir, args.command_timeout_seconds)
            status = res.get("status", "failed")
            if status == "ok":
                metrics = res.get("metrics", {})
                for task_name, score in metrics.items():
                    records.append(
                        {
                            "case_name": case.name,
                            "benchmark": f"lm_eval_{task_name}",
                            "status": "ok",
                            "score": score,
                            "meta": res.get("output_json", ""),
                        }
                    )
                if not metrics:
                    records.append(
                        {
                            "case_name": case.name,
                            "benchmark": "lm_eval",
                            "status": "ok",
                            "score": "",
                            "meta": "No numeric accuracy metric parsed",
                        }
                    )
            else:
                records.append(
                    {
                        "case_name": case.name,
                        "benchmark": "lm_eval",
                        "status": status,
                        "score": "",
                        "meta": res.get("reason", res.get("log", "")),
                    }
                )
            console.print(f"[dim]{case.name} lm_eval: {status}[/dim]")
    return records, passkey_trials_rows


async def run_llm_accuracy_evaluation(args: argparse.Namespace, report_dir: Path, cases: List[APICase]) -> Dict[str, Any]:
    benchmarks = set(_parse_csv_list(args.accuracy_benchmarks))
    console.print(f"[bold]Accuracy benchmarks:[/bold] {', '.join(sorted(benchmarks))}")
    accuracy_dir = report_dir / "accuracy"
    raw_dir = accuracy_dir / "raw_outputs"
    records_json_path = accuracy_dir / "accuracy_records.json"
    passkey_trials_path = accuracy_dir / "passkey_trials.csv"

    records: List[Dict[str, Any]] = []
    passkey_trials_rows: List[Dict[str, Any]] = []

    if getattr(args, "resume", False) and records_json_path.exists():
        try:
            records = json.loads(records_json_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        if passkey_trials_path.exists():
            try:
                with open(passkey_trials_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    passkey_trials_rows = list(reader)
            except Exception:
                pass

    completed_bmarks_by_case = {}
    for r in records:
        if r.get("status") in {"ok", "not_supported_for_api"}:
            completed_bmarks_by_case.setdefault(r["case_name"], set()).add(r["benchmark"])
    
    for case_name, bmarks in completed_bmarks_by_case.items():
        if any(b.startswith("lm_eval") for b in bmarks):
            bmarks.add("lm_eval")

    for case in cases:
        console.print(f"[bold cyan]Accuracy benchmark case:[/bold cyan] {case.name} ({case.base_url})")
        case_completed = completed_bmarks_by_case.get(case.name, set())
        case_records, case_passkey_trials = await _collect_accuracy_for_case(args, case, benchmarks, raw_dir, case_completed)
        records.extend(case_records)
        passkey_trials_rows.extend(case_passkey_trials)
    summary: Dict[str, Dict[str, Any]] = {}
    for row in records:
        case_name = row["case_name"]
        summary.setdefault(case_name, {})
        summary[case_name][row["benchmark"]] = {"status": row["status"], "score": row["score"], "meta": row["meta"]}
    records_csv_path = accuracy_dir / "accuracy_records.csv"
    summary_json_path = accuracy_dir / "summary.json"
    report_html_path = accuracy_dir / "report.html"
    records_json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_rows_csv(records_csv_path, records, ["case_name", "benchmark", "status", "score", "meta"])
    _write_rows_csv(
        passkey_trials_path,
        passkey_trials_rows,
        ["case_name", "trial", "expected_passkey", "prediction", "prediction_digits", "hit", "error"],
    )
    summary_json_path.write_text(
        json.dumps(
            {
                "suite": "accuracy",
                "benchmarks": sorted(list(benchmarks)),
                "cases": [case.__dict__ for case in cases],
                "summary_by_case": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_html_path.write_text(_build_accuracy_report_html(records), encoding="utf-8")
    return {
        "suite": "accuracy",
        "case_names": [c.name for c in cases],
        "records": records,
        "passkey_trials": passkey_trials_rows,
        "summary": summary
    }


async def run_agent_evaluation(_: argparse.Namespace, __: Path) -> Dict[str, Any]:
    raise NotImplementedError("Agent-task evaluation is reserved for future extensions.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BatchAgent evaluation entrypoint")
    parser.add_argument("--action", default="speed", choices=["speed", "accuracy", "compare"], help="Action to perform")
    parser.add_argument("--output-json", default="result.json", help="Path to save the JSON result (for speed/accuracy)")
    parser.add_argument("--compare-files", nargs="+", default=[], help="JSON files to compare and draw figures from (for compare)")
    parser.add_argument("--compare-output", default="comparison.html", help="Path to save the comparison HTML snippet/report (for compare)")
    
    # LLM API Config (simplified)
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", os.environ.get("VLLM_MODEL", "qwen3.5-9b")), help="Model name. Examples: 'qwen3.5-9b' (vLLM) or 'gemma-4' (llama.cpp)")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", os.environ.get("VLLM_API_KEY", "EMPTY")), help="API Key for the server")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", os.environ.get("VLLM_BASE_URL", "http://100.110.236.127:8000/v1")), help="Base URL of the OpenAI-compatible server. Examples: 'http://100.110.236.127:8000/v1' (vLLM) or 'http://100.65.193.60:8011/v1' (llama.cpp)")
    parser.add_argument("--base-url-name", default=os.environ.get("OPENAI_BASE_URL_NAME", os.environ.get("VLLM_BASE_URL_NAME", "default_endpoint")), help="Name for this endpoint in reports")
    parser.add_argument("--backend", default="vllm", help="Backend type (e.g., vllm, llama.cpp, openai, mlx)")
    parser.add_argument("--tokenizer", default="", help="Tokenizer string for lm_eval local-completions")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--model-max-context", type=int, default=16384)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    
    # Speed test config
    parser.add_argument("--batch-sizes", default="1,2,4", help="Comma-separated batch sizes for oMLX speed test")
    parser.add_argument("--prefill-tokens", default="1024,4096", help="Comma-separated prefill targets for oMLX speed test")
    parser.add_argument("--max-output-tokens", type=int, default=128, help="Target generation tokens (tg)")
    
    # Accuracy test config
    parser.add_argument("--accuracy-benchmarks", default="passkey,humaneval,mbpp,lm_eval")
    parser.add_argument("--passkey-ctx", type=int, default=4096)
    parser.add_argument("--passkey-depth", type=float, default=0.5)
    parser.add_argument("--passkey-trials", type=int, default=10)
    parser.add_argument("--evalplus-parallel", type=int, default=8)
    parser.add_argument("--lm-eval-tasks", default="humaneval")
    parser.add_argument("--lm-eval-batch-size", type=int, default=8)
    parser.add_argument("--command-timeout-seconds", type=int, default=7200)
    
    # Legacy args
    parser.add_argument("--eval-type", default="llm", choices=["llm", "agent"])
    parser.add_argument("--output-dir", default="./agent_workspace")
    parser.add_argument("--resume", action="store_true")
    return parser


async def main_async() -> None:
    parser = build_parser()
    args = parser.parse_args()
    workspace_dir = Path(args.output_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_type == "agent":
        await run_agent_evaluation(args, workspace_dir)
        return
        
    if args.action == "compare":
        from BatchAgent.evaluator_compare import generate_comparison_report
        generate_comparison_report(args.compare_files, args.compare_output)
        return

    cases = _build_llm_cases(args)
    if not cases:
        raise ValueError("No LLM cases configured.")
    case = cases[0]
    
    output_json_path = Path(args.output_json).resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # We use a scratch workspace for logs and temp files
    report_dir = (workspace_dir / "evaluations" / "llm" / now_stamp()).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    
    if args.action == "speed":
        result_data = await run_llm_speed_evaluation(args, report_dir, case)
    elif args.action == "accuracy":
        result_data = await run_llm_accuracy_evaluation(args, report_dir, [case])
    else:
        raise ValueError(f"Unknown action: {args.action}")
        
    output_json_path.write_text(json.dumps(result_data, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[bold green]Evaluation completed.[/bold green]")
    console.print(f"[green]Data saved to:[/green] {output_json_path}")

if __name__ == "__main__":
    asyncio.run(main_async())

"""
python evaluator_main.py --action speed --model "gemma-4" --base-url "http://100.65.193.60:8011/v1" --base-url-name "gemma-4-llamacpp" --output-json gemma4_speed.json


python evaluator_main.py --eval-type llm --resume-dir /Developer/AIserver/agent_workspace/evaluations/llm/2026-03-15_163359 --tokenizer Qwen/Qwen3.5-9B --resume

python evaluator_main.py --action speed --output-json qwen_9b_speed.json

python evaluator_main.py --action compare --compare-files qwen_9b_speed.json
python evaluator_main.py --action compare --compare-files qwen_9b_speed.json --compare-output comparison.html


python evaluator_main.py --action speed --base-url http://localhost:8000/v1 --base-url-name "qwen3.5-9b-local" --output-json qwen_9b_speed_local.json
python evaluator_main.py --action compare --compare-files qwen_9b_speed_local.json --compare-output comparison_local9b.html
"""