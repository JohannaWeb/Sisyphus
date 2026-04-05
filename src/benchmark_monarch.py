#!/usr/bin/env python3
"""Benchmark Monarch KV Paging vs Standard Attention."""

from __future__ import annotations

import argparse
import time
import torch
import math
from pathlib import Path
from model import ByteGPT, GPTConfig


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def benchmark_generation(model, prompt_idx, max_new_tokens, use_cache, mode_name):
    print(f"\nBenchmarking {mode_name}...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    out = model.generate(
        prompt_idx,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache
    )
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    duration = end_time - start_time
    tokens_per_sec = max_new_tokens / duration
    print(f"Duration: {duration:.2f}s")
    print(f"Throughput: {tokens_per_sec:.2f} tok/s")
    return tokens_per_sec


def compute_perplexity(model, data, block_size, use_cache):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        if not use_cache:
            for i in range(0, len(data) - block_size - 1, 32):
                xb = data[i : i + block_size].unsqueeze(0).to(data.device)
                yb = data[i + 1 : i + block_size + 1].unsqueeze(0).to(data.device)
                _, loss = model(xb, yb)
                if loss is not None:
                    total_loss += loss.item() * block_size
                    total_tokens += block_size
        else:
            # Optimized cached PPL: process chunks to trigger paging
            # We take a few long sequences instead of many short ones
            seq_len = min(len(data) - 1, 512)
            for i in range(0, len(data) - seq_len, seq_len):
                # Clear cache for each new sequence
                for block in model.blocks:
                    if block.attn.kv_cache: block.attn.kv_cache.clear()
                
                seq = data[i : i + seq_len].unsqueeze(0).to(data.device)
                # We can't do the whole seq in one forward(use_cache=True) because 
                # that would append all at once. Paging happens on append.
                # But we can do it in smaller steps.
                step_size = 64
                for s in range(0, seq_len - 1, step_size):
                    end = min(s + step_size, seq_len - 1)
                    chunk = seq[:, s:end]
                    targets = seq[:, s+1:end+1]
                    logits, loss = model(chunk, targets=targets, use_cache=True)
                    if loss is not None:
                        total_loss += loss.item() * (end - s)
                        total_tokens += (end - s)
                
                if total_tokens > 1000: break # Cap for speed

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()

    device = resolve_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = checkpoint["config"]["model"]
    model = ByteGPT(GPTConfig(**model_cfg)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = b"fn main() {\n    let x = 5;\n"
    prompt_idx = torch.tensor([list(prompt)], dtype=torch.long, device=device)

    # 1. Benchmark Throughput
    std_speed = benchmark_generation(model, prompt_idx, args.max_new_tokens, use_cache=False, mode_name="Standard (Full Attn)")
    monarch_speed = benchmark_generation(model, prompt_idx, args.max_new_tokens, use_cache=True, mode_name="Monarch-v3 (Paged KV)")
    
    gain = (monarch_speed / std_speed - 1) * 100
    print(f"\nSpeed Gain: {gain:+.1f}%")

    # 2. Perplexity (Simulated)
    # Load some data for perplexity
    corpus_path = Path("data/processed/corpus.txt")
    if corpus_path.exists():
        data = torch.tensor(list(corpus_path.read_bytes()[-5000:]), dtype=torch.long, device=device)
        print("\nComputing Perplexity (sampled)...")
        std_ppl = compute_perplexity(model, data, model_cfg["block_size"], use_cache=False)
        mon_ppl = compute_perplexity(model, data, model_cfg["block_size"], use_cache=True)
        print(f"Standard PPL: {std_ppl:.4f}")
        print(f"Monarch PPL:  {mon_ppl:.4f}")
        print(f"PPL Delta:    {mon_ppl - std_ppl:+.4f}")


if __name__ == "__main__":
    main()
