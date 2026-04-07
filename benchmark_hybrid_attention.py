#!/usr/bin/env python3
"""Benchmark hybrid attention with and without cache."""

import sys
import time
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import ByteGPT, GPTConfig


def benchmark_generation(use_cache: bool, max_new_tokens: int = 100, runs: int = 3):
    """Benchmark text generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(
        vocab_size=256,
        block_size=256,
        n_layer=4,
        n_head=6,
        n_embd=192,
        window_size=64,
        dropout=0.0,
    )

    model = ByteGPT(config).to(device)
    model.eval()

    prompt = torch.tensor([[102, 110, 32, 109, 97, 105, 110]], device=device)  # "fn main"

    # Warmup
    with torch.no_grad():
        _ = model.generate(prompt, max_new_tokens=10, use_cache=use_cache)

    # Benchmark
    times = []
    for _ in range(runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=max_new_tokens, use_cache=use_cache)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    tokens_per_sec = (output.shape[1] - prompt.shape[1]) / avg_time

    return avg_time, tokens_per_sec, output.shape[1]


def main():
    """Run benchmarks."""
    print("=" * 70)
    print("HybridAttention Performance Benchmark")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Benchmark without cache
    print("Benchmark 1: Generation WITHOUT Cache (Full Attention O(n²))")
    print("-" * 70)
    time_no_cache, tps_no_cache, seq_len = benchmark_generation(use_cache=False, max_new_tokens=100, runs=3)
    print(f"  Time per generation: {time_no_cache:.3f}s")
    print(f"  Throughput: {tps_no_cache:.1f} tokens/sec")
    print(f"  Sequence length generated: {seq_len} tokens")

    print()

    # Benchmark with cache (HybridAttention)
    print("Benchmark 2: Generation WITH Cache (HybridAttention O(n·W + n·D))")
    print("-" * 70)
    time_cache, tps_cache, _ = benchmark_generation(use_cache=True, max_new_tokens=100, runs=3)
    print(f"  Time per generation: {time_cache:.3f}s")
    print(f"  Throughput: {tps_cache:.1f} tokens/sec")

    print()

    # Calculate speedup
    speedup = time_no_cache / time_cache
    print("=" * 70)
    print(f"Speedup with HybridAttention + Cache: {speedup:.2f}x")
    print(f"Throughput improvement: {(tps_cache / tps_no_cache):.2f}x")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
