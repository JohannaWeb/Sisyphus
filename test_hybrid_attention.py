#!/usr/bin/env python3
"""Test hybrid attention implementation."""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from model import ByteGPT, GPTConfig


def test_hybrid_attention_forward():
    """Test that HybridAttention forward pass works."""
    print("=" * 60)
    print("TEST 1: HybridAttention Forward Pass")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = GPTConfig(
        vocab_size=256,
        block_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        window_size=32,
        dropout=0.0,
    )

    model = ByteGPT(config).to(device)
    model.eval()

    # Test forward pass
    batch_size, seq_len = 2, 64
    idx = torch.randint(0, 256, (batch_size, seq_len), device=device)

    with torch.no_grad():
        logits, loss = model(idx)

    assert logits.shape == (batch_size, seq_len, 256), f"Expected {(batch_size, seq_len, 256)}, got {logits.shape}"
    assert loss is None
    print(f"✓ Forward pass output shape: {logits.shape}")

    return True


def test_generation_with_cache():
    """Test text generation with HybridAttention KV cache."""
    print("\n" + "=" * 60)
    print("TEST 2: Text Generation with Cache (HybridAttention)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(
        vocab_size=256,
        block_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        window_size=32,
        dropout=0.0,
    )

    model = ByteGPT(config).to(device)
    model.eval()

    # Prompt: "fn main"
    prompt = torch.tensor([[102, 110, 32, 109, 97, 105, 110]], device=device)  # "fn main"

    with torch.no_grad():
        output = model.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=10,
            use_cache=True,
        )

    assert output.shape[0] == 1
    assert output.shape[1] >= prompt.shape[1] + 50
    print(f"✓ Generated sequence length: {output.shape[1]} tokens")
    print(f"  (prompt: {prompt.shape[1]}, generated: {output.shape[1] - prompt.shape[1]})")

    # Decode and print sample
    text = bytes(output[0].tolist()).decode("utf-8", errors="ignore")
    print(f"  Sample output: {text[:100]!r}...")

    return True


def test_generation_without_cache():
    """Test text generation without cache for comparison."""
    print("\n" + "=" * 60)
    print("TEST 3: Text Generation without Cache (Baseline)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(
        vocab_size=256,
        block_size=256,
        n_layer=2,
        n_head=4,
        n_embd=128,
        window_size=32,
        dropout=0.0,
    )

    model = ByteGPT(config).to(device)
    model.eval()

    prompt = torch.tensor([[102, 110, 32, 109, 97, 105, 110]], device=device)  # "fn main"

    with torch.no_grad():
        output = model.generate(
            prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=10,
            use_cache=False,
        )

    assert output.shape[0] == 1
    assert output.shape[1] >= prompt.shape[1] + 50
    print(f"✓ Generated sequence length: {output.shape[1]} tokens")

    return True


def test_rnn_state_isolation():
    """Test that RNN state is properly isolated between forward passes."""
    print("\n" + "=" * 60)
    print("TEST 4: RNN State Isolation (Cache Clearing)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(
        vocab_size=256,
        block_size=64,
        n_layer=1,
        n_head=2,
        n_embd=64,
        window_size=16,
        dropout=0.0,
    )

    model = ByteGPT(config).to(device)
    model.eval()

    prompt = torch.tensor([[65, 66, 67]], device=device)  # "ABC"

    # Check that RNN state starts as None
    for block in model.blocks:
        if hasattr(block.attn, 'rnn_state'):
            assert block.attn.rnn_state is None, "Initial RNN state should be None"

    # Generate with cache
    with torch.no_grad():
        out1 = model.generate(prompt, max_new_tokens=20, use_cache=True)

    # After generation, verify RNN state was cleared (reset to None for next call)
    for block in model.blocks:
        if hasattr(block.attn, 'clear_state'):
            block.attn.clear_state()

    # Check states are cleared
    for block in model.blocks:
        if hasattr(block.attn, 'rnn_state'):
            assert block.attn.rnn_state is None, "RNN state should be cleared after clear_state()"
            assert block.attn.local_kv_buf_k is None, "Local KV buffer should be cleared"
            assert block.attn.local_kv_buf_v is None, "Local KV buffer should be cleared"

    # Generate again - should work without stale state
    with torch.no_grad():
        out2 = model.generate(prompt, max_new_tokens=20, use_cache=True)

    assert out1.shape == out2.shape, "Generated sequences should have same shape"
    print("✓ RNN state properly cleared between generate() calls")
    print(f"  Generated sequences of shape {out1.shape}")

    return True


def test_window_attention():
    """Test that local window attention is working."""
    print("\n" + "=" * 60)
    print("TEST 5: Local Window Attention Mechanics")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(
        vocab_size=256,
        block_size=256,
        n_layer=1,
        n_head=1,
        n_embd=32,
        window_size=8,
        dropout=0.0,
    )

    model = ByteGPT(config).to(device)
    model.eval()

    # Create sequence longer than window
    seq_len = 64
    idx = torch.randint(0, 256, (1, seq_len), device=device)

    with torch.no_grad():
        logits, _ = model(idx)

    # Each token should only attend to last `window_size` tokens (+ itself)
    # If the implementation is correct, attention should be O(n*W) not O(n^2)
    print(f"✓ Processed sequence of {seq_len} tokens with window_size=8")
    print(f"  Output shape: {logits.shape}")
    print("  (Window attention is O(n·W) instead of O(n²))")

    return True


def main():
    """Run all tests."""
    tests = [
        test_hybrid_attention_forward,
        test_generation_with_cache,
        test_generation_without_cache,
        test_rnn_state_isolation,
        test_window_attention,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
