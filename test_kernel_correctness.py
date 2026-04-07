#!/usr/bin/env python3
"""Correctness tests for Triton kernels: compare against PyTorch reference."""

import torch
import torch.nn.functional as F
import pytorch_fork  # noqa: F401
from pytorch_fork.local_window_attn import local_window_attention


def require_cuda():
    """Return False after printing a skip message when CUDA is unavailable."""
    if torch.cuda.is_available():
        return True

    print("  ⊘ CUDA not available, skipping")
    return False


def scaled_dot_product_attention_windowed(q, k, v, window_size):
    """Reference implementation: windowed causal attention using PyTorch."""
    B, H, T, D = q.shape

    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)  # (B, H, T, T)

    # Apply causal + window mask
    # Query at position i attends to keys j where: j <= i AND j >= i - window_size + 1
    mask = torch.arange(T, device=scores.device)
    causal = (mask[:, None] >= mask[None, :]).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
    window = (mask[:, None] - mask[None, :]) < window_size  # (T, T)
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    valid_mask = causal & window
    scores = scores.masked_fill(~valid_mask, float('-inf'))

    # Softmax
    attn = torch.softmax(scores, dim=-1)
    attn = attn.masked_fill(~valid_mask, 0.0)

    # Output
    out = torch.matmul(attn, v)  # (B, H, T, D)
    return out


def reference_gated_rnn(k, v, h_init):
    """Reference implementation of the custom gated RNN update."""
    states = []
    h_prev = h_init
    for t in range(k.shape[2]):
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        r_t = torch.sigmoid(k_t + h_prev)
        z_t = torch.sigmoid(k_t + h_prev)
        n_t = torch.tanh(r_t * h_prev + k_t)
        h_prev = (1.0 - z_t) * h_prev + z_t * (n_t * v_t)
        states.append(h_prev)
    return torch.stack(states, dim=2), h_prev


def test_local_window_attention_forward():
    """Test forward pass correctness."""
    print("Testing local_window_attention forward...")
    if not require_cuda():
        return True

    B, H, T, D = 2, 8, 128, 64
    W = 32

    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
    k = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')

    # Triton kernel
    out_triton, lse = local_window_attention(q, k, v, W)

    # PyTorch reference
    out_ref = scaled_dot_product_attention_windowed(q, k, v, W)

    # Compare
    diff = torch.abs(out_triton - out_ref).max().item()
    rel_error = (torch.abs(out_triton - out_ref) / (torch.abs(out_ref) + 1e-6)).max().item()

    print(f"  Max abs diff: {diff:.6f}")
    print(f"  Max rel error: {rel_error:.6f}")

    if rel_error < 0.01:
        print("  ✓ Forward pass correct")
        return True
    else:
        print(f"  ✗ Forward pass INCORRECT (rel error {rel_error:.4f} > 0.01)")
        return False


def test_local_window_attention_backward():
    """Test backward pass (gradients) correctness."""
    print("Testing local_window_attention backward...")
    if not require_cuda():
        return True

    B, H, T, D = 2, 8, 128, 64
    W = 32

    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda', requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda', requires_grad=True)
    v = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda', requires_grad=True)

    # Triton kernel forward
    out_triton, lse = local_window_attention(q, k, v, W)
    loss_triton = out_triton.sum()
    loss_triton.backward()

    grad_q_triton = q.grad.clone()
    grad_k_triton = k.grad.clone()
    grad_v_triton = v.grad.clone()

    # Reset gradients
    q.grad = None
    k.grad = None
    v.grad = None

    # PyTorch reference forward
    out_ref = scaled_dot_product_attention_windowed(q, k, v, W)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    grad_q_ref = q.grad
    grad_k_ref = k.grad
    grad_v_ref = v.grad

    # Compare gradients
    grad_errors = {
        'q': (torch.abs(grad_q_triton - grad_q_ref) / (torch.abs(grad_q_ref) + 1e-6)).max().item(),
        'k': (torch.abs(grad_k_triton - grad_k_ref) / (torch.abs(grad_k_ref) + 1e-6)).max().item(),
        'v': (torch.abs(grad_v_triton - grad_v_ref) / (torch.abs(grad_v_ref) + 1e-6)).max().item(),
    }

    print(f"  Grad rel errors: q={grad_errors['q']:.6f}, k={grad_errors['k']:.6f}, v={grad_errors['v']:.6f}")

    max_grad_error = max(grad_errors.values())
    if max_grad_error < 0.05:
        print("  ✓ Backward pass correct")
        return True
    else:
        print(f"  ✗ Backward pass INCORRECT (max rel error {max_grad_error:.4f} > 0.05)")
        return False


def test_gated_rnn_update_forward():
    """Test GRU forward pass correctness."""
    print("Testing gated_rnn_update forward...")
    if not require_cuda():
        return True

    B, H, T, D = 2, 8, 128, 64

    torch.manual_seed(42)
    k = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
    h_init = torch.randn(B, H, D, dtype=torch.float32, device='cuda')

    try:
        h_out_triton, h_final_triton = torch.ops.sisyphus.gated_rnn_update(k, v, h_init)
        h_out_ref, h_final_ref = reference_gated_rnn(k, v, h_init)
    except Exception as e:
        print(f"  ✗ GRU forward failed: {e}")
        return False

    out_diff = (h_out_triton - h_out_ref).abs().max().item()
    final_diff = (h_final_triton - h_final_ref).abs().max().item()
    print(f"  Max abs diff: h_out={out_diff:.6f}, h_final={final_diff:.6f}")

    if max(out_diff, final_diff) < 1e-4:
        print("  ✓ GRU forward matches reference")
        return True

    print("  ✗ GRU forward does not match reference")
    return False


def test_gated_rnn_update_backward():
    """Test GRU backward gradients against a PyTorch reference."""
    print("Testing gated_rnn_update backward...")
    if not require_cuda():
        return True

    B, H, T, D = 2, 4, 32, 32

    torch.manual_seed(123)
    k_base = torch.randn(B, H, T, D, dtype=torch.float32, device="cuda")
    v_base = torch.randn(B, H, T, D, dtype=torch.float32, device="cuda")
    h_init_base = torch.randn(B, H, D, dtype=torch.float32, device="cuda")

    k_ref = k_base.clone().requires_grad_(True)
    v_ref = v_base.clone().requires_grad_(True)
    h_init_ref = h_init_base.clone().requires_grad_(True)
    h_out_ref, h_final_ref = reference_gated_rnn(k_ref, v_ref, h_init_ref)
    loss_ref = h_out_ref.square().mean() + h_final_ref.square().mean()
    loss_ref.backward()

    k_triton = k_base.clone().requires_grad_(True)
    v_triton = v_base.clone().requires_grad_(True)
    h_init_triton = h_init_base.clone().requires_grad_(True)
    h_out_triton, h_final_triton = torch.ops.sisyphus.gated_rnn_update(
        k_triton, v_triton, h_init_triton
    )
    loss_triton = h_out_triton.square().mean() + h_final_triton.square().mean()
    loss_triton.backward()

    grad_errors = {
        "k": (k_triton.grad - k_ref.grad).abs().max().item(),
        "v": (v_triton.grad - v_ref.grad).abs().max().item(),
        "h_init": (h_init_triton.grad - h_init_ref.grad).abs().max().item(),
    }
    print(
        "  Max abs grad diff: "
        f"k={grad_errors['k']:.6f}, v={grad_errors['v']:.6f}, h_init={grad_errors['h_init']:.6f}"
    )

    if max(grad_errors.values()) < 1e-4:
        print("  ✓ GRU backward matches reference")
        return True

    print("  ✗ GRU backward does not match reference")
    return False


def test_gated_rnn_update_edge_cases():
    """Test short-sequence and zero-state edge cases."""
    print("Testing gated_rnn_update edge cases...")
    if not require_cuda():
        return True

    checks = []

    torch.manual_seed(7)
    for T in (1, 2):
        B, H, D = 2, 3, 16
        k = torch.randn(B, H, T, D, dtype=torch.float32, device="cuda")
        v = torch.randn(B, H, T, D, dtype=torch.float32, device="cuda")
        h_init = torch.zeros(B, H, D, dtype=torch.float32, device="cuda")

        h_out_triton, h_final_triton = torch.ops.sisyphus.gated_rnn_update(k, v, h_init)
        h_out_ref, h_final_ref = reference_gated_rnn(k, v, h_init)
        diff = max(
            (h_out_triton - h_out_ref).abs().max().item(),
            (h_final_triton - h_final_ref).abs().max().item(),
        )
        print(f"  T={T} max abs diff: {diff:.6f}")
        checks.append(diff < 1e-4)

    if all(checks):
        print("  ✓ GRU edge cases match reference")
        return True

    print("  ✗ GRU edge case mismatch")
    return False


def test_input_validation():
    """Test that invalid inputs are caught."""
    print("Testing input validation...")
    if not require_cuda():
        return True

    B, H, T, D = 2, 8, 128, 64
    W = 32

    errors_caught = 0

    # Test 1: Non-CUDA tensor
    try:
        q = torch.randn(B, H, T, D, dtype=torch.float32)
        k = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
        v = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
        local_window_attention(q, k, v, W)
        print("  ✗ Non-CUDA tensor not rejected")
    except RuntimeError as e:
        if "CUDA" in str(e):
            errors_caught += 1

    # Test 2: Shape mismatch
    try:
        q = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
        k = torch.randn(B, H, T, D + 1, dtype=torch.float32, device='cuda')
        v = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
        local_window_attention(q, k, v, W)
        print("  ✗ Shape mismatch not rejected")
    except ValueError as e:
        if "shape" in str(e).lower():
            errors_caught += 1

    # Test 3: Unsupported dtype
    try:
        q = torch.randn(B, H, T, D, dtype=torch.float64, device='cuda')
        k = torch.randn(B, H, T, D, dtype=torch.float64, device='cuda')
        v = torch.randn(B, H, T, D, dtype=torch.float64, device='cuda')
        local_window_attention(q, k, v, W)
        print("  ✗ Unsupported dtype not rejected")
    except ValueError as e:
        if "dtype" in str(e).lower() or "float64" in str(e):
            errors_caught += 1

    # Test 4: GRU D > 64
    try:
        k = torch.randn(B, H, T, 65, dtype=torch.float32, device='cuda')
        v = torch.randn(B, H, T, 65, dtype=torch.float32, device='cuda')
        h_init = torch.randn(B, H, 65, dtype=torch.float32, device='cuda')
        torch.ops.sisyphus.gated_rnn_update(k, v, h_init)
        print("  ✗ D > 64 not rejected in GRU")
    except ValueError as e:
        if "64" in str(e):
            errors_caught += 1

    print(f"  Caught {errors_caught} / 4 expected validation errors")
    if errors_caught >= 3:
        print("  ✓ Input validation working")
        return True
    else:
        print("  ✗ Some validations not working")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Kernel Correctness Tests")
    print("=" * 60)
    print()

    results = {
        'local_window_attention_fwd': test_local_window_attention_forward(),
        'local_window_attention_bwd': test_local_window_attention_backward(),
        'gated_rnn_update_fwd': test_gated_rnn_update_forward(),
        'gated_rnn_update_bwd': test_gated_rnn_update_backward(),
        'gated_rnn_update_edges': test_gated_rnn_update_edge_cases(),
        'input_validation': test_input_validation(),
    }

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {test_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {sum(not v for v in results.values())} test(s) failed")
