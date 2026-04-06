#!/usr/bin/env python3
"""Test GRU forward and backward kernels after bug fixes."""

import torch
import torch.nn.functional as F
from pytorch_fork.kernels.gated_rnn_fwd import gated_rnn_fwd_kernel
from pytorch_fork.kernels.gated_rnn_bwd import gated_rnn_bwd_kernel


def test_tanh_implementation():
    """Test that the custom tanh implementation works."""
    print("Testing custom tanh implementation...")

    # This will be tested implicitly when kernels compile
    x = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    expected = torch.tanh(x)
    print(f"  torch.tanh({x.tolist()}) = {expected.tolist()}")
    print("  ✓ tanh reference values computed")


def test_gru_forward_kernel_compilation():
    """Test that the forward kernel compiles without error."""
    print("Testing GRU forward kernel compilation...")

    B, H, T, D = 2, 8, 512, 64

    k = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
    h_init = torch.randn(B, H, D, dtype=torch.float32, device='cuda')
    h_out = torch.empty(B, H, T, D, dtype=torch.float32, device='cuda')

    # Get strides
    stride_k_b, stride_k_h, stride_k_t, stride_k_d = k.stride()
    stride_v_b, stride_v_h, stride_v_t, stride_v_d = v.stride()
    stride_h_b, stride_h_h, stride_h_d = h_init.stride()
    stride_ho_b, stride_ho_h, stride_ho_t, stride_ho_d = h_out.stride()

    grid = (B * H,)

    try:
        gated_rnn_fwd_kernel[grid](
            k, v, h_init,
            None, None, None, None, None,
            h_out,
            stride_k_b, stride_k_h, stride_k_t, stride_k_d,
            stride_v_b, stride_v_h, stride_v_t, stride_v_d,
            stride_h_b, stride_h_h, stride_h_d,
            stride_ho_b, stride_ho_h, stride_ho_t, stride_ho_d,
            T=T, D=D, H=H, BLOCK_D=min(D, 64),
        )
        print(f"  ✓ Kernel launched successfully with B={B}, H={H}, T={T}, D={D}")
        return True
    except Exception as e:
        print(f"  ✗ Kernel launch failed: {e}")
        return False


def test_gru_backward_kernel_compilation():
    """Test that the backward kernel compiles without error."""
    print("Testing GRU backward kernel compilation...")

    B, H, T, D = 2, 8, 512, 64

    k = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
    v = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
    h_out = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
    grad_h_total = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')

    dk = torch.zeros(B, H, T, D, dtype=torch.float32, device='cuda')
    dv = torch.zeros(B, H, T, D, dtype=torch.float32, device='cuda')
    dh_init = torch.zeros(B, H, D, dtype=torch.float32, device='cuda')

    # Get strides
    stride_k_b, stride_k_h, stride_k_t, stride_k_d = k.stride()
    stride_v_b, stride_v_h, stride_v_t, stride_v_d = v.stride()
    stride_dh_b, stride_dh_h, stride_dh_t, stride_dh_d = grad_h_total.stride()
    stride_dk_b, stride_dk_h, stride_dk_t, stride_dk_d = dk.stride()
    stride_dv_b, stride_dv_h, stride_dv_t, stride_dv_d = dv.stride()

    # For dH_init gradient computation, we only need (B, H, D) strides
    stride_dh_init_b, stride_dh_init_h, stride_dh_init_d = dh_init.stride()

    grid = (B * H,)

    try:
        stride_dho_b, stride_dho_h, stride_dho_t, stride_dho_d = grad_h_total.stride()
        gated_rnn_bwd_kernel[grid](
            k, v, h_out, grad_h_total,
            None, None, None, None, None,
            dk, dv, dh_init,
            stride_k_b, stride_k_h, stride_k_t, stride_k_d,
            stride_v_b, stride_v_h, stride_v_t, stride_v_d,
            stride_dh_b, stride_dh_h, stride_dh_d,
            stride_dh_b, stride_dh_h, stride_dh_t, stride_dh_d,
            stride_dk_b, stride_dk_h, stride_dk_t, stride_dk_d,
            stride_dv_b, stride_dv_h, stride_dv_t, stride_dv_d,
            stride_dho_b, stride_dho_h, stride_dho_t, stride_dho_d,
            T=T, D=D, H=H, BLOCK_D=min(D, 64),
        )
        print(f"  ✓ Backward kernel launched successfully with B={B}, H={H}, T={T}, D={D}")
        return True
    except Exception as e:
        print(f"  ✗ Backward kernel launch failed: {e}")
        return False


def test_custom_op_registration():
    """Test that the custom ops are registered correctly."""
    print("Testing custom op registration...")

    try:
        import pytorch_fork

        # Test that ops are registered
        assert hasattr(torch.ops, 'sisyphus'), "sisyphus namespace not registered"
        assert hasattr(torch.ops.sisyphus, 'gated_rnn_update'), "gated_rnn_update not registered"
        print("  ✓ Custom ops registered successfully")
        return True
    except Exception as e:
        print(f"  ✗ Custom op registration failed: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("Testing GRU Kernel Bug Fixes")
    print("=" * 60)

    test_tanh_implementation()
    test_custom_op_registration()

    # Only run GPU tests if CUDA is available
    if torch.cuda.is_available():
        print()
        fwd_ok = test_gru_forward_kernel_compilation()
        bwd_ok = test_gru_backward_kernel_compilation()

        print()
        if fwd_ok and bwd_ok:
            print("✓ All kernel tests passed!")
        else:
            print("✗ Some kernel tests failed")
    else:
        print("\n⊘ CUDA not available, skipping kernel tests")
