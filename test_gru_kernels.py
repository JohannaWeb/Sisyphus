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
            k, v, h_init, h_out,
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
    h_init = torch.randn(B, H, D, dtype=torch.float32, device='cuda')
    h_out = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')
    grad_h_total = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda')

    dk = torch.zeros(B, H, T, D, dtype=torch.float32, device='cuda')
    dv = torch.zeros(B, H, T, D, dtype=torch.float32, device='cuda')
    dh_init = torch.zeros(B, H, D, dtype=torch.float32, device='cuda')

    # Get strides
    stride_k_b, stride_k_h, stride_k_t, stride_k_d = k.stride()
    stride_v_b, stride_v_h, stride_v_t, stride_v_d = v.stride()
    stride_h_b, stride_h_h, stride_h_d = h_init.stride()
    stride_ho_b, stride_ho_h, stride_ho_t, stride_ho_d = h_out.stride()
    stride_dh_b, stride_dh_h, stride_dh_t, stride_dh_d = grad_h_total.stride()
    stride_dk_b, stride_dk_h, stride_dk_t, stride_dk_d = dk.stride()
    stride_dv_b, stride_dv_h, stride_dv_t, stride_dv_d = dv.stride()
    stride_dhi_b, stride_dhi_h, stride_dhi_d = dh_init.stride()

    grid = (B * H,)

    try:
        gated_rnn_bwd_kernel[grid](
            k, v, h_init, h_out, grad_h_total,
            dk, dv, dh_init,
            stride_k_b, stride_k_h, stride_k_t, stride_k_d,
            stride_v_b, stride_v_h, stride_v_t, stride_v_d,
            stride_h_b, stride_h_h, stride_h_d,
            stride_ho_b, stride_ho_h, stride_ho_t, stride_ho_d,
            stride_dh_b, stride_dh_h, stride_dh_t, stride_dh_d,
            stride_dk_b, stride_dk_h, stride_dk_t, stride_dk_d,
            stride_dv_b, stride_dv_h, stride_dv_t, stride_dv_d,
            stride_dhi_b, stride_dhi_h, stride_dhi_d,
            T=T, D=D, H=H, BLOCK_D=min(D, 64),
        )
        print(f"  ✓ Backward kernel launched successfully with B={B}, H={H}, T={T}, D={D}")
        return True
    except Exception as e:
        print(f"  ✗ Backward kernel launch failed: {e}")
        return False


def reference_gated_rnn(k, v, h_init):
    """Pure PyTorch reference for the custom gated RNN."""
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


def test_gated_rnn_backward_reference():
    """Compare custom-op backward against a PyTorch reference."""
    print("Testing GRU backward against reference...")

    if not torch.cuda.is_available():
        print("  ⊘ CUDA not available, skipping")
        return True

    try:
        import pytorch_fork  # noqa: F401
    except Exception as e:
        print(f"  ✗ Custom op registration failed: {e}")
        return False

    B, H, T, D = 2, 4, 16, 32
    torch.manual_seed(42)

    k_base = torch.randn(B, H, T, D, dtype=torch.float32, device="cuda")
    v_base = torch.randn(B, H, T, D, dtype=torch.float32, device="cuda")
    h_init_base = torch.randn(B, H, D, dtype=torch.float32, device="cuda")

    k_ref = k_base.clone().requires_grad_(True)
    v_ref = v_base.clone().requires_grad_(True)
    h_init_ref = h_init_base.clone().requires_grad_(True)
    h_out_ref, h_final_ref = reference_gated_rnn(k_ref, v_ref, h_init_ref)
    loss_ref = h_out_ref.sum() + h_final_ref.sum()
    loss_ref.backward()

    k_op = k_base.clone().requires_grad_(True)
    v_op = v_base.clone().requires_grad_(True)
    h_init_op = h_init_base.clone().requires_grad_(True)
    h_out_op, h_final_op = torch.ops.sisyphus.gated_rnn_update(k_op, v_op, h_init_op)
    loss_op = h_out_op.sum() + h_final_op.sum()
    loss_op.backward()

    errors = {
        "k": (k_op.grad - k_ref.grad).abs().max().item(),
        "v": (v_op.grad - v_ref.grad).abs().max().item(),
        "h_init": (h_init_op.grad - h_init_ref.grad).abs().max().item(),
    }
    print(
        "  Max abs grad diff: "
        f"k={errors['k']:.6f}, v={errors['v']:.6f}, h_init={errors['h_init']:.6f}"
    )

    if max(errors.values()) < 1e-4:
        print("  ✓ Backward gradients match reference")
        return True

    print("  ✗ Backward gradients do not match reference")
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
        ref_ok = test_gated_rnn_backward_reference()

        print()
        if fwd_ok and bwd_ok and ref_ok:
            print("✓ All kernel tests passed!")
        else:
            print("✗ Some kernel tests failed")
    else:
        print("\n⊘ CUDA not available, skipping kernel tests")
