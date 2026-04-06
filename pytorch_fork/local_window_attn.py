"""PyTorch registration of local window attention via torch.library."""

import torch
import triton
from torch import Tensor

from .kernels.local_window_attn_fwd_fixed import local_window_attn_fwd_kernel
from .kernels.local_window_attn_bwd import local_window_attn_bwd_kernel


@torch.library.triton_op("sisyphus::local_window_attention", mutates_args=())
def local_window_attention(
    q: Tensor,  # (B, H, T, D)
    k: Tensor,
    v: Tensor,
    window_size: int,
) -> tuple[Tensor, Tensor]:  # (out, lse)
    """
    Local window causal attention via Triton kernel.

    Args:
        q, k, v: (B, H, T, D) attention tensors
        window_size: W, causal window size (e.g., 128)

    Returns:
        out: (B, H, T, D) attention output
        lse: (B, H, T) log-sum-exp for backward stability
    """
    # BUG FIX #7: Input validation
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("local_window_attention requires CUDA tensors")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"q, k, v must have same shape. Got q={q.shape}, k={k.shape}, v={v.shape}"
        )

    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            f"local_window_attention requires float32/float16/bfloat16, got {q.dtype}"
        )

    B, H, T, D = q.shape
    W = window_size

    # Allocate output and LSE
    out = torch.empty_like(q)
    lse = torch.empty(B, H, T, dtype=torch.float32, device=q.device)

    # Block sizes for tiling
    BLOCK_M = 16  # query tile size (reduced from 32 to fit backward kernel SRAM)
    BLOCK_D = D   # head dimension (fixed)

    # BLOCK_N = 64: Reduced from 128 to fit multi-tile loop within SRAM
    # For query tile [q_start, q_start+BLOCK_M), valid keys span ~W+BLOCK_M-1 positions (~143 with W=128, BLOCK_M=16).
    # With BLOCK_N=64: requires ~3 key tiles per query tile.
    # Multi-tile iteration with online softmax ensures FULL coverage (BUG #1 fix).
    # SRAM usage: K+V tiles (16KB/iter) + Q (4KB) + accumulators (2KB) = ~22KB per iteration.
    BLOCK_N = 64

    grid = (triton.cdiv(T, BLOCK_M), B * H)

    # Get strides
    stride_q_b, stride_q_h, stride_q_t, stride_q_d = q.stride()
    stride_k_b, stride_k_h, stride_k_t, stride_k_d = k.stride()
    stride_v_b, stride_v_h, stride_v_t, stride_v_d = v.stride()
    stride_o_b, stride_o_h, stride_o_t, stride_o_d = out.stride()
    stride_lse_b, stride_lse_h, stride_lse_t = lse.stride()

    # Launch kernel
    local_window_attn_fwd_kernel[grid](
        q, k, v, out, lse,
        stride_q_b, stride_q_h, stride_q_t, stride_q_d,
        stride_k_b, stride_k_h, stride_k_t, stride_k_d,
        stride_v_b, stride_v_h, stride_v_t, stride_v_d,
        stride_o_b, stride_o_h, stride_o_t, stride_o_d,
        stride_lse_b, stride_lse_h, stride_lse_t,
        T=T, D=D, H=H, W=W, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return out, lse


@torch.library.register_fake("sisyphus::local_window_attention")
def _(q, k, v, window_size):
    """Fake implementation for shape inference."""
    lse = q.new_empty(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32)
    return torch.empty_like(q), lse


def _local_window_attn_backward(ctx, grad_out, grad_lse):
    """Backward pass: compute dQ, dK, dV from grad_out."""
    q, k, v, out, lse = ctx.saved_tensors
    W = ctx.window_size

    B, H, T, D = q.shape
    BLOCK_M = 16  # Reduced from 32 to fit backward kernel SRAM (load Q, K, V, O, dO, LSE)
    BLOCK_D = D
    BLOCK_N = 64

    # Allocate gradients
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    # Grid: (num_key_tiles, B*H)
    grid = (triton.cdiv(T, BLOCK_N), B * H)

    # Get strides
    stride_q_b, stride_q_h, stride_q_t, stride_q_d = q.stride()
    stride_k_b, stride_k_h, stride_k_t, stride_k_d = k.stride()
    stride_v_b, stride_v_h, stride_v_t, stride_v_d = v.stride()
    stride_o_b, stride_o_h, stride_o_t, stride_o_d = out.stride()
    stride_lse_b, stride_lse_h, stride_lse_t = lse.stride()
    stride_dq_b, stride_dq_h, stride_dq_t, stride_dq_d = dq.stride()
    stride_dk_b, stride_dk_h, stride_dk_t, stride_dk_d = dk.stride()
    stride_dv_b, stride_dv_h, stride_dv_t, stride_dv_d = dv.stride()
    stride_do_b, stride_do_h, stride_do_t, stride_do_d = grad_out.stride()

    # Launch backward kernel
    local_window_attn_bwd_kernel[grid](
        q, k, v, out, lse, grad_out.contiguous(),
        dq, dk, dv,
        stride_q_b, stride_q_h, stride_q_t, stride_q_d,
        stride_k_b, stride_k_h, stride_k_t, stride_k_d,
        stride_v_b, stride_v_h, stride_v_t, stride_v_d,
        stride_o_b, stride_o_h, stride_o_t, stride_o_d,
        stride_lse_b, stride_lse_h, stride_lse_t,
        stride_dq_b, stride_dq_h, stride_dq_t, stride_dq_d,
        stride_dk_b, stride_dk_h, stride_dk_t, stride_dk_d,
        stride_dv_b, stride_dv_h, stride_dv_t, stride_dv_d,
        stride_do_b, stride_do_h, stride_do_t, stride_do_d,
        T=T, D=D, H=H, W=W, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return dq, dk, dv, None  # None for window_size (not differentiable)


def _local_window_attn_setup_context(ctx, inputs, output):
    """Save tensors for backward."""
    q, k, v, window_size = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)
    ctx.window_size = window_size


torch.library.register_autograd(
    "sisyphus::local_window_attention",
    _local_window_attn_backward,
    setup_context=_local_window_attn_setup_context,
)
