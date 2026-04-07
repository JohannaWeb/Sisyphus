"""PyTorch registration of GRU update via torch.library."""

import torch
from torch import Tensor

from .kernels.gated_rnn_fwd import gated_rnn_fwd_kernel
from .kernels.gated_rnn_bwd import gated_rnn_bwd_kernel


@torch.library.custom_op("sisyphus::gated_rnn_update", mutates_args=())
def gated_rnn_update(
    k: Tensor,  # (B, H, T, D) reset/update gate inputs
    v: Tensor,  # (B, H, T, D) candidate state * value weighting
    h_init: Tensor,  # (B, H, D) initial hidden state
) -> tuple[Tensor, Tensor]:  # (h_out, h_final)
    r"""
    Simplified gated recurrent update (NOT a standard GRU) for HybridAttention.

    ⚠️  IMPORTANT: This is NOT a true GRU. It is a simplified gated-RNN cell
    designed specifically for the HybridAttention architecture where:
    - k, v are already projections from the attention module
    - Gates and candidates are computed from these projections only
    - No separate learnable weight matrices (W_r, U_r, W_z, U_z, etc.)

    The actual recurrence is:
        r_t = sigmoid(k_t + h_{t-1})         # reset gate (simplified: no weights)
        z_t = sigmoid(k_t + h_{t-1})         # update gate (simplified: no weights)
        n_t = tanh(r_t * h_{t-1} + k_t)      # candidate state
        h_t = (1-z_t) * h_{t-1} + z_t * (n_t * v_t)

    This differs from a standard GRU in that:
    - No separate W and U weight matrices for gates
    - Gates computed from pre-projected inputs (k, v) only
    - Candidate state mixing involves element-wise v_t multiplication
    - Designed for integration with local-window attention, not standalone RNN use

    For a true GRU implementation, use torch.nn.GRUCell or torch.nn.GRU.

    Args:
        k: (B, H, T, D) gating signal (from K projection in attention)
        v: (B, H, T, D) value modulation (from V projection in attention)
        h_init: (B, H, D) initial hidden state

    Returns:
        h_out: (B, H, T, D) all hidden states
        h_final: (B, H, D) final hidden state at t=T-1
    """
    # BUG FIX #7: Input validation
    if not (k.is_cuda and v.is_cuda and h_init.is_cuda):
        raise RuntimeError("gated_rnn_update requires CUDA tensors")

    if k.dtype != v.dtype or k.dtype != h_init.dtype:
        raise ValueError(
            f"k, v, h_init must have same dtype. Got k={k.dtype}, v={v.dtype}, h_init={h_init.dtype}"
        )

    if k.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            f"gated_rnn_update requires float32/float16/bfloat16, got {k.dtype}"
        )

    if k.shape != v.shape:
        raise ValueError(
            f"k and v must have same shape. Got k={k.shape}, v={v.shape}"
        )

    if h_init.shape != (k.shape[0], k.shape[1], k.shape[3]):
        raise ValueError(
            f"h_init must have shape (B, H, D). Got h_init={h_init.shape}, expected {(k.shape[0], k.shape[1], k.shape[3])}"
        )

    B, H, T, D = k.shape

    # BUG FIX #6: Reject D > 64 instead of silently truncating
    if D > 64:
        raise ValueError(
            f"gated_rnn_update requires D <= 64, got D={D}. "
            "This is a Triton kernel limitation. "
            "Consider using a smaller hidden dimension or projecting to 64."
        )

    # Allocate output
    h_out = torch.empty(B, H, T, D, dtype=k.dtype, device=k.device)

    # Grid: (B*H,) — one program per (batch, head)
    grid = (B * H,)

    # Get strides
    stride_k_b, stride_k_h, stride_k_t, stride_k_d = k.stride()
    stride_v_b, stride_v_h, stride_v_t, stride_v_d = v.stride()
    stride_h_b, stride_h_h, stride_h_d = h_init.stride()
    stride_ho_b, stride_ho_h, stride_ho_t, stride_ho_d = h_out.stride()

    # Launch kernel (no weight matrices; k, v are already projections from attention)
    gated_rnn_fwd_kernel[grid](
        k, v, h_init, h_out,
        stride_k_b, stride_k_h, stride_k_t, stride_k_d,
        stride_v_b, stride_v_h, stride_v_t, stride_v_d,
        stride_h_b, stride_h_h, stride_h_d,
        stride_ho_b, stride_ho_h, stride_ho_t, stride_ho_d,
        T=T, D=D, H=H, BLOCK_D=min(D, 64),
    )

    # h_final is the last hidden state
    h_final = h_out[:, :, -1, :].clone()

    return h_out, h_final


@torch.library.register_fake("sisyphus::gated_rnn_update")
def _(k, v, h_init):
    """Fake implementation for shape inference."""
    B, H, T, D = k.shape
    h_out = k.new_empty(B, H, T, D)
    h_final = k.new_empty(B, H, D)
    return h_out, h_final


def _gru_backward(ctx, grad_h_out, grad_h_final):
    """Backward pass: compute dk, dv from grads."""
    k, v, h_init, h_out = ctx.saved_tensors
    B, H, T, D = k.shape

    # Allocate gradients
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dh_init = torch.zeros_like(h_init)

    # Combine grad_h_out and grad_h_final
    grad_h_total = grad_h_out.clone()
    grad_h_total[:, :, -1, :] += grad_h_final

    # Grid: (B*H,)
    grid = (B * H,)

    # Get strides
    stride_k_b, stride_k_h, stride_k_t, stride_k_d = k.stride()
    stride_v_b, stride_v_h, stride_v_t, stride_v_d = v.stride()
    stride_h_b, stride_h_h, stride_h_d = h_init.stride()
    stride_ho_b, stride_ho_h, stride_ho_t, stride_ho_d = h_out.stride()
    stride_dh_b, stride_dh_h, stride_dh_t, stride_dh_d = grad_h_total.stride()
    stride_dk_b, stride_dk_h, stride_dk_t, stride_dk_d = dk.stride()
    stride_dv_b, stride_dv_h, stride_dv_t, stride_dv_d = dv.stride()
    stride_dhi_b, stride_dhi_h, stride_dhi_d = dh_init.stride()

    # Launch backward kernel
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

    return dk, dv, dh_init


def _gru_setup_context(ctx, inputs, output):
    """Save tensors for backward."""
    k, v, h_init = inputs
    h_out, h_final = output
    ctx.save_for_backward(k, v, h_init, h_out)


torch.library.register_autograd(
    "sisyphus::gated_rnn_update",
    _gru_backward,
    setup_context=_gru_setup_context,
)
