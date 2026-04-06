"""Triton kernel for GRU forward pass (sequential scan)."""

import triton
import triton.language as tl
import torch


@triton.jit
def gated_rnn_fwd_kernel(
    K_ptr, V_ptr, H_init_ptr, H_out_ptr,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_h_b, stride_h_h, stride_h_d,
    stride_ho_b, stride_ho_h, stride_ho_t, stride_ho_d,
    T: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Simplified gated-RNN forward kernel: sequential scan over T tokens.

    Each program (batch, head) scans through all T tokens serially.
    Stores h_t at each step for BPTT (backward pass).

    K, V are already projections from the attention module; no separate weight matrices.

    Recurrence per token:
        r_t = sigmoid(k_t + h_{t-1})
        z_t = sigmoid(k_t + h_{t-1})
        n_t = tanh(r_t * h_{t-1} + k_t)
        h_t = (1-z_t) * h_{t-1} + z_t * (n_t * v_t)

    Args:
        K, V: (B, H, T, D) input sequences (from attention module)
        H_init: (B, H, D) initial hidden state
        H_out: (B, H, T, D) output hidden states
        T: sequence length
        D: hidden dimension
        H: number of heads
    """
    pid = tl.program_id(0)  # program per (batch, head)

    # Unpack batch and head from linearized index
    pid_b = pid // H
    pid_h = pid % H

    # Load initial hidden state
    h_init_ptrs = (
        H_init_ptr
        + pid_b * stride_h_b
        + pid_h * stride_h_h
        + tl.arange(0, BLOCK_D) * stride_h_d
    )
    h = tl.load(h_init_ptrs, mask=tl.arange(0, BLOCK_D) < D, other=0.0)  # (D,)

    # Sequential loop through time
    for t in tl.range(0, T):
        # Load k_t, v_t
        k_ptrs = (
            K_ptr
            + pid_b * stride_k_b
            + pid_h * stride_k_h
            + t * stride_k_t
            + tl.arange(0, BLOCK_D) * stride_k_d
        )
        k_t = tl.load(k_ptrs, mask=tl.arange(0, BLOCK_D) < D, other=0.0)  # (D,)

        v_ptrs = (
            V_ptr
            + pid_b * stride_v_b
            + pid_h * stride_v_h
            + t * stride_v_t
            + tl.arange(0, BLOCK_D) * stride_v_d
        )
        v_t = tl.load(v_ptrs, mask=tl.arange(0, BLOCK_D) < D, other=0.0)  # (D,)

        # GRU step: h_t = GRU(h_{t-1}, k_t, v_t)
        # r_t = sigmoid(Wr @ h + Ur @ k)
        # Simple matrix-vector multiply inline (since D=64, can do it per-thread)
        # For now, approximate with a simplified GRU without actual matrix multiplies
        # (full implementation would need explicit matmul)

        # Simplified GRU (no weights for now, just state propagation)
        # r_t = sigmoid(h + k)
        r_t = tl.sigmoid(h + k_t)  # (D,)

        # z_t = sigmoid(h + k)
        z_t = tl.sigmoid(h + k_t)  # (D,)

        # n_t = tanh(r * h + k)
        # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        x = r_t * h + k_t
        e2x = tl.exp(2.0 * x)
        n_t = (e2x - 1.0) / (e2x + 1.0)  # (D,)

        # h_new = (1-z) * h + z * (n * v)
        h_new = (1.0 - z_t) * h + z_t * (n_t * v_t)  # (D,)

        # Store h_t to H_out
        h_out_ptrs = (
            H_out_ptr
            + pid_b * stride_ho_b
            + pid_h * stride_ho_h
            + t * stride_ho_t
            + tl.arange(0, BLOCK_D) * stride_ho_d
        )
        tl.store(h_out_ptrs, h_new, mask=tl.arange(0, BLOCK_D) < D)

        h = h_new
