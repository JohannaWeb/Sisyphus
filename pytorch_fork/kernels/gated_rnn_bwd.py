"""Triton kernel for GRU backward pass (reverse-time scan)."""

import triton
import triton.language as tl
import torch


@triton.jit
def gated_rnn_bwd_kernel(
    K_ptr, V_ptr, H_out_ptr, dH_out_ptr,
    dK_ptr, dV_ptr, dH_init_ptr,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_h_b, stride_h_h, stride_h_d,
    stride_dh_b, stride_dh_h, stride_dh_t, stride_dh_d,
    stride_dk_b, stride_dk_h, stride_dk_t, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_t, stride_dv_d,
    T: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Simplified gated-RNN backward kernel: reverse-time scan.

    Each program scans backward through T tokens, accumulating gradients.
    No weight matrices to differentiate (K, V are inputs from attention module).

    Args:
        K, V: (B, H, T, D) input sequences
        H_out: (B, H, T, D) forward activations
        dH_out: (B, H, T, D) gradient w.r.t. outputs
        dK, dV: output gradients
        dH_init: gradient w.r.t. initial state
        H: number of heads
    """
    pid = tl.program_id(0)

    # Unpack batch and head
    pid_b = pid // H
    pid_h = pid % H

    # Initialize dh accumulator (will flow back through time)
    dh = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Reverse loop through time
    for t_idx in tl.range(T):
        t = T - 1 - t_idx  # reverse time

        # Load h_t, h_{t-1}
        h_t_ptrs = (
            H_out_ptr
            + pid_b * stride_h_b
            + pid_h * stride_h_h
            + t * stride_dh_t
            + tl.arange(0, BLOCK_D) * stride_dh_d
        )
        h_t = tl.load(h_t_ptrs, mask=tl.arange(0, BLOCK_D) < D, other=0.0)

        h_prev_ptrs = (
            H_out_ptr
            + pid_b * stride_h_b
            + pid_h * stride_h_h
            + tl.maximum(0, t - 1) * stride_dh_t
            + tl.arange(0, BLOCK_D) * stride_dh_d
        )
        h_prev = tl.load(h_prev_ptrs, mask=tl.arange(0, BLOCK_D) < D, other=0.0)

        # Load k_t, v_t
        k_ptrs = (
            K_ptr
            + pid_b * stride_k_b
            + pid_h * stride_k_h
            + t * stride_k_t
            + tl.arange(0, BLOCK_D) * stride_k_d
        )
        k_t = tl.load(k_ptrs, mask=tl.arange(0, BLOCK_D) < D, other=0.0)

        v_ptrs = (
            V_ptr
            + pid_b * stride_v_b
            + pid_h * stride_v_h
            + t * stride_v_t
            + tl.arange(0, BLOCK_D) * stride_v_d
        )
        v_t = tl.load(v_ptrs, mask=tl.arange(0, BLOCK_D) < D, other=0.0)

        # Load dH_out_t (gradient from downstream)
        dh_out_ptrs = (
            dH_out_ptr
            + pid_b * stride_dho_b
            + pid_h * stride_dho_h
            + t * stride_dho_t
            + tl.arange(0, BLOCK_D) * stride_dho_d
        )
        dh_out_t = tl.load(dh_out_ptrs, mask=tl.arange(0, BLOCK_D) < D, other=0.0)

        # Accumulate gradient from downstream + recurrent
        dh_total = dh + dh_out_t  # (D,)

        # Recompute gates (simplified GRU)
        r_t = tl.sigmoid(h_prev + k_t)
        z_t = tl.sigmoid(h_prev + k_t)
        # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        x = r_t * h_prev + k_t
        e2x = tl.exp(2.0 * x)
        n_t = (e2x - 1.0) / (e2x + 1.0)

        # Backward through h_new = (1-z) * h_prev + z * (n * v)
        # dz = dh * (n*v - h_prev)
        dz = dh_total * (n_t * v_t - h_prev)

        # dn = dh * z * v
        dn = dh_total * z_t * v_t

        # dv = dh * z * n
        dv = dh_total * z_t * n_t

        # Backward through z = sigmoid(h_prev + k_t)
        dz_pre = dz * z_t * (1.0 - z_t)
        dh_prev_z = dz_pre  # from z
        dk_z = dz_pre

        # Backward through n = tanh(r*h_prev + k_t)
        dn_pre = dn * (1.0 - n_t * n_t)
        dr_h = dn_pre * h_prev
        dk_n = dn_pre

        # Backward through r = sigmoid(h_prev + k_t)
        dr = dr_h * h_prev
        dr_pre = dr * r_t * (1.0 - r_t)
        dh_prev_r = dr_pre
        dk_r = dr_pre

        # Total gradient w.r.t. h_prev (will propagate back)
        dh_prev_total = dh_total * (1.0 - z_t) + dh_prev_z + dh_prev_r
        dk_total = dk_z + dk_n + dk_r

        # Store dK, dV
        dk_ptrs = (
            dK_ptr
            + pid_b * stride_dk_b
            + pid_h * stride_dk_h
            + t * stride_dk_t
            + tl.arange(0, BLOCK_D) * stride_dk_d
        )
        tl.atomic_add(dk_ptrs, dk_total, mask=tl.arange(0, BLOCK_D) < D)

        dv_ptrs = (
            dV_ptr
            + pid_b * stride_dv_b
            + pid_h * stride_dv_h
            + t * stride_dv_t
            + tl.arange(0, BLOCK_D) * stride_dv_d
        )
        tl.atomic_add(dv_ptrs, dv, mask=tl.arange(0, BLOCK_D) < D)

        # Propagate gradient back through time
        dh = dh_prev_total

    # Store dH_init (gradient w.r.t. initial state)
    dh_init_ptrs = (
        dH_init_ptr
        + pid_b * stride_dh_b
        + pid_h * stride_dh_h
        + tl.arange(0, BLOCK_D) * stride_dh_d
    )
    tl.store(dh_init_ptrs, dh, mask=tl.arange(0, BLOCK_D) < D)
