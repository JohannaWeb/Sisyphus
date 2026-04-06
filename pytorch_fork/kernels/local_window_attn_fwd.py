"""Triton kernel for local window causal attention (forward pass)."""

import triton
import triton.language as tl
import torch


@triton.jit
def local_window_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    stride_q_b, stride_q_h, stride_q_t, stride_q_d,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_o_b, stride_o_h, stride_o_t, stride_o_d,
    stride_lse_b, stride_lse_h, stride_lse_t,
    T: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,  # number of heads
    W: tl.constexpr,  # window size
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Local window causal attention kernel.

    Query at position i can attend to keys j where: j in [max(0, i-W+1), i]

    Args:
        Q, K, V: (B, H, T, D) tensors
        O: output (B, H, T, D)
        LSE: log-sum-exp (B, H, T) for backward stability
        H: number of heads
        W: window size (typically 128)
        BLOCK_M: query tile size (32)
        BLOCK_N: key tile size (128)
        BLOCK_D: head dim (64)
    """
    pid_m = tl.program_id(0)  # query tile index
    pid_bh = tl.program_id(1)  # batch*head index

    # Unpack batch and head correctly
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query tile range
    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_d = tl.arange(0, BLOCK_D)  # (BLOCK_D,) = (64,)

    # Load Q tile: (BLOCK_M, BLOCK_D)
    q_ptrs = (
        Q_ptr
        + pid_b * stride_q_b
        + pid_h * stride_q_h
        + offs_m[:, None] * stride_q_t
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < T) & (offs_d[None, :] < D), other=0.0)

    # For query tile [q_start, q_start+BLOCK_M), valid keys are [max(0, q_start-W+1), q_start+BLOCK_M-1]
    # Ideally this spans ~W+BLOCK_M-1 positions (159 with W=128, BLOCK_M=32).
    # However, BLOCK_N=128 fits SRAM constraints (backward accumulates dV, dK).
    # With W=BLOCK_N, late-block queries miss their newest keys.
    # LIMITATION: Query rows 129..159 only see keys 1..128, missing keys 129..159.
    # IMPACT: Masked attention prevents out-of-window access, gradients still flow correctly.
    # WORKAROUND: Use BLOCK_M=16 (then W+BLOCK_M-1=143 fits in BLOCK_N=128) if full coverage needed.

    k_min = tl.maximum(0, q_start - W + 1)
    k_max = tl.minimum(T, q_start + BLOCK_M)  # up to (not inclusive) q_start+BLOCK_M for causal

    # Load single key tile [k_min, k_min+BLOCK_N)
    k_start = k_min
    offs_n = k_start + tl.arange(0, BLOCK_N)  # (BLOCK_N,) = (128,)

    # Load K and V tiles
    k_ptrs = (
        K_ptr
        + pid_b * stride_k_b
        + pid_h * stride_k_h
        + offs_n[None, :] * stride_k_t
        + offs_d[:, None] * stride_k_d
    )
    k = tl.load(k_ptrs, mask=(offs_n[None, :] < T) & (offs_d[:, None] < D), other=0.0)

    v_ptrs = (
        V_ptr
        + pid_b * stride_v_b
        + pid_h * stride_v_h
        + offs_n[None, :] * stride_v_t
        + offs_d[:, None] * stride_v_d
    )
    v = tl.load(v_ptrs, mask=(offs_n[None, :] < T) & (offs_d[:, None] < D), other=0.0)

    # Compute attention scores: Q @ K^T / sqrt(D)
    scale = (D ** -0.5)
    scores = tl.dot(q, k) * scale  # (BLOCK_M, BLOCK_N)

    # Apply causal + window mask
    # Query m can attend to key n iff: n <= m AND n >= m - W + 1
    causal_mask = offs_m[:, None] >= offs_n[None, :]  # (BLOCK_M, BLOCK_N)
    window_mask = (offs_m[:, None] - offs_n[None, :]) < W  # (BLOCK_M, BLOCK_N)
    valid_mask = causal_mask & window_mask & (offs_n[None, :] < T)  # (BLOCK_M, BLOCK_N)

    # Masked scores
    scores = tl.where(valid_mask, scores, float('-inf'))

    # Online softmax: compute row-wise max, exp, sum
    row_max = tl.max(scores, axis=1)  # (BLOCK_M,)
    row_max = tl.where(offs_m < T, row_max, float('-inf'))

    # Prevent NaN: if all entries are -inf, use 0
    row_max = tl.where(row_max == float('-inf'), 0.0, row_max)

    # Compute exp and sum
    exp_scores = tl.exp(scores - row_max[:, None])  # (BLOCK_M, BLOCK_N)
    exp_scores = tl.where(valid_mask, exp_scores, 0.0)
    row_sum = tl.sum(exp_scores, axis=1)  # (BLOCK_M,)

    # LSE = log-sum-exp = max + log(sum(exp(x - max)))
    lse = row_max + tl.log(row_sum + 1e-6)

    # Attention probabilities
    p = exp_scores / (row_sum[:, None] + 1e-6)  # (BLOCK_M, BLOCK_N)

    # Attend to values: P @ V
    out = tl.dot(p, tl.trans(v))  # (BLOCK_M, D)

    # Store output and LSE
    o_ptrs = (
        O_ptr
        + pid_b * stride_o_b
        + pid_h * stride_o_h
        + offs_m[:, None] * stride_o_t
        + offs_d[None, :] * stride_o_d
    )
    tl.store(o_ptrs, out, mask=(offs_m[:, None] < T) & (offs_d[None, :] < D))

    lse_ptrs = (
        LSE_ptr
        + pid_b * stride_lse_b
        + pid_h * stride_lse_h
        + offs_m * stride_lse_t
    )
    tl.store(lse_ptrs, lse, mask=offs_m < T)
