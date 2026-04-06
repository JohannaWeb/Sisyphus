"""Triton kernel for local window causal attention (forward pass) - FIXED VERSION.

This version properly covers the full key range for all query rows using
multi-tile key iteration with online softmax accumulation (FlashAttention-2 style).
Uses BLOCK_N=64 to fit within SRAM constraints while supporting full causal coverage.
"""

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
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Local window causal attention kernel (fixed multi-tile version).

    Query at position i can attend to keys j where: j in [max(0, i-W+1), i]

    CRITICAL: This kernel iterates over multiple key tiles to ensure FULL coverage
    of the valid key range for every query row. Uses online softmax to accumulate
    attention across tiles.

    With BLOCK_N=64, BLOCK_M=32, W=128: covers [q-127, q+31], ~159 keys per query tile.
    Requires ~3 key tiles per query tile, all fitting in SRAM.

    Args:
        Q, K, V: (B, H, T, D) tensors
        O: output (B, H, T, D)
        LSE: log-sum-exp (B, H, T) for backward stability
        H: number of heads
        W: window size (typically 128)
        BLOCK_M: query tile size (32)
        BLOCK_N: key tile size (64) - reduced to fit SRAM with multi-tile loop
        BLOCK_D: head dim (64)
    """
    pid_m = tl.program_id(0)  # query tile index
    pid_bh = tl.program_id(1)  # batch*head index

    # Unpack batch and head
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query tile range
    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_d = tl.arange(0, BLOCK_D)  # (BLOCK_D,)

    # Load Q tile: (BLOCK_M, BLOCK_D)
    q_ptrs = (
        Q_ptr
        + pid_b * stride_q_b
        + pid_h * stride_q_h
        + offs_m[:, None] * stride_q_t
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < T) & (offs_d[None, :] < D), other=0.0)

    # Key range that this query tile needs to cover
    # Query tile [q_start, q_start+BLOCK_M) needs keys [max(0, q_start-W+1), q_start+BLOCK_M-1]
    k_min = tl.maximum(0, q_start - W + 1)
    k_max = tl.minimum(T, q_start + BLOCK_M)  # up to (not inclusive) q_start+BLOCK_M for causal

    # Online softmax state
    scale = (D ** -0.5)
    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)  # row max
    s = tl.zeros((BLOCK_M,), dtype=tl.float32)  # row sum
    out = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)  # output accumulator

    # Iterate over key tiles: [k_min, k_min+BLOCK_N), [k_min+BLOCK_N, ...), until >= k_max
    num_key_tiles = (k_max - k_min + BLOCK_N - 1) // BLOCK_N
    for k_tile_idx in tl.range(num_key_tiles):
        k_start = k_min + k_tile_idx * BLOCK_N
        k_end = tl.minimum(k_start + BLOCK_N, k_max)

        # Offset for this key tile
        offs_n = k_start + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

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
        scores = tl.dot(q, k) * scale  # (BLOCK_M, BLOCK_N)

        # Apply causal + window mask
        # Query m can attend to key n iff: n <= m AND n >= m - W + 1
        causal_mask = offs_m[:, None] >= offs_n[None, :]  # (BLOCK_M, BLOCK_N)
        window_mask = (offs_m[:, None] - offs_n[None, :]) < W  # (BLOCK_M, BLOCK_N)
        valid_mask = causal_mask & window_mask & (offs_n[None, :] < T)  # (BLOCK_M, BLOCK_N)

        # Masked scores
        scores = tl.where(valid_mask, scores, float('-inf'))

        # Online softmax update (Algorithm 1 from FlashAttention-2)
        # Step 1: m_new = max(m_old, row_max(scores))
        m_new = tl.max(scores, axis=1)  # (BLOCK_M,)
        m_new = tl.maximum(m, m_new)

        # Step 2: Reweight old output and sum
        alpha = tl.exp(m - m_new)  # (BLOCK_M,) — scale factor for old state
        s = s * alpha  # reweight old sum

        # Step 3: Compute new exp scores and sum
        exp_scores = tl.exp(scores - m_new[:, None])  # (BLOCK_M, BLOCK_N)
        exp_scores = tl.where(valid_mask, exp_scores, 0.0)
        s_tile = tl.sum(exp_scores, axis=1)  # (BLOCK_M,) — sum of new scores
        s = s + s_tile

        # Step 4: Reweight old output and add new contribution
        out = out * alpha[:, None] + tl.dot(exp_scores, tl.trans(v))

        # Step 5: Update max
        m = m_new

    # Finalize output
    m = tl.where(offs_m < T, m, float('-inf'))
    m = tl.where(m == float('-inf'), 0.0, m)

    # Compute LSE for backward stability
    lse = m + tl.log(s + 1e-6)

    # Normalize output
    out = out / (s[:, None] + 1e-6)

    # Store output
    o_ptrs = (
        O_ptr
        + pid_b * stride_o_b
        + pid_h * stride_o_h
        + offs_m[:, None] * stride_o_t
        + offs_d[None, :] * stride_o_d
    )
    tl.store(o_ptrs, out, mask=(offs_m[:, None] < T) & (offs_d[None, :] < D))

    # Store LSE
    lse_ptrs = (
        LSE_ptr
        + pid_b * stride_lse_b
        + pid_h * stride_lse_h
        + offs_m * stride_lse_t
    )
    tl.store(lse_ptrs, lse, mask=offs_m < T)
