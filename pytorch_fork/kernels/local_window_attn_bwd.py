"""Triton kernel for local window causal attention (backward pass)."""

import triton
import triton.language as tl
import torch


@triton.jit
def local_window_attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr, dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_q_b, stride_q_h, stride_q_t, stride_q_d,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_v_b, stride_v_h, stride_v_t, stride_v_d,
    stride_o_b, stride_o_h, stride_o_t, stride_o_d,
    stride_lse_b, stride_lse_h, stride_lse_t,
    stride_dq_b, stride_dq_h, stride_dq_t, stride_dq_d,
    stride_dk_b, stride_dk_h, stride_dk_t, stride_dk_d,
    stride_dv_b, stride_dv_h, stride_dv_t, stride_dv_d,
    stride_do_b, stride_do_h, stride_do_t, stride_do_d,
    T: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Backward pass for local window attention.

    Recomputes attention probabilities from stored LSE and O.
    Computes dQ, dK, dV from dO.

    Uses FlashAttention-2 backward style: recompute P rather than storing it.
    """
    pid_n = tl.program_id(0)  # key tile index
    pid_bh = tl.program_id(1)

    # Unpack batch and head
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    k_start = pid_n * BLOCK_N
    offs_n = k_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Load K, V tiles for this key block
    k_ptrs = (
        K_ptr
        + pid_b * stride_k_b
        + pid_h * stride_k_h
        + offs_n[None, :] * stride_k_t
        + offs_d[:, None] * stride_k_d
    )
    k = tl.load(k_ptrs, mask=(offs_n[None, :] < T), other=0.0)

    v_ptrs = (
        V_ptr
        + pid_b * stride_v_b
        + pid_h * stride_v_h
        + offs_n[None, :] * stride_v_t
        + offs_d[:, None] * stride_v_d
    )
    v = tl.load(v_ptrs, mask=(offs_n[None, :] < T), other=0.0)

    # Accumulator for dV and dK
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    # Query tiles that can attend to this key block
    # Query i attends to keys in [i-W+1, i], so i in [k_start, k_start+W-1]
    # For all key positions in [k_start, k_start+BLOCK_N), the query range is:
    # i_min = k_start (smallest query that attends to k_start)
    # i_max = min(T, k_start + W + BLOCK_N - 1)
    q_start_min = k_start
    q_start_max = tl.minimum(T, k_start + W + BLOCK_N)

    scale = (D ** -0.5)

    for q_start in tl.range(q_start_min, q_start_max, BLOCK_M):
        offs_m = q_start + tl.arange(0, BLOCK_M)

        # Load Q, dO, O, LSE for this query block
        q_ptrs = (
            Q_ptr
            + pid_b * stride_q_b
            + pid_h * stride_q_h
            + offs_m[:, None] * stride_q_t
            + offs_d[None, :] * stride_q_d
        )
        q = tl.load(q_ptrs, mask=(offs_m[:, None] < T), other=0.0)

        do_ptrs = (
            dO_ptr
            + pid_b * stride_do_b
            + pid_h * stride_do_h
            + offs_m[:, None] * stride_do_t
            + offs_d[None, :] * stride_do_d
        )
        do = tl.load(do_ptrs, mask=(offs_m[:, None] < T), other=0.0)

        o_ptrs = (
            O_ptr
            + pid_b * stride_o_b
            + pid_h * stride_o_h
            + offs_m[:, None] * stride_o_t
            + offs_d[None, :] * stride_o_d
        )
        o = tl.load(o_ptrs, mask=(offs_m[:, None] < T), other=0.0)

        lse_ptrs = (
            LSE_ptr
            + pid_b * stride_lse_b
            + pid_h * stride_lse_h
            + offs_m * stride_lse_t
        )
        lse = tl.load(lse_ptrs, mask=offs_m < T, other=0.0)  # (BLOCK_M,)

        # Recompute attention scores and softmax
        scores = tl.dot(q, k) * scale  # (BLOCK_M, BLOCK_N)

        # Apply causal + window mask
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        window_mask = (offs_m[:, None] - offs_n[None, :]) < W
        valid_mask = causal_mask & window_mask & (offs_n[None, :] < T)

        scores = tl.where(valid_mask, scores, float('-inf'))

        # Recompute probabilities: P = exp(scores - LSE)
        p = tl.exp(scores - lse[:, None])
        p = tl.where(valid_mask, p, 0.0)

        # dV += P^T @ dO
        # p: (BLOCK_M, BLOCK_N), do: (BLOCK_M, BLOCK_D)
        # P^T @ dO: (BLOCK_N, BLOCK_M) @ (BLOCK_M, BLOCK_D) = (BLOCK_N, BLOCK_D)
        dv_acc += tl.dot(tl.trans(p), do)

        # D_i = rowsum(dO * O) — diagonal correction for softmax
        delta = tl.sum(do * o, axis=1)  # (BLOCK_M,)

        # dP = dO @ V^T
        # do: (BLOCK_M, BLOCK_D), v: (BLOCK_D, BLOCK_N)
        # dO @ V^T: (BLOCK_M, BLOCK_D) @ (BLOCK_N, BLOCK_D).T = (BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_N) = (BLOCK_M, BLOCK_N)
        dp = tl.dot(do, v)  # (BLOCK_M, BLOCK_N)

        # dS = P * (dP - delta[:, None])  (gradient w.r.t. scores)
        ds = p * (dp - delta[:, None])

        # dK += dS^T @ Q
        dk_acc += tl.dot(tl.trans(ds), q) * scale

        # dQ: accumulate via atomic_add (one write per query position per key block)
        # ds: (BLOCK_M, BLOCK_N), k: (BLOCK_D, BLOCK_N)
        # dQ = dS @ K^T: (BLOCK_M, BLOCK_N) @ (BLOCK_N, BLOCK_D) = (BLOCK_M, BLOCK_D)
        dq = tl.dot(ds, tl.trans(k)) * scale  # (BLOCK_M, BLOCK_D)
        dq_ptrs = (
            dQ_ptr
            + pid_b * stride_dq_b
            + pid_h * stride_dq_h
            + offs_m[:, None] * stride_dq_t
            + offs_d[None, :] * stride_dq_d
        )
        tl.atomic_add(dq_ptrs, dq, mask=(offs_m[:, None] < T) & (offs_d[None, :] < D))

    # Store accumulated dV and dK
    dv_ptrs = (
        dV_ptr
        + pid_b * stride_dv_b
        + pid_h * stride_dv_h
        + offs_n[:, None] * stride_dv_t
        + offs_d[None, :] * stride_dv_d
    )
    tl.store(dv_ptrs, dv_acc, mask=(offs_n[:, None] < T) & (offs_d[None, :] < D))

    dk_ptrs = (
        dK_ptr
        + pid_b * stride_dk_b
        + pid_h * stride_dk_h
        + offs_n[:, None] * stride_dk_t
        + offs_d[None, :] * stride_dk_d
    )
    tl.store(dk_ptrs, dk_acc, mask=(offs_n[:, None] < T) & (offs_d[None, :] < D))
