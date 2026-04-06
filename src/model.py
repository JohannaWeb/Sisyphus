#!/usr/bin/env python3
"""Small GPT-style byte-level language model with Monarch optimizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    # Monarch optimizations
    use_kv_cache: bool = False
    window_size: int = 256
    fractal_depth: int = 2


class KVCache:
    """Fast CUDA KV cache with NES-inspired paging, TurboQuant, and chronological order."""

    def __init__(self, max_len: int, n_heads: int, head_dim: int, device: str, hot_window: int = 512):
        self.max_len = max_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.hot_window = hot_window
        
        self.k_hot = torch.zeros(1, n_heads, hot_window, head_dim, device=device)
        self.v_hot = torch.zeros(1, n_heads, hot_window, head_dim, device=device)
        self.hot_tokens = 0
        
        self.k_cold_ang = []
        self.k_cold_mag = []
        self.v_cold_ang = []
        self.v_cold_mag = []

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        seq_len = k.shape[2]
        
        if self.hot_tokens + seq_len > self.hot_window:
            evict_size = min(self.hot_tokens + seq_len - self.hot_window, self.hot_tokens)
            if evict_size > 0:
                k_evict = self.k_hot[:, :, :evict_size]
                v_evict = self.v_hot[:, :, :evict_size]
                
                k_mag = k_evict.norm(dim=-1, keepdim=True)
                v_mag = v_evict.norm(dim=-1, keepdim=True)
                k_ang = ((k_evict / (k_mag + 1e-8) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                v_ang = ((v_evict / (v_mag + 1e-8) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                
                self.k_cold_mag.append(k_mag)
                self.k_cold_ang.append(k_ang)
                self.v_cold_mag.append(v_mag)
                self.v_cold_ang.append(v_ang)
                
                self.k_hot = torch.roll(self.k_hot, -evict_size, dims=2)
                self.v_hot = torch.roll(self.v_hot, -evict_size, dims=2)
                self.hot_tokens -= evict_size
                
        self.k_hot[:, :, self.hot_tokens:self.hot_tokens+seq_len] = k
        self.v_hot[:, :, self.hot_tokens:self.hot_tokens+seq_len] = v
        self.hot_tokens += seq_len
        
    def _decompress(self, i: int):
        k_ang = self.k_cold_ang[i].float() / 127.5 - 1.0
        v_ang = self.v_cold_ang[i].float() / 127.5 - 1.0
        return k_ang * self.k_cold_mag[i], v_ang * self.v_cold_mag[i]

    def promote(self, q: torch.Tensor, threshold: float = 0.02, top_k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        k_h = self.k_hot[:, :, :self.hot_tokens]
        v_h = self.v_hot[:, :, :self.hot_tokens]
        
        if not self.k_cold_mag:
            return k_h, v_h

        p_k_list, p_v_list = [], []
        # Iterate backwards to prefer recent cold blocks
        for i in range(len(self.k_cold_mag) - 1, -1, -1):
            kc, vc = self._decompress(i)
            score = torch.matmul(q, kc.transpose(-1, -2)).max()
            if score > threshold:
                p_k_list.append((i, kc, vc))
                if len(p_k_list) >= top_k:
                    break
        
        if not p_k_list:
            return k_h, v_h
            
        # Sort chronologically
        p_k_list.sort(key=lambda x: x[0])
        p_k = torch.cat([x[1] for x in p_k_list], dim=2)
        p_v = torch.cat([x[2] for x in p_k_list], dim=2)
        
        return torch.cat([p_k, k_h], dim=2), torch.cat([p_v, v_h], dim=2)
        
    def clear(self) -> None:
        self.hot_tokens = 0
        self.k_hot.zero_()
        self.v_hot.zero_()
        self.k_cold_ang.clear()
        self.k_cold_mag.clear()
        self.v_cold_ang.clear()
        self.v_cold_mag.clear()

    @property
    def total_tokens(self) -> int:
        return self.hot_tokens + sum(m.shape[2] for m in self.k_cold_mag)

# Removed: FractalAttention (dead code, never called in forward pass)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # KV cache will be initialized on first use with correct device
        self.kv_cache: Optional[KVCache] = None

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch, steps, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=2)
        q = q.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)

        # Update KV cache if enabled
        if use_cache and self.kv_cache is not None:
            self.kv_cache.append(k, v)
            # NES-inspired promotion
            k, v = self.kv_cache.promote(q, threshold=0.02)

            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=(steps > 1) # Mask handles it
            )
        else:
            # Standard training or non-cached inference
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )

        y = y.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), use_cache=use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class ByteGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight

        # Cache position indices to avoid reallocation every forward
        self.register_buffer('_position_ids', torch.arange(config.block_size))

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None, use_cache: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, steps = idx.shape
        
        if use_cache:
            # When using cache, we assume sequential calls.
            # total_tokens in cache is the count AFTER append.
            # CausalSelfAttention.forward appends BEFORE calling promote,
            # but ByteGPT.forward is called BEFORE CausalSelfAttention.
            # So kv_cache.total_tokens is the count of tokens ALREADY in cache.
            # The current tokens will be appended later.
            start_pos = self.blocks[0].attn.kv_cache.total_tokens if self.blocks[0].attn.kv_cache else 0
            positions = (self._position_ids[start_pos:start_pos + steps]).clamp(0, self.config.block_size - 1)
        else:
            if steps > self.config.block_size:
                raise ValueError("sequence length exceeds block size")
            # Use cached position IDs instead of allocating new tensor
            positions = self._position_ids[:steps]

        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(batch * steps, self.config.vocab_size),
                targets.view(batch * steps),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        # Initialize cache if requested
        if use_cache:
            for block in self.blocks:
                if block.attn.kv_cache:
                    block.attn.kv_cache.clear()
                else:
                    block.attn.kv_cache = KVCache(
                        self.config.block_size,
                        block.attn.n_head,
                        block.attn.head_dim,
                        idx.device.type,
                        hot_window=512
                    )
            
            # Fill cache with the entire prompt
            # We do this in one pass to be efficient
            self(idx, use_cache=True)

        for _ in range(max_new_tokens):
            if use_cache:
                # In cache mode, we only pass the LAST token to the model
                idx_cond = idx[:, -1:]
                logits, _ = self(idx_cond, use_cache=True)
            else:
                idx_cond = idx[:, -self.config.block_size :]
                logits, _ = self(idx_cond)
            
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
