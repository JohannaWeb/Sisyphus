import re

with open("src/model.py", "r") as f:
    content = f.read()

new_class = '''class KVCache:
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

'''

content = re.sub(r'class KVCache:.*?class FractalAttention', new_class + 'class FractalAttention', content, flags=re.DOTALL)

with open("src/model.py", "w") as f:
    f.write(content)
