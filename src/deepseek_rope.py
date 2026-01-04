"""
Rotary Position Embeddings (RoPE)
=================================

DeepSeek-V3 uses RoPE for positional encoding in the attention mechanism.
RoPE encodes position by rotating query/key vectors in 2D subspaces.

Key insight: Position information is encoded via rotation, so it naturally
handles relative positions - the dot product between q[i] and k[j] only
depends on (i-j), not the absolute positions.

In MLA, RoPE is applied only to the "rope" portion of Q and K, not the
"nope" (content) portion. This separation allows caching only the compressed
KV representation while preserving position-awareness.

### **Architecture Flow Verification**
```
Input Tokens [B, S]
       ↓
   embed_tokens (shared with MTP)
       ↓
   [B, S, hidden_dim]
       ↓
┌──────────────────────────────────────┐
│  Transformer Stack                    │
│  ├─ Layers 0..num_dense_layers-1     │
│  │   └─ MLA + Dense FFN              │
│  └─ Layers num_dense_layers..N-1     │
│      └─ MLA + MoE (shared + routed)  │
└──────────────────────────────────────┘
       ↓
   Final RMSNorm
       ↓
   lm_head (shared with MTP) → main_logits [B, S, V]
       ↓
   MTP Module (training only)
   ├─ Depth 1: hidden + embed(input[1:]) → transformer → lm_head → logits for +2
   └─ Depth 2: hidden + embed(input[2:]) → transformer → lm_head → logits for +3
       ↓
   compute_mtp_loss(main_logits, mtp_logits, shifted_labels)
"""

import torch
from typing import Tuple, cast

class RoPE(torch.nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert dim > 0 and dim % 2 == 0, f"RoPE dim must be positive and even, got {dim}"
        assert max_seq_len > 0, f"max_seq_len must be positive, got {max_seq_len}"
        assert base > 0, f"base must be positive, got {base}"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

        # Build initial cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Pre-build cache for max_seq_len"""
        t = torch.arange(seq_len).to(device=cast(torch.device, self.inv_freq.device), dtype=torch.float32)
        freqs = torch.outer(t, cast(torch.Tensor, self.inv_freq).to(torch.float32))
        emb = torch.cat([freqs, freqs], dim=-1)

        # Update buffers (can't use register_buffer after init, so direct assign)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()
        self._cached_seq_len = self.max_seq_len

    @property
    def cos_cached(self) -> torch.Tensor:
        return self._cos_cached

    @property
    def sin_cached(self) -> torch.Tensor:
        return self._sin_cached

    def forward(
        self,
        seq_len: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (cos, sin) for positions [0..seq_len-1].
        Shapes: (seq_len, dim). These broadcast against q/k shaped (..., seq_len, dim).
        """
        assert seq_len > 0, f"seq_len must be positive, got {seq_len}"
        seq_len_int: int = int(seq_len)

        if seq_len_int > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len_int} is longer than max_seq_len {self.max_seq_len}. "
                "Increase max_seq_len when constructing RoPE."
            )

        # Rebuild cache if needed (larger sequence or different device)
        target_device = torch.device(device) if device is not None else cast(torch.device, self.inv_freq.device)
        needs_rebuild = (
            seq_len > self._cached_seq_len
            or self.cos_cached.device != target_device
        )
        
        if needs_rebuild:
            # Rebuild for the larger of current need or cached size
            new_len = max(seq_len, self._cached_seq_len)
            self._build_cache(new_len)
            
            # Move to target device if different
            if str(self.cos_cached.device) != str(target_device):
                self._cos_cached = self._cos_cached.to(device=target_device)
                self._sin_cached = self._sin_cached.to(device=target_device)
        
        # Slice and cast to requested dtype
        out_dtype = dtype if dtype is not None else torch.get_default_dtype()
        cos = self.cos_cached[:seq_len].to(dtype=out_dtype)
        sin = self.sin_cached[:seq_len].to(dtype=out_dtype)
        
        return cos, sin

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_seq_len={self.max_seq_len}, base={self.base}"

def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to q and k.
    Expected:
      - q, k: (..., seq_len, dim)
      - cos, sin: (seq_len, dim) or broadcastable to q/k
    Returns:
      - (q_rot, k_rot) with same shape as inputs
    """
    assert q.size(-1) == k.size(-1), f"q/k dim mismatch: {q.size(-1)} vs {k.size(-1)}"
    assert q.size(-1) % 2 == 0, f"RoPE requires even dim, got {q.size(-1)}"

    # Make cos/sin broadcast to (..., seq_len, dim)
    while cos.ndim < q.ndim:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    
    return q_rot, k_rot

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Split on the last dimension.
    d = x.size(-1)
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)