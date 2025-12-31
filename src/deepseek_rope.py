import torch
from typing import Tuple, cast

class RoPE(torch.nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        
        # Inititalize cache placeholders
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._cached_seq_len = 0

        # Build initial cache
        self._build_cache()

    def _build_cache(self):
        """Pre-build cache for max_seq_len"""
        t = torch.arange(0, self.max_seq_len).to(device=cast(torch.device, self.inv_freq.device), dtype=torch.float32)
        freqs = torch.outer(t, cast(torch.Tensor, self.inv_freq).to(torch.float32))
        emb = torch.cat([freqs, freqs], dim=-1)

        # Update buffers (can't use register_buffer after init, so direct assign)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        self._cached_seq_len = self.max_seq_len

    def forward(
        self,
        seq_len: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (cos, sin) for positions [0..seq_len-1].
        Shapes: (seq_len, dim). These broadcast against q/k shaped (..., seq_len, dim).
        """
        seq_len_int: int = int(seq_len)

        if seq_len_int > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len_int} is longer than max_seq_len {self.max_seq_len}. "
                "Increase max_seq_len when constructing RoPE."
            )

        device_: torch.device = (
            cast(torch.device, self.inv_freq.device) if device is None else torch.device(device)
        )
        # For numerical stability, compute angles in fp32 and cast at the end.
        out_dtype = dtype if dtype is not None else torch.get_default_dtype() 

        needs_refresh = (
            self.cos_cached is None
            or self.sin_cached is None
            or self._cached_seq_len < seq_len_int
            or self.cos_cached.device != device_
        )

        if needs_refresh:
            t = torch.arange(0, seq_len_int).to(device=device_, dtype=torch.float32)
            freqs = torch.outer(t, cast(torch.Tensor, self.inv_freq).to(device=device_, dtype=torch.float32))
            emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
            cos = emb.cos().to(dtype=out_dtype)
            sin = emb.sin().to(dtype=out_dtype)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
            self._cached_seq_len = seq_len
        else:
            cos = self.cos_cached[:seq_len_int].to(dtype=out_dtype)
            sin = self.sin_cached[:seq_len_int].to(dtype=out_dtype)

        return cos, sin

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
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q and k must have same last dim, got {q.shape[-1]} vs {k.shape[-1]}")
    if q.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE requires even last dim, got {q.shape[-1]}")

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