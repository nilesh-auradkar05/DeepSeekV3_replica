"""
Multi-Head Latent Attention (MLA)
=================================

DeepSeek-V3's key innovation for efficient KV caching.

The core insight: Instead of caching full K and V tensors (expensive at scale),
MLA caches a compressed "latent" representation that can reconstruct K and V.

Architecture:
    Query path:  x -> compress -> [q_nope, q_rope] -> apply RoPE to q_rope -> concat
    Key path:    x -> compress_kv -> k_nope (up-project from c_kv)
                 x -> k_rope (direct, not from c_kv) -> apply RoPE
    Value path:  c_kv -> v (up-project)

Cache contents: (c_kv, k_rope) - compressed KV + rotary key component
    - c_kv: (batch, seq_len, kv_compress_dim) ~128-512 dims
    - k_rope: (batch, seq_len, qk_rope_head_dim) ~32-64 dims
    
This achieves ~57x compression vs standard MHA caching at 128K context.

Why k_rope from x directly (not c_kv)?
    - Position information shouldn't be entangled with content compression
    - Allows decoupled position encoding quality
    - Paper explicitly states this design choice
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from .deepseek_rope import rotate_half, RoPE
from utils.rms_norm import RMSNorm

class MultiHeadLatentAttention(torch.nn.Module):
    """
    Multi-Head Latent Attention with efficient KV compression.
    
    Key dimensions:
        - qk_nope_head_dim: Content (non-positional) dimension per head
        - qk_rope_head_dim: Positional (RoPE) dimension per head  
        - v_head_dim: Value dimension per head
        - kv_compress_dim: Compressed KV cache dimension (total, not per-head)
        - q_compress_dim: Query compression dimension
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Validate configuration
        assert config.hidden_dim > 0, "hidden_dim must be positive"
        assert config.num_attention_heads > 0, "num_attention_heads must be positive"
        assert config.kv_compress_dim > 0, "kv_compress_dim must be positive"
        assert config.q_compress_dim > 0, "q_compress_dim must be positive"
        
        self.hidden_dim = self.config.hidden_dim
        self.num_heads = self.config.num_attention_heads
        self.kv_compress_dim = self.config.kv_compress_dim
        self.q_compress_dim = self.config.q_compress_dim
        self.qk_nope_head_dim = self.config.qk_nope_head_dim
        self.qk_rope_head_dim = self.config.qk_rope_head_dim
        self.v_head_dim = self.config.v_head_dim
        
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        
        # scaling factor for attention scores
        self.scale = self.qk_head_dim ** -0.5 #(1 / sqrt(qk_head_dim))

        # Query Path
        # Compress: hidden_dim -> q_compress_dim
        self.q_down_proj = torch.nn.Linear(
            self.hidden_dim,
            self.q_compress_dim,
            bias=False,
        )
        self.q_norm = RMSNorm(self.q_compress_dim)

        self.q_nope_proj = torch.nn.Linear(
            self.q_compress_dim,
            self.num_heads * self.qk_nope_head_dim,
            bias=False,
        )

        self.q_rope_proj = torch.nn.Linear(
            self.q_compress_dim,
            self.num_heads * self.qk_rope_head_dim,
            bias=False,
        )

        # Key Path
        # Compress KV: hidden_dim -> kv_compress_dim
        self.kv_down_proj = torch.nn.Linear(
            self.hidden_dim,
            self.kv_compress_dim,
            bias=False,
        )
        self.kv_norm = RMSNorm(self.kv_compress_dim)

        self.k_nope_proj = torch.nn.Linear(
            self.kv_compress_dim,
            self.num_heads * self.qk_nope_head_dim,
            bias=False,
        )

        # Up-project V from compressed KV
        self.v_up_proj = torch.nn.Linear(
            self.kv_compress_dim,
            self.num_heads * self.v_head_dim,
            bias=False,
        )

        # K rope projection - directly from x
        self.k_rope_proj = torch.nn.Linear(
            self.hidden_dim,
            self.qk_rope_head_dim,
            bias=False,
        )

        # Output projection
        self.o_proj = torch.nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_dim,
            bias=False,
        )

        # RoPE module
        self.rope = RoPE(
            dim=self.qk_rope_head_dim,
            max_seq_len=self.config.max_seq_len,
            base=self.config.rope_base,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        """
        Forward pass with optional KV caching.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            attention_mask: Optional mask (batch, 1, seq_len, kv_seq_len)
            position_ids: Optional position indices (unused, positions inferred)
            past_key_value: Cached (c_kv, k_rope) from previous forward
            use_cache: Whether to return cache for next forward
        
        Returns:
            output: (batch, seq_len, hidden_dim)
            present_key_value: (c_kv, k_rope) if use_cache else None
        """
        batch_size, seq_len, hidden_dim = x.shape

        # Validate input
        assert hidden_dim == self.hidden_dim, \
            f"Input hidden_dim {hidden_dim} != expected {self.hidden_dim}"

        # 1. Query Path
        # 1.1 Compress query
        c_q = self.q_down_proj(x)  # (B, S, q_compress_dim)
        c_q = self.q_norm(c_q)
        
        # 1.2 Get q_nope (content)
        q_nope = self.q_nope_proj(c_q)  # (B, S, num_heads * qk_nope_head_dim)
        q_nope = q_nope.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim)
        q_nope = q_nope.transpose(1, 2)  # (B, H, S, qk_nope_head_dim)
        
        # 1.3 Get q_rope (position) 
        q_rope = self.q_rope_proj(c_q)  # (B, S, num_heads * qk_rope_head_dim)
        q_rope = q_rope.view(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim)
        q_rope = q_rope.transpose(1, 2)  # (B, H, S, qk_rope_head_dim)
        
        # 2. KV Path
        # 2.1 Compress KV
        c_kv = self.kv_down_proj(x)  # (B, S, kv_compress_dim)
        c_kv = self.kv_norm(c_kv)
        
        # 2.2 K rope from raw input (not compressed)
        k_rope = self.k_rope_proj(x)  # (B, S, qk_rope_head_dim)
        
        # 2.3 Handle KV Cache
        if past_key_value is not None:
            c_kv_cached, k_rope_cached = past_key_value
            
            # Validate cache shapes
            assert c_kv_cached.size(0) == batch_size, "Cache batch size mismatch"
            assert c_kv_cached.size(2) == self.kv_compress_dim, "Cache c_kv dim mismatch"
            
            # Concatenate with cached values
            c_kv = torch.cat([c_kv_cached, c_kv], dim=1)
            k_rope = torch.cat([k_rope_cached, k_rope], dim=1)
        
        kv_seq_len = c_kv.size(1)
        
        # 2.4 Prepare cache for return
        present_key_value = (c_kv, k_rope) if use_cache else None
        
        # 3. Apply RoPE
        # 3.1 Get cos/sin for full KV sequence length
        cos, sin = self.rope(kv_seq_len, device=x.device, dtype=x.dtype)
        
        # 3.2 Apply RoPE to k_rope: (B, kv_seq_len, qk_rope_head_dim)
        k_cos = cos.unsqueeze(0)  # (1, kv_seq_len, dim)
        k_sin = sin.unsqueeze(0)
        k_rope = (k_rope * k_cos) + (rotate_half(k_rope) * k_sin)
        
        # 3.3 Apply RoPE to q_rope: use last seq_len positions
        # Query positions are the most recent positions in the full sequence
        q_start = kv_seq_len - seq_len
        q_cos = cos[q_start:].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        q_sin = sin[q_start:].unsqueeze(0).unsqueeze(0)
        q_rope = (q_rope * q_cos) + (rotate_half(q_rope) * q_sin)
        
        # 4. Up-project K and V
        # 4.1 K content from compressed KV
        k_nope = self.k_nope_proj(c_kv)  # (B, kv_seq_len, num_heads * qk_nope_head_dim)
        k_nope = k_nope.view(batch_size, kv_seq_len, self.num_heads, self.qk_nope_head_dim)
        k_nope = k_nope.transpose(1, 2)  # (B, H, kv_seq_len, qk_nope_head_dim)
        
        # 4.2 V from compressed KV
        v = self.v_up_proj(c_kv)  # (B, kv_seq_len, num_heads * v_head_dim)
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.v_head_dim)
        v = v.transpose(1, 2)  # (B, H, kv_seq_len, v_head_dim)
        
        # 5. Assemble Full Q and K
        # 5.1 Broadcast k_rope to all heads: (B, kv_seq_len, dim) -> (B, H, kv_seq_len, dim)
        k_rope = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # 5.2 Concatenate nope and rope components
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, H, seq_len, qk_head_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, H, kv_seq_len, qk_head_dim)
        
        # 6. Attention
        # 6.1 Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 6.2 Apply mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # 6.3 Softmax and apply to values
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn_weights, v)  # (B, H, seq_len, v_head_dim)
        
        # 7. Output Projection
        # 7.1 Reshape: (B, H, S, v_head_dim) -> (B, S, H * v_head_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        
        # 7.2 Project to hidden_dim
        output = self.o_proj(output)
        
        return output, present_key_value
    
    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, "
            f"kv_compress_dim={self.kv_compress_dim}, q_compress_dim={self.q_compress_dim}"
        )


# ============== Test Code ==============
if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        hidden_dim: int = 768
        num_attention_heads: int = 12
        kv_compress_dim: int = 128
        q_compress_dim: int = 256
        qk_nope_head_dim: int = 32
        qk_rope_head_dim: int = 32
        v_head_dim: int = 64
        max_seq_len: int = 2048
        rope_base: float = 10000.0
    
    config = TestConfig()
    mla = MultiHeadLatentAttention(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.hidden_dim)
    
    # Without cache
    output, cache = mla(x, use_cache=True)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Cache c_kv shape:   {cache[0].shape}")
    print(f"Cache k_rope shape: {cache[1].shape}")
    
    # With cache (simulate generation)
    x_new = torch.randn(batch_size, 1, config.hidden_dim)
    output_new, cache_new = mla(x_new, past_key_value=cache, use_cache=True)
    print("\nGeneration step:")
    print(f"New input shape:  {x_new.shape}")
    print(f"New output shape: {output_new.shape}")
    print(f"Updated cache c_kv shape:   {cache_new[0].shape}")
    print(f"Updated cache k_rope shape: {cache_new[1].shape}")
    
    print("\nMLA test passed!")