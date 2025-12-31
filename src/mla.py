import torch
from typing import Optional, Tuple
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepseek_rope import rotate_half, RoPE
from utils.rms_norm import RMSNorm

class MultiHeadLatentAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
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

        # Projection layers
        self.W_down_kv = torch.nn.Linear(
            self.hidden_dim,
            self.kv_compress_dim,
            bias=False,
        )

        self.kv_norm = RMSNorm(self.kv_compress_dim)

        self.k_up_proj = torch.nn.Linear(
            self.kv_compress_dim,
            self.num_heads * self.qk_nope_head_dim,
            bias=False,
        )

        self.v_up_proj = torch.nn.Linear(
            self.kv_compress_dim,
            self.num_heads * self.v_head_dim,
            bias=False,
        )

        self.k_rope_proj = torch.nn.Linear(
            self.hidden_dim,
            self.qk_rope_head_dim,
            bias=False,
        )

        self.q_down_proj = torch.nn.Linear(
            self.hidden_dim,
            self.q_compress_dim,
            bias=False,
        )

        self.q_norm = RMSNorm(self.q_compress_dim)

        self.q_up_proj = torch.nn.Linear(
            self.q_compress_dim,
            self.num_heads * self.qk_nope_head_dim,
            bias=False,
        )

        self.q_rope_proj = torch.nn.Linear(
            self.q_compress_dim,
            self.num_heads * self.qk_rope_head_dim,
            bias=False,
        )

        self.o_proj = torch.nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_dim,
            bias=False,
        )

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
        
        batch_size, seq_len, _ = x.shape

        # 1. Query computation
        c_q = self.q_down_proj(x)
        c_q = self.q_norm(c_q)

        # 2. Up-project query content
        q_nope = self.q_up_proj(c_q)        # (batch, seq_len, num_heads * qk_nope_head_dim)
        
        q_nope = q_nope.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim)
        # Transpose to (batch, num_heads, seq_len, qk_nope_head_dim)
        q_nope = q_nope.transpose(1,2)
        

        # 2. ======== KV Path ========
        # 2.1. Compress KV
        # x: (batch, seq_len, hidden_dim)
        # c_kv: (batch, seq_len, kv_compress_dim)
        c_kv = self.W_down_kv(x)
        c_kv = self.kv_norm(c_kv)

        # 2.2. Compute k_rope (Computed from x directly unlike q_rope which is computed from c_q)
        # k_rope: (batch, seq_len, qk_rope_head_dim)
        k_rope = self.k_rope_proj(x)

        # 2.3. ======== Handle Cache ========
        if past_key_value is not None:
            # Unpacking cached values
            c_kv_cached, k_rope_cached = past_key_value

            # Concatenate with new values along sequence dimension
            c_kv = torch.cat([c_kv_cached, c_kv], dim=1)
            k_rope = torch.cat([k_rope_cached, k_rope], dim=1)

        # Prepare cache for return (if use_cache is True)
        if use_cache:
            present_key_value = (c_kv, k_rope)

        else:
            present_key_value = None

        # 3. Apply RoPE to k_rope and up-project K
        # Get full sequence length (including cache)
        kv_seq_len = c_kv.shape[1]

        # Apply RoPE to k_rope
        cos, sin = self.rope(kv_seq_len, device=x.device, dtype=x.dtype)

        # Unsqueeze to add dim for batch and heads: (1, 1, seq_len, qk_rope_head_dim)
        k_cos = cos.unsqueeze(0)
        k_sin = sin.unsqueeze(0)

        # Apply rotation: rotate = x * cos + rotate_half(x) * sin
        k_rope = (k_rope * k_cos) + (rotate_half(k_rope) * k_sin)

        # 4. Compute Query RoPE
        q_rope = self.q_rope_proj(c_q)        # (batch, seq_len, num_heads * qk_rope_head_dim)
    
        q_rope = q_rope.view(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim)
        # Transpose to (batch, num_heads, seq_len, qk_rope_head_dim)
        q_rope = q_rope.transpose(1,2)

        # 4.1. Apply RoPE to q_rope
        # Unsqueeze to add dim for batch and heads: (1, 1, seq_len, qk_rope_head_dim)
        q_cos = cos[-seq_len:].unsqueeze(0).unsqueeze(0)
        q_sin = sin[-seq_len:].unsqueeze(0).unsqueeze(0)
        
        # Apply rotation: rotate = x * cos + rotate_half(x) * sin
        q_rope = (q_rope * q_cos) + (rotate_half(q_rope) * q_sin)

        # 5. Up project K and V
        k_nope = self.k_up_proj(c_kv)
        k_nope = k_nope.view(batch_size, kv_seq_len, self.num_heads, self.qk_nope_head_dim)
        k_nope = k_nope.transpose(1,2)

        # 6. Up project V
        v = self.v_up_proj(c_kv)
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.v_head_dim)
        v = v.transpose(1,2)

        # 7. Combine Content and RoPE
        # 7.1 Broadcast k_rope to all heads
        # (batch, seq_len, qk_rope_head_dim) -> (batch, num_heads, seq_len, qk_rope_head_dim)
        k_rope = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1) 

        # 7.2 Concatenate to form full K and Q
        # (batch, num_heads, seq_len, qk_nope_head_dim + qk_rope_head_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # 8. Compute Attention Scores
        # q: (batch, num_heads, seq_len, qk_head_dim)
        # k: (batch, num_heads, kv_seq_len, qk_head_dim)

        attn_scores = (torch.matmul(q, k.transpose(-2, -1))) * self.scale
        
        # Apply attention mask (for causal masking)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # 9. Softmax
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        # 10. Apply attention to values
        # v: (batch, num_heads, kv_seq_len, v_head_dim)
        output = torch.matmul(attn_weights, v)

        # 11. ====================== Output ==========================
        # output: (batch, num_heads, seq_len, v_head_dim)

        # Reshape: merge heads back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Project to hidden_dim
        output = self.o_proj(output)

        return output, present_key_value

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