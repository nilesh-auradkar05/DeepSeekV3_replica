
import torch
from typing import Optional, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLUFFN
from mla import MultiHeadLatentAttention
from MoE import DeepSeekMoE

class DeepSeekV3TransformerBlock(torch.nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim
        self.is_moe_layer = layer_idx >= config.num_dense_layers

        # Pre-attention norm
        self.attn_norm = RMSNorm(config.hidden_dim)

        # Attention
        self.attn = MultiHeadLatentAttention(config)

        # Pre-ffn norm
        self.ffn_norm = RMSNorm(self.hidden_dim)

        # FFN: Dense for first few layers, MoE for rest of the layers
        if layer_idx < config.num_dense_layers:
            self.ffn = SwiGLUFFN(self.hidden_dim, config.intermediate_dim)
        else:
            self.ffn = DeepSeekMoE(
                hidden_dim=self.hidden_dim,
                num_shared_experts=config.num_shared_experts,
                num_routed_experts=config.num_routed_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                expert_intermediate_dim=config.expert_intermediate_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[torch.Tensor]]:

        """
        Returns:
            output: (batch, seq_len, hidden_dim)
            present_key_value: Updated kv cache (if use_cache is True)
            top_k_indices: routing indices for MoE else None
        """
        # 1. Attention with residual connection
        residual = x
        x = self.attn_norm(x)
        x, present_key_value = self.attn(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache)
        x = x + residual

        # 2. FFN with residual
        residual = x
        x = self.ffn_norm(x)
        top_k_indices: Optional[torch.Tensor] = None

        if self.is_moe_layer:
            x, top_k_indices = self.ffn(x)
        else:
            x = self.ffn(x)

        x = x + residual

        return x, present_key_value, top_k_indices

class TransformerBlockStack(torch.nn.Module):
    """
    Stack of Transformer Blocks.

    Handles:
        - Creating blocks with appropriate layer indices
        - Collecting kv caches from all layers
        - Collecting routing indices from MoE layers
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers

        # Create all Transformer Blocks
        self.layers = torch.nn.ModuleList([
            DeepSeekV3TransformerBlock(config, layer_idx)
            for layer_idx in range(self.num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple], list]:
        """
        Forward pass through all transformer blocks.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_values: Tuple of (c_kv, k_rope) for each layer
            use_cache: Whether to return caches
        
        Returns:
            output: (batch, seq_len, hidden_size)
            present_key_values: Tuple of caches for each layer (if use_cache)
            all_top_k_indices: List of routing indices from MoE layers 
        """
        present_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...] = () if use_cache else ()
        all_top_k_indices = []

        for i, layer in enumerate(self.layers):
            # Get pas KV for this layer if available
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # Forward pass through layer
            x, present_key_value, top_k_indices = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                use_cache=use_cache
            )

            if use_cache:
                present_key_values = present_key_values + (present_key_value,)

            # Collect routing indices (for load balancing updates)
            if top_k_indices is not None:
                all_top_k_indices.append(top_k_indices)

        return x, present_key_values, all_top_k_indices

# ============== Test Code ==============
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from configs.model_config import get_1b_config
    
    # Get config
    config = get_1b_config()
    
    # Test single block
    print("=" * 50)
    print("Testing TransformerBlock")
    print("=" * 50)
    
    # Test dense layer (layer 0)
    block_dense = DeepSeekV3TransformerBlock(config, layer_idx=0)
    print("\nDense block (layer 0):")
    print(f"  is_moe_layer: {block_dense.is_moe_layer}")
    print(f"  FFN type: {type(block_dense.ffn).__name__}")
    
    # Test MoE layer (layer 2, after num_dense_layers=2)
    block_moe = DeepSeekV3TransformerBlock(config, layer_idx=2)
    print("\nMoE block (layer 2):")
    print(f"  is_moe_layer: {block_moe.is_moe_layer}")
    print(f"  FFN type: {type(block_moe.ffn).__name__}")
    
    # Forward pass test
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, config.hidden_dim)
    
    print(f"\nInput shape: {x.shape}")
    
    # Test dense block
    out_dense, cache_dense, indices_dense = block_dense(x, use_cache=True)
    print("\nDense block output:")
    print(f"  Output shape: {out_dense.shape}")
    print(f"  Cache: c_kv={cache_dense[0].shape}, k_rope={cache_dense[1].shape}")
    print(f"  Routing indices: {indices_dense}")  # Should be None
    
    # Test MoE block
    out_moe, cache_moe, indices_moe = block_moe(x, use_cache=True)
    print("\nMoE block output:")
    print(f"  Output shape: {out_moe.shape}")
    print(f"  Cache: c_kv={cache_moe[0].shape}, k_rope={cache_moe[1].shape}")
    print(f"  Routing indices shape: {indices_moe.shape}")
    
    # Test full stack
    print("\n" + "=" * 50)
    print("Testing TransformerBlockStack")
    print("=" * 50)
    
    stack = TransformerBlockStack(config)
    print(f"\nStack has {len(stack.layers)} layers")
    print(f"  Dense layers: {config.num_dense_layers}")
    print(f"  MoE layers: {config.num_moe_layers}")
    
    out_stack, caches, all_indices = stack(x, use_cache=True)
    print("\nStack output:")
    print(f"  Output shape: {out_stack.shape}")
    print(f"  Number of caches: {len(caches)}")
    print(f"  Number of routing index tensors: {len(all_indices)}")
    
    print("\nTransformer block tests passed!")
