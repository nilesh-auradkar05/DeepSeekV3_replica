"""
Mixture of Experts (MoE) with Auxiliary-Loss-Free Load Balancing
================================================================

DeepSeek-V3's MoE uses several innovations:

1. Sigmoid Gating (not softmax)
   - Each expert score is independent: sigmoid(logit)
   - Normalized after top-k selection
   - Avoids softmax's "competition" between experts

2. Auxiliary-Loss-Free Load Balancing  
   - Traditional MoE adds auxiliary loss to encourage balanced routing
   - Problem: Auxiliary loss conflicts with main task loss
   - Solution: Dynamic bias adjustment on router outputs
   - If expert is overloaded -> decrease its bias -> fewer tokens routed to it
   - No extra loss term, just adaptive routing

3. Fine-grained Expert Segmentation
   - More smaller experts (64-256) vs fewer larger experts
   - Better specialization with similar parameter count

4. Shared Experts
   - Some experts process ALL tokens (not routed)
   - Captures common patterns, prevents capacity collapse
"""

import torch
import torch.nn as nn
from typing import Tuple

from utils.swiglu import SwiGLUFFN

class Expert(torch.nn.Module):
    """Single expert: a SwiGLU FFN."""

    def __init__(self, hidden_dim: int, expert_intermediate_dim: int):
        super().__init__()
        self.ffn = SwiGLUFFN(hidden_dim, expert_intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class DeepSeekMoE(torch.nn.Module):
    """
    Mixture of Experts layer with auxiliary-loss-free load balancing.
    
    Architecture:
        - num_shared_experts: Always-active experts (process all tokens)
        - num_routed_experts: Conditionally-active experts (top-k routing)
        - Router selects top-k routed experts per token
    
    Load Balancing:
        - Maintains per-expert bias added to routing scores
        - After each batch: experts with above-average load get bias decreased
        - No auxiliary loss needed - balancing happens through routing adjustment
    """

    def __init__(
        self,
        hidden_dim: int,
        num_shared_experts: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        expert_intermediate_dim: int
    ):
        super().__init__()

        # Validate inputs
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"
        assert num_shared_experts >= 0, "num_shared_experts must be non-negative"
        assert num_routed_experts >= 0, "num_routed_experts must be non-negative"
        assert num_experts_per_tok > 0, "num_experts_per_tok must be positive"
        assert num_experts_per_tok <= num_routed_experts or num_routed_experts == 0, \
            f"num_experts_per_tok ({num_experts_per_tok}) > num_routed_experts ({num_routed_experts})"
        
        self.hidden_dim = hidden_dim
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok    
        self.expert_intermediate_dim = expert_intermediate_dim

        # Shared experts (always active)
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                Expert(hidden_dim, expert_intermediate_dim)
                for _ in range(num_shared_experts)
            ])
        else:
            self.shared_experts = nn.ModuleList()
        
        # Routed experts (conditionally active)
        if num_routed_experts > 0:
            self.routed_experts = nn.ModuleList([
                Expert(hidden_dim, expert_intermediate_dim) 
                for _ in range(num_routed_experts)
            ])
            
            # Router gate (no bias - bias is separate for load balancing)
            self.gate = nn.Linear(hidden_dim, num_routed_experts, bias=False)
            
            # Load balancing bias (not a learnable parameter)
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_routed_experts),
                persistent=True,  # Save in state_dict for checkpoint resume
            )
        else:
            self.routed_experts = nn.ModuleList()
            self.gate = None
            self.expert_bias = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            output: (batch, seq_len, hidden_dim)
            top_k_indices: (batch, seq_len, num_experts_per_tok) for bias updates
        """
        B, S, H = x.shape
        assert H == self.hidden_dim, f"hidden_dim mismatch: got {H}, expected {self.hidden_dim}"

        # 1) Shared experts
        if self.num_shared_experts > 0:
            shared_outputs = sum((expert(x) for expert in self.shared_experts), torch.zeros_like(x))
        else:
            shared_outputs = torch.zeros_like(x)

        # 2) Early exit if no routed experts
        if self.num_routed_experts == 0 or self.num_experts_per_tok == 0:
            empty_indices = x.new_zeros((B, S, 0), dtype=torch.long)
            return shared_outputs, empty_indices

        # 3) ROUTING (compute in fp32 for stability + deterministic tie-break)
        # NOTE: Using .float() keeps gradients flowing back through the cast.
        assert self.gate is not None and self.expert_bias is not None
        x_fp32 = x.float()
        gate_w_fp32 = self.gate.weight.float()

        # Compute roting scores with sigmoid
        gate_logits = torch.nn.functional.linear(x_fp32, gate_w_fp32)        # (B,S,E) fp32
        scores = torch.sigmoid(gate_logits)                                   # (B,S,E) fp32

        # Add load balancing bias for selection
        bias = self.expert_bias.float()                               # (1,1,E) fp32
        scores_for_selection = scores + bias

        # deterministic tie-breaker for equal/near-equal scores
        tie = (torch.arange(self.num_routed_experts, device=x.device, dtype=torch.float32) * 1e-6)
        scores_for_routing = scores_for_selection + tie

        _, top_k_indices = torch.topk(scores_for_routing, k=self.num_experts_per_tok, dim=-1, largest=True, sorted=False) # (B, S, K)

        # Get original scores
        top_k_scores = scores.gather(dim=-1, index=top_k_indices)             # (B,S,K) fp32

        # Normalize weights

        top_k_weights = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        top_k_weights = top_k_weights.to(x.dtype)                                  # (B,S,K) fp32

        # 4) Dispatch + expert compute
        routed_out = self._dispatch_experts(x, top_k_indices, top_k_weights)
        
        # Combine shared and routed outputs
        output = shared_outputs + routed_out
        
        return output, top_k_indices

    def _dispatch_experts(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dispatch tokens to experts and combine outputs.
        
        Uses sorted dispatch for better GPU utilization:
        1. Sort all (token, expert) pairs by expert ID
        2. Process each expert's tokens as a contiguous batch
        3. Scatter results back to original positions
        """
        B, S, H = x.shape
        K = self.num_experts_per_tok
        T = B * S  # Total tokens
        
        # Flatten for dispatch
        x_flat = x.view(T, H)  # (T, H)
        idx_flat = indices.view(T, K)  # (T, K)
        w_flat = weights.view(T, K)  # (T, K)
        
        # Create (token_id, expert_id, weight) for all assignments
        # Each token has K expert assignments
        token_ids = torch.arange(T, device=x.device).unsqueeze(1).expand(T, K).reshape(-1)
        expert_ids = idx_flat.reshape(-1)
        assignment_weights = w_flat.reshape(-1)
        
        # Sort by expert ID for contiguous processing
        expert_order = torch.argsort(expert_ids, stable=True)
        token_ids_sorted = token_ids[expert_order]
        expert_ids_sorted = expert_ids[expert_order]
        weights_sorted = assignment_weights[expert_order]
        
        # Gather tokens in expert order
        x_sorted = x_flat.index_select(0, token_ids_sorted)  # (T*K, H)
        
        # Count tokens per expert
        expert_counts = torch.bincount(expert_ids_sorted, minlength=self.num_routed_experts)
        expert_offsets = torch.cumsum(expert_counts, dim=0)
        
        # Process each expert
        out_flat = x_flat.new_zeros(T, H)
        start = 0
        
        for e in range(self.num_routed_experts):
            end = expert_offsets[e].item()
            if end > start:
                # Get this expert's tokens
                x_e = x_sorted[start:end]  # (num_tokens, H)
                
                # Expert forward pass
                y_e = self.routed_experts[e](x_e)
                
                # Apply weights
                y_e = y_e * weights_sorted[start:end].unsqueeze(1)
                
                # Scatter back to original positions
                out_flat.index_add_(0, token_ids_sorted[start:end], y_e)
                
            start = end
        
        return out_flat.view(B, S, H)

    @torch.no_grad()
    def update_load_balancing(
        self,
        routing_indices: torch.Tensor,
        bias_update_speed: float = 0.001,
    ) -> None:
        """
        Update expert biases based on routing statistics.
        
        Called after each training step. Adjusts biases to encourage
        balanced expert utilization without auxiliary loss.
        
        Args:
            routing_indices: (batch, seq_len, num_experts_per_tok) from forward pass
            bias_update_speed: How fast to adjust biases (default 0.001)
        """
        if self.num_routed_experts == 0:
            return

        assert self.expert_bias is not None
        
        # Count how many times each expert was selected
        flat_indices = routing_indices.view(-1)
        expert_counts = torch.bincount(
            flat_indices, 
            minlength=self.num_routed_experts,
        ).float()
        
        # Expected count if perfectly balanced
        total_selections = routing_indices.numel()
        expected_count = total_selections / self.num_routed_experts
        
        # Adjust bias: decrease for overloaded, increase for underloaded
        # If count > expected -> bias -= speed (fewer tokens routed)
        # If count < expected -> bias += speed (more tokens routed)
        adjustment = torch.where(
            expert_counts > expected_count,
            torch.tensor(-bias_update_speed, device=self.expert_bias.device),
            torch.tensor(+bias_update_speed, device=self.expert_bias.device),
        )
        
        self.expert_bias.add_(adjustment.to(self.expert_bias.dtype))
    
    def get_load_balance_stats(self) -> dict:
        """Get current load balancing statistics for logging."""
        if self.expert_bias is None:
            return {}
        
        return {
            "expert_bias_mean": self.expert_bias.mean().item(),
            "expert_bias_std": self.expert_bias.std().item(),
            "expert_bias_min": self.expert_bias.min().item(),
            "expert_bias_max": self.expert_bias.max().item(),
        }
    
    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"shared_experts={self.num_shared_experts}, "
            f"routed_experts={self.num_routed_experts}, "
            f"experts_per_tok={self.num_experts_per_tok}"
        )

if __name__ == "__main__":
    # Test configuration
    batch_size = 2
    seq_len = 4
    hidden_size = 64
    num_shared_experts = 1
    num_routed_experts = 4
    num_experts_per_tok = 2
    expert_intermediate_size = 128
    
    # Create MoE layer
    moe = DeepSeekMoE(
        hidden_dim=hidden_size,
        num_shared_experts=num_shared_experts,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        expert_intermediate_dim=expert_intermediate_size
    )
    
    # Test input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    output, top_k_indices = moe(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Top-K indices shape: {top_k_indices.shape}")
    print("\nRouting decisions (which experts for each token):")
    print(f"  Batch 0, Token 0: Experts {top_k_indices[0, 0].tolist()}")
    print(f"  Batch 0, Token 1: Experts {top_k_indices[0, 1].tolist()}")
    print(f"  Batch 0, Token 2: Experts {top_k_indices[0, 2].tolist()}")
    print(f"  Batch 0, Token 3: Experts {top_k_indices[0, 3].tolist()}")
    
    print(f"\nExpert biases before update: {moe.expert_bias}")
    
    # Simulate training step
    moe.update_load_balancing(top_k_indices, bias_update_speed=0.01)
    
    print(f"Expert biases after update:  {moe.expert_bias}")
    
    print("\nMoE test passed!")