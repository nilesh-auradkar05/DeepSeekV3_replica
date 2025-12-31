import torch
from typing import Tuple, Optional

from utils.swiglu import SwiGLUFFN

class Expert(torch.nn.Module):
    def __init__(self, hidden_dim: int, expert_intermediate_dim: int):
        super().__init__()
        self.ffn = SwiGLUFFN(hidden_dim, expert_intermediate_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class Router(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_routed_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.num_routed_experts = num_routed_experts

        self.gate = torch.nn.Linear(hidden_dim, num_routed_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(num_routed_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            top_k_indices: (batch, seq_len, num_expertes_per_tok) - which experts to route to
            top_k_weights: (batch, seq_len, num_expertes_per_tok) - gating weights (sum to 1)
        """
        # 1. Compute affinity scores
        scores = torch.nn.functional.sigmoid(self.gate(x))

        # 2. Add expert bias
        scores_for_selection = scores + self.expert_bias

        # 3. Get top k scores and indices
        _, top_k_indices = torch.topk(scores_for_selection, k=self.num_experts_per_tok, dim=-1)

        # 4. Get original scores for the selected experts
        top_k_scores = torch.gather(scores, dim=-1, index=top_k_indices)

        # 4. Normalize weights to sum to 1
        top_k_weights = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        return top_k_indices, top_k_weights

    def update_expert_bias(self, top_k_indices: torch.Tensor, bias_update_speed: float):
        """
        Update expert bias based on load.

        Args:
            top_k_indices: (batch, seq_len, num_experts_per_tok) - indices of the top k experts to route per token
            bias_update_speed: gamma parameter to control how fast to adjust the bias
        """
        # 1. Count how many tokens each expert received
        flat_indices = top_k_indices.flatten()
        expert_counts = torch.bincount(flat_indices, minlength=self.num_routed_experts).float()

        # 2. Calculate expected count (If routing is perfectly balanced)
        total_tokens = top_k_indices.shape[0] * top_k_indices.shape[1]
        total_selections = total_tokens * self.num_experts_per_tok
        expected_count = total_selections / self.num_routed_experts

        # 3. Update biases
        # if expert received more tokens than expected -> decrease bias
        # if expert received less tokens than expected -> increase bias
        adjustment = torch.where(
            expert_counts > expected_count,
            -bias_update_speed,
            +bias_update_speed
        )
        self.expert_bias += adjustment

class DeepSeekMoE(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_shared_experts: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        expert_intermediate_dim: int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_tok = num_experts_per_tok    

        self.shared_experts = torch.nn.ModuleList([
            Expert(self.hidden_dim, expert_intermediate_dim) for _ in range(num_shared_experts)
        ])
        self.routed_experts = torch.nn.ModuleList([
            Expert(self.hidden_dim, expert_intermediate_dim) for _ in range(num_routed_experts)
        ])
        self.router = Router(hidden_dim, num_routed_experts, num_experts_per_tok)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)

        Returns:
            output: (batch, seq_len, hidden_dim)
            top_k_indices: (batch, seq_len, num_experts_per_tok) for bias updates
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 1. Apply shared experts
        shared_outputs = sum(expert(x) for expert in self.shared_experts)

        # 2. Apply Routed experts
        # 2.1 Get top k indices and weights
        top_k_indices, top_k_weights = self.router(x)

        # 2.2 Apply routed experts
        routed_outputs = torch.zeros_like(x)

        # Slower implementation of the routed experts
        # in production it is to be done by batched operations and padding, masking to process all experts in parallel
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(self.num_experts_per_tok):
                    expert_idx = top_k_indices[b, s, k].item()
                    weight = top_k_weights[b, s, k]
                    token = x[b, s].unsqueeze(0)                        # (1, hidden_dim)

                    expert_output = self.routed_experts[expert_idx](token)
                    routed_outputs[b, s] += weight * expert_output.squeeze(0)

        # 4. Combine shared and routed outputs
        output = shared_outputs + routed_outputs

        return output, top_k_indices

    def update_load_balancing(self, top_k_indices: torch.Tensor, bias_update_speed: float):
        self.router.update_expert_bias(top_k_indices, bias_update_speed)

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
    
    print(f"\nExpert biases before update: {moe.router.expert_bias}")
    
    # Simulate training step
    moe.update_load_balancing(top_k_indices, bias_update_speed=0.01)
    
    print(f"Expert biases after update:  {moe.router.expert_bias}")
    
    print("\nMoE test passed!")