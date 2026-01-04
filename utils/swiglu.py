"""
SwiGLU Feed-Forward Network
===========================

DeepSeek-V3 uses SwiGLU activation (Swish-Gated Linear Unit) in FFN layers.
SwiGLU = Swish(xW) * (xV), where Swish(x) = x * sigmoid(x)

This combines the benefits of:
- Gated linear units (GLU) for expressiveness  
- Swish activation for smooth gradients
"""

import torch

class SwiGLUFFN(torch.nn.Module):
    """input x: (batch, seq_len, hidden_dim)

        SwiGLU:
        x ──┬──> [W_gate] ──> Swish ──┐
            │                         ⊙ ──> [W_down] ──> output
            └──> [W_up] ──────────────┘

        Gate path
        - gate = x @ W_gate            # output of intermediate_dim
        gate: (batch, seq_len, intermediate_dim)

        Up path
        - up = x @ W_up                  # output of intermediate_dim
        up: (batch, seq_len, intermediate_dim)

        Element-wise multiply (Shape should be same of both gate and up)
        - output = gate ⊙ up     # (batch, seq_len, intermediate_dim)

        Down projection back to hidden_dim
        - output = hidden_dim @ W_down
        output: (batch, seq_len, hidden_dim)
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        assert hidden_dim > 0, f"hidden_dim must be positive, got {hidden_dim}"
        assert intermediate_dim > 0, f"intermediate_dim must be positive, got {intermediate_dim}"

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        self.w_gate = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_up = torch.nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_down = torch.nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.hidden_dim, \
            f"Input dim {x.size(-1)} != hidden_dim {self.hidden_dim}"
        
        # Implement SwiGLU: (Swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down
        gate = self.w_gate(x)
        up = self.w_up(x)
        output = (torch.nn.functional.silu(gate) * up)
        output = self.w_down(output)
        return output

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, intermediate_dim={self.intermediate_dim}"