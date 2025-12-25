import torch

class SwiGLUFFN(torch.nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
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

        self.w_gate = torch.nn.Linear(hidden_dim, intermediate_dim)
        self.w_up = torch.nn.Linear(hidden_dim, intermediate_dim)
        self.w_down = torch.nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement SwiGLU: (Swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down
        gate = self.w_gate(x)
        up = self.w_up(x)
        output = (torch.nn.functional.silu(gate) * up)
        output = self.w_down(output)
        return output