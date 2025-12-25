import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms_x = torch.mean(x_float**2, dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(rms_x + self.eps)
        output = x_normed.type_as(x) * self.scale
        return output

