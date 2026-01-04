"""
RMSNorm: Root Mean Square Layer Normalization
==============================================

DeepSeek-V3 uses RMSNorm instead of LayerNorm for efficiency.
RMSNorm skips the mean centering step, reducing computation by ~50%.

Formula: y = x / sqrt(mean(x^2) + eps) * weight
"""

import torch

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More efficient than LayerNorm - no mean subtraction, just scale by RMS.
    Used throughout DeepSeek-V3 for all normalization layers.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        assert dim > 0, f"Dimension must be positive, got {dim}"

        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.scale.size(0), \
            f"Input dim {x.size(-1)} != weight dim {self.scale.size(0)}"
        dtype = x.dtype
        x_float = x.float()
        rms_x = torch.mean(x_float**2, dim=-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(rms_x + self.eps)
        output = x_normed * self.scale
        return output.to(dtype=dtype)

    def extra_repr(self) -> str:
        return f"dim={self.scale.size(0)}, eps={self.eps}"

