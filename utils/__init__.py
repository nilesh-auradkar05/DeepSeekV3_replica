"""
DeepSeek-V3 Nano: Model Components
==================================

This package contains all model components for the DeepSeek-V3 Nano implementation.

Main classes:
- DeepSeekV3ForCausalLM: The complete model for causal language modeling
- DeepSeekV3Config: Configuration dataclass

Components:
- RMSNorm: Root Mean Square Normalization
- RotaryEmbedding: Rotary Position Embeddings (RoPE)
- MultiHeadLatentAttention: MLA with KV compression
- SwiGLUFFN: Gated FFN with SiLU activation
- DeepSeekMoE: Mixture of Experts layer
- TransformerBlock: Single transformer layer
- MultiTokenPrediction: MTP module for enhanced training
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLUFFN
from src.mtp import MTPBlock, MultiTokenPrediction, compute_mtp_loss
from src.MoE import Router, Expert, DeepSeekMoE
from src.deepseekv3 import DeepSeekV3ForCausalLM
from src.deepseek_transformer import DeepSeekV3TransformerBlock, TransformerBlockStack
from src.deepseek_rope import rotate_half, apply_rotary_pos_emb, RoPE
from src.mla import MultiHeadLatentAttention

__all__ = [
    # Full Model
    "DeepSeekV3ForCausalLM",

    # Foundations
    "RMSNorm",
    "RoPE",
    "rotate_half",
    "apply_rotary_pos_emb",

    # Attention
    "MultiHeadLatentAttention",

    # MoE
    "SwiGLUFFN",
    "Router",
    "Expert",
    "DeepSeekMoE",

    # Transformer
    "DeepSeekV3TransformerBlock",
    "TransformerBlockStack",

    # MTP
    "MTPBlock",
    "MultiTokenPrediction",
    "compute_mtp_loss",
]