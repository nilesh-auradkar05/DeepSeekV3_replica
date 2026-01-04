"""
DeepSeek-V3 Model Configuration
===============================

Defines model architecture hyperparameters for different scales.

The architecture follows DeepSeek-V3's design:
    - Multi-Head Latent Attention (MLA) with compressed KV cache
    - Mixture of Experts (MoE) in later layers
    - Multi-Token Prediction (MTP) auxiliary heads
    - SwiGLU activation in all FFNs

Presets:
    - nano: ~150M params, for development and testing
    - small: ~500M params, for serious experimentation  
    - 1b: ~1B params, for demonstrating full capabilities
    - medium: ~2B params, requires 80GB GPU memory
"""

from dataclasses import dataclass


@dataclass
class DeepSeekV3Config:
    """
    Configuration for DeepSeek-V3 model architecture.
    
    All dimensions and counts validated in __post_init__.
    """
    
    # ================= Core Architecture =================
    vocab_size: int = 129280          # DeepSeek tokenizer vocabulary size
    hidden_dim: int = 768             # Model hidden dimension
    num_layers: int = 12              # Total transformer layers
    max_seq_len: int = 2048           # Maximum sequence length
    
    # ================= Attention (MLA) =================
    num_attention_heads: int = 12     # Number of attention heads
    
    # KV compression (the key MLA innovation)
    kv_compress_dim: int = 128        # Compressed KV dimension (cached)
    q_compress_dim: int = 256         # Query compression dimension
    
    # Head dimensions (per head)
    qk_nope_head_dim: int = 32        # Content (non-positional) Q/K dim
    qk_rope_head_dim: int = 32        # Positional (RoPE) Q/K dim  
    v_head_dim: int = 64              # Value dimension
    
    # RoPE parameters
    rope_base: float = 10000.0        # RoPE base frequency
    
    # ================= MoE Configuration =================
    num_dense_layers: int = 2         # Initial dense FFN layers
    
    # Expert configuration
    num_shared_experts: int = 1       # Always-active experts
    num_routed_experts: int = 16      # Conditionally-active experts
    num_experts_per_tok: int = 2      # Top-k experts per token
    
    # FFN dimensions
    intermediate_dim: int = 2048      # Dense FFN intermediate dim
    expert_intermediate_dim: int = 512  # MoE expert intermediate dim
    
    # Load balancing
    aux_loss_alpha: float = 0.001     # (Not used - auxiliary-loss-free)
    bias_update_speed: float = 0.001  # Load balancing bias adjustment rate
    
    # ================= Multi-Token Prediction =================
    mtp_depth: int = 1                # Number of MTP heads (0 to disable)
    mtp_lambda: float = 0.3           # MTP loss weight
    
    # ================= Training =================
    dropout: float = 0.0              # Dropout rate (0 for pretraining)
    attention_dropout: float = 0.0    # Attention dropout
    initializer_range: float = 0.006  # Weight initialization std
    
    # Memory optimization
    use_gradient_checkpointing: bool = True  # Trade compute for memory
    
    @property
    def num_moe_layers(self) -> int:
        """Number of layers using MoE FFN."""
        return self.num_layers - self.num_dense_layers
    
    @property
    def total_experts_per_layer(self) -> int:
        """Total experts in each MoE layer."""
        return self.num_shared_experts + self.num_routed_experts
    
    @property
    def qk_head_dim(self) -> int:
        """Total Q/K dimension per head (nope + rope)."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim
    
    @property  
    def head_dim(self) -> int:
        """Alias for qk_head_dim for compatibility."""
        return self.qk_head_dim
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Basic dimension checks
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_layers > 0, "num_layers must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        
        # Attention checks
        assert self.num_attention_heads > 0, "num_attention_heads must be positive"
        assert self.hidden_dim % self.num_attention_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        
        # MoE checks  
        assert self.num_dense_layers >= 0, "num_dense_layers must be non-negative"
        assert self.num_dense_layers <= self.num_layers, \
            f"num_dense_layers ({self.num_dense_layers}) > num_layers ({self.num_layers})"
        assert self.num_routed_experts >= 0, "num_routed_experts must be non-negative"
        assert self.num_experts_per_tok >= 0, "num_experts_per_tok must be non-negative"
        if self.num_routed_experts > 0:
            assert self.num_experts_per_tok <= self.num_routed_experts, \
                f"num_experts_per_tok ({self.num_experts_per_tok}) > num_routed_experts ({self.num_routed_experts})"
        
        # MTP checks
        assert self.mtp_depth >= 0, "mtp_depth must be non-negative"
        assert 0.0 <= self.mtp_lambda <= 1.0, f"mtp_lambda must be in [0, 1], got {self.mtp_lambda}"
        
        # RoPE dimension must be even
        assert self.qk_rope_head_dim % 2 == 0, \
            f"qk_rope_head_dim must be even for RoPE, got {self.qk_rope_head_dim}"


def get_nano_config() -> DeepSeekV3Config:
    """
    Nano configuration for development and testing.
    
    ~150M total params, ~40M activated per token.
    Suitable for single GPU experimentation.
    """
    return DeepSeekV3Config(
        # Core
        hidden_dim=768,
        num_layers=12,
        max_seq_len=2048,
        
        # Attention
        num_attention_heads=12,
        kv_compress_dim=128,
        q_compress_dim=256,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        
        # MoE
        num_dense_layers=2,
        num_shared_experts=1,
        num_routed_experts=16,
        num_experts_per_tok=2,
        intermediate_dim=2048,
        expert_intermediate_dim=512,
        
        # MTP
        mtp_depth=1,
        mtp_lambda=0.3,
    )


def get_small_config() -> DeepSeekV3Config:
    """
    Small configuration for serious experimentation.
    
    ~500M total params, ~150M activated per token.
    Good balance of capacity and training speed.
    """
    return DeepSeekV3Config(
        # Core
        hidden_dim=1024,
        num_layers=16,
        max_seq_len=2048,
        
        # Attention
        num_attention_heads=16,
        kv_compress_dim=256,
        q_compress_dim=512,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        
        # MoE
        num_dense_layers=2,
        num_shared_experts=1,
        num_routed_experts=32,
        num_experts_per_tok=4,
        intermediate_dim=4096,
        expert_intermediate_dim=1024,
        
        # MTP
        mtp_depth=1,
        mtp_lambda=0.3,
    )


def get_1b_config() -> DeepSeekV3Config:
    """
    1B parameter configuration for demonstration.
    
    ~1B total params, ~300M activated per token.
    Suitable for A100/H100 training.
    """
    return DeepSeekV3Config(
        # Core
        hidden_dim=1536,
        num_layers=24,
        max_seq_len=2048,
        
        # Attention
        num_attention_heads=24,
        kv_compress_dim=384,
        q_compress_dim=768,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        
        # MoE (more experts for 1B scale)
        num_dense_layers=3,
        num_shared_experts=2,
        num_routed_experts=32,
        num_experts_per_tok=4,
        intermediate_dim=6144,
        expert_intermediate_dim=1536,
        
        # MTP (2 depths for better representations)
        mtp_depth=2,
        mtp_lambda=0.3,
    )


def get_medium_config() -> DeepSeekV3Config:
    """
    Medium configuration for larger experiments.
    
    ~2B total params, ~500M activated per token.
    Requires 80GB GPU memory.
    """
    return DeepSeekV3Config(
        # Core
        hidden_dim=2048,
        num_layers=28,
        max_seq_len=4096,
        
        # Attention
        num_attention_heads=32,
        kv_compress_dim=512,
        q_compress_dim=1024,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        
        # MoE
        num_dense_layers=4,
        num_shared_experts=2,
        num_routed_experts=64,
        num_experts_per_tok=8,
        intermediate_dim=8192,
        expert_intermediate_dim=2048,
        
        # MTP
        mtp_depth=2,
        mtp_lambda=0.3,
    )


def estimate_params(config: DeepSeekV3Config) -> dict:
    """
    Estimate parameter counts for a configuration.
    
    Returns dictionary with total and activated parameter counts.
    """
    V = config.vocab_size
    D = config.hidden_dim
    L = config.num_layers
    Ld = config.num_dense_layers
    Lm = L - Ld
    
    # Embeddings (assuming not tied)
    P_embed = V * D * 2  # embed + lm_head
    
    # Attention per layer (approximate for MLA)
    # This is a rough estimate - MLA has different structure
    P_attn = (
        D * config.q_compress_dim +  # q_down
        config.q_compress_dim * (config.num_attention_heads * config.qk_nope_head_dim) +  # q_nope
        config.q_compress_dim * (config.num_attention_heads * config.qk_rope_head_dim) +  # q_rope
        D * config.kv_compress_dim +  # kv_down
        config.kv_compress_dim * (config.num_attention_heads * config.qk_nope_head_dim) +  # k_nope
        config.kv_compress_dim * (config.num_attention_heads * config.v_head_dim) +  # v
        D * config.qk_rope_head_dim +  # k_rope
        config.num_attention_heads * config.v_head_dim * D  # o_proj
    )
    
    # Dense FFN (SwiGLU: 3 * D * intermediate)
    P_dense_ffn = 3 * D * config.intermediate_dim
    
    # MoE FFN
    P_expert = 3 * D * config.expert_intermediate_dim
    E_total = config.num_shared_experts + config.num_routed_experts
    E_active = config.num_shared_experts + config.num_experts_per_tok
    P_router = D * config.num_routed_experts
    
    P_moe_total = E_total * P_expert + P_router
    P_moe_active = E_active * P_expert + P_router
    
    # Norms (small, but count them)
    P_norm = D * 2  # weight only, per norm
    P_norms_per_layer = P_norm * 2  # attn_norm + ffn_norm
    P_final_norm = P_norm
    
    # Total
    P_total = (
        P_embed +
        L * (P_attn + P_norms_per_layer) +
        Ld * P_dense_ffn +
        Lm * P_moe_total +
        P_final_norm
    )
    
    # Activated
    P_activated = (
        P_embed +
        L * (P_attn + P_norms_per_layer) +
        Ld * P_dense_ffn +
        Lm * P_moe_active +
        P_final_norm
    )
    
    return {
        "total_params": P_total,
        "activated_params": P_activated,
        "total_params_B": P_total / 1e9,
        "activated_params_B": P_activated / 1e9,
        "activation_ratio": P_activated / P_total,
    }


if __name__ == "__main__":
    # Quick test and parameter estimates
    configs = {
        "nano": get_nano_config(),
        "small": get_small_config(),
        "1b": get_1b_config(),
        "medium": get_medium_config(),
    }
    
    print("DeepSeek-V3 Configuration Presets")
    print("=" * 60)
    
    for name, config in configs.items():
        est = estimate_params(config)
        print(f"\n{name.upper()} Configuration:")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Layers: {config.num_layers} ({config.num_dense_layers} dense, {config.num_moe_layers} MoE)")
        print(f"  Attention heads: {config.num_attention_heads}")
        print(f"  Experts: {config.num_shared_experts} shared + {config.num_routed_experts} routed (top-{config.num_experts_per_tok})")
        print(f"  MTP depth: {config.mtp_depth}")
        print(f"  Estimated total params: {est['total_params_B']:.2f}B")
        print(f"  Estimated activated params: {est['activated_params_B']:.2f}B")
        print(f"  Activation ratio: {est['activation_ratio']:.1%}")