from dataclasses import dataclass

@dataclass
class DeepSeekV3Config:
    """
    DeepSeek V3 Nano Model Architecture configuration.
    """

    # ================= General Model Configs ===========================
    vocab_size: int = 129_280   # original model has 129_280 tokens
    max_seq_len: int = 2048     # context length
    hidden_dim: int = 768       # original model has 7168
    num_layers: int = 12

    # ================= MLA configs =====================================
    num_attention_heads: int = 12
    head_dim: int = 64

    # kv compression config
    kv_compress_dim: int = 128

    # query compression config
    q_compress_dim: int = 256

    # RoPE config
    qk_rope_head_dim: int = 32
    qk_nope_head_dim: int = 32
    v_head_dim: int = 64

    rope_base: float = 10_000.0

    # ================= DeepSeekMoE configs =============================
    num_dense_layers: int = 2

    # Expert config
    num_shared_experts: int = 1
    num_routed_experts: int = 16
    num_experts_per_tok: int = 2

    # FFN dimension
    # For dense layers
    intermediate_dim: int = 2048
    # for MoE layers
    expert_intermediate_dim: int = 512

    # Auxiliary-loss free load balancing
    aux_loss_alpha: float = 0.001
    bias_update_speed: float = 0.001

    # Node-limited routing
    max_routed_nodes: int = 1

    # MTP
    mtp_depth: int = 1
    mtp_lambda: float = 0.3

    # ================= Training Settings ===========================
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # initialization
    initializer_range: float = 0.006

    # Gradient checkpointing for memory efficiency
    use_gradient_checkpointing: bool = True

    # Derived properties
    @property
    def num_moe_layers(self) -> int:
        """Number of layers that use MoE"""
        return self.num_layers - self.num_dense_layers

    @property
    def total_experts_per_moe_layer(self) -> int:
        """Total experts in each MoE layer (shared + routed)"""
        return self.num_shared_experts + self.num_routed_experts

    def __post_init__(self):
        """Validate config settings"""
        assert self.hidden_dim % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_dim}) must be divisible by num_attention_heads ({self.num_attention_heads})"

        assert self.num_dense_layers <= self.num_layers, \
            f"num_dense_layers ({self.num_dense_layers}) cannot exceed num_layers ({self.num_layers})"

        assert self.num_experts_per_tok <= self.num_routed_experts, \
            f"num_experts_per_tok ({self.num_experts_per_tok}) cannot exceed num_routed_experts ({self.num_routed_experts})"

def get_nano_config() -> DeepSeekV3Config:
    """Nano config for DeepSeek V3 nano model (~150M params, ~40M activated)"""
    return DeepSeekV3Config()


def get_small_config() -> DeepSeekV3Config:
    """Small config for more serious training (~500M params, ~150M activated)."""
    return DeepSeekV3Config(
        hidden_dim=1024,
        num_layers=16,
        num_attention_heads=16,
        kv_compress_dim=256,
        q_compress_dim=512,
        num_routed_experts=32,
        num_experts_per_tok=4,
        intermediate_dim=4096,
        expert_intermediate_dim=1024,
    )

# def estimate_params(cfg: DeepSeekV3Config, tied_embeddings: bool = True):
#     V = cfg.vocab_size
#     D = cfg.hidden_dim
#     L = cfg.num_layers
#     Ld = cfg.num_dense_layers
#     Lm = L - Ld

#     Dff = cfg.intermediate_dim
#     Dexp = cfg.expert_intermediate_dim

#     E = cfg.num_routed_experts
#     Es = cfg.num_shared_experts
#     k = cfg.num_experts_per_tok

#     P_emb = V * D if tied_embeddings else 2 * V * D
#     P_attn = 4 * D * D  # baseline MHA count

#     P_dense_ffn = 3 * D * Dff  # SwiGLU
#     P_expert = 3 * D * Dexp    # SwiGLU per expert

#     P_moe_ffn_total = (E + Es) * P_expert
#     P_moe_ffn_act = (k + Es) * P_expert

#     P_total = (
#         P_emb
#         + Ld * (P_attn + P_dense_ffn)
#         + Lm * (P_attn + P_moe_ffn_total)
#     )

#     P_activated = (
#         P_emb
#         + Ld * (P_attn + P_dense_ffn)
#         + Lm * (P_attn + P_moe_ffn_act)
#     )

#     return {
#         "total_params": P_total,
#         "activated_params": P_activated,
#         "total_params_B": P_total / 1e9,
#         "activated_params_B": P_activated / 1e9,
#     }
 

def get_1b_config() -> DeepSeekV3Config:
    """
        ~1B-ish total parameters (MoE makes exact counting depend on implementation).
    """
    return DeepSeekV3Config(
        hidden_dim=1536,
        num_layers=20,
        num_attention_heads=24,
        max_seq_len=2048,
        kv_compress_dim=384,
        q_compress_dim=768,
        num_dense_layers=2,
        num_shared_experts=1,
        num_routed_experts=16,
        num_experts_per_tok=2,
        intermediate_dim=6144,
        expert_intermediate_dim=1024
    )

def get_medium_config() -> DeepSeekV3Config:
    """Medium config (~2B params, ~500M activated) - needs full 80GB GPU memory."""
    return DeepSeekV3Config(
        hidden_dim=2048,
        num_layers=24,
        num_attention_heads=32,
        kv_compress_dim=512,
        q_compress_dim=1024,
        num_routed_experts=64,
        num_experts_per_tok=8,
        intermediate_dim=8192,
        expert_intermediate_dim=2048,
        max_seq_len=4096,
    )

if __name__ == "__main__":
    # Quick test
    config = get_1b_config()
    print("Nano config created successfully!")
    print(f"  Hidden size: {config.hidden_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Dense layers: {config.num_dense_layers}")
    print(f"  MoE layers: {config.num_moe_layers}")
    print(f"  Experts per MoE layer: {config.total_experts_per_moe_layer}")
    print(f"  Activated experts per token: {config.num_experts_per_tok + config.num_shared_experts}")
    # param_cnt = estimate_params(config)
    # print(f"    Total params: {param_cnt["total_params"]:,}")
    # print(f"    Activated params: {param_cnt["activated_params"]:,}")
    # print(f"    Total params (B): {param_cnt["total_params_B"]:.2f}")
    # print(f"    Activated params (B): {param_cnt["activated_params_B"]:.2f}")