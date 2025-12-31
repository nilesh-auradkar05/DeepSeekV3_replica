import torch
from typing import Optional, Tuple, Dict, Any
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MoE import DeepSeekMoE
from deepseek_transformer import DeepSeekV3TransformerBlock, TransformerBlockStack
from utils.rms_norm import RMSNorm
from mtp import MultiTokenPrediction, compute_mtp_loss

class DeepSeekV3ForCausalLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Token Embedding (input)
        self.embed_tokens = torch.nn.Embedding(self.config.vocab_size, self.config.hidden_dim)

        # 2. Transformer layers
        self.layers = TransformerBlockStack(config)

        # 3. Normalize
        self.norm = RMSNorm(self.config.hidden_dim)

        # 4. LM head (output) - seperate from embedding
        self.lm_head = torch.nn.Linear(self.config.hidden_dim, self.config.vocab_size, bias=False)

        # 5. Multi Token Prediction
        if self.config.mtp_depth > 0:
            self.mtp = MultiTokenPrediction(config, self.embed_tokens)
        else:
            self.mtp = None

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module):
        std = self.config.initializer_range
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def _mask_causal_mask(
        self,
        seq_len: int,
        kv_seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:

        """
        Create causal attention mask.

        During training: seq_len == kv_seq_len (full sequence)
        During generation: seq_len += 1, kv_seq_len = total length with cache
        """
        query_start = kv_seq_len - seq_len

        # Q @ K.T
        query_positions = torch.arange(query_start, kv_seq_len, device=device)
        key_positions = torch.arange(kv_seq_len, device=device)

        # Boolean mask: True where query can't attent (query_pos < key_pos)
        bool_mask = query_positions.unsqueeze(1) < key_positions.unsqueeze(0)

        # Convert to attention mask format
        causal_mask = torch.zeros((seq_len, kv_seq_len), device=device, dtype=dtype)
        causal_mask.masked_fill_(bool_mask, float("-inf"))

        return causal_mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor|Any]:
        batch_size, seq_len = input_ids.shape

        # 1. Embed token
        hidden_states = self.embed_tokens(input_ids)

        # 2. Create causal mask if not provided
        if attention_mask is None:
            if past_key_values is not None:
                kv_seq_len = past_key_values[0][0].shape[1] + seq_len
            else:
                kv_seq_len = seq_len
            
            attention_mask = self._mask_causal_mask(
                seq_len=seq_len,
                kv_seq_len=kv_seq_len,
                device=input_ids.device,
                dtype=hidden_states.dtype
            )

        # 3. Pass through transformer layers
        hidden_states, present_key_values, routing_indices = self.layers(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # 4. Normalize
        hidden_states = self.norm(hidden_states)

        # 5. Apply LM head (main next-token prediction)
        logits = self.lm_head(hidden_states)

        # 6. Apply Multi Token Prediction
        if self.mtp is not None and self.training:
            mtp_logits = self.mtp(hidden_states, input_ids)
        else:
            mtp_logits = []

        return {
            "logits": logits,
            "mtp_logits": mtp_logits,
            "past_key_values": present_key_values if use_cache else None,
            "routing_indices": routing_indices
        }

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training.

        Args:
            input_ids: (batch_size, seq_len) input tokens
            labels: (batch_size, seq_len) target tokens (same as input_ids)
            attention_mask: Optional attention mask.

        Returns:
            Dictionary with total_loss and individual loss components
        """
        # Forward pass
        outputs = self.forward(input_ids, attention_mask=attention_mask, **kwargs)

        logits = outputs["logits"]
        mtp_logits = outputs["mtp_logits"]

        # Use input_ids as labels for next-token-prediction
        if labels is None:
            labels = input_ids

        # compute combined loss
        total_loss, loss_dict = compute_mtp_loss(
            main_logits=logits,
            mtp_logits=mtp_logits,
            input_ids=input_ids,
            mtp_lambda=self.config.mtp_lambda,
        )

        # Add routing indices for load balancing updates
        loss_dict["routing_indices"] = outputs["routing_indices"]

        return loss_dict

    def update_moe_load_balancing(
        self,
        routing_indices: torch.Tensor,
        bias_update_speed: float = 0.0,
    ):
        """
        Update MoE router biases based on routing indices.

        Args:
            routing_indices: List of routing tensors from forward pass
            bias_update_speed: Override config's bias_update_speed if provided
        """

        if bias_update_speed == 0.0:
            bias_update_speed = self.config.bias_update_speed

        # Each MoE layer has its routing indices
        moe_layer_idx = 0
        for layer in self.layers.layers:
            if getattr(layer, "is_moe_layer", False) and isinstance(layer.ffn, DeepSeekMoE):
                if moe_layer_idx < len(routing_indices):
                    # routing_indices[moe_layer_idx]: (batch, seq_len, num_experts_per_tok)
                    layer.ffn.update_load_balancing(
                        routing_indices[moe_layer_idx],
                        bias_update_speed,
                    )
                    moe_layer_idx += 1

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:

        """
        Generate text autoregressively.

        Args:
            input_ids: (batch_size, seq_len) prompt tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature(1.0 = neutral)
            top_k: keep only top_k tokens for sampling
            top_p: keep tokens with cumulative probability <= top_p
            do_sample: If false, use greedy decoding

        Returns:
            (batch_size, seq_len + max_new_tokens) generated sequence
        """
        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None
        current_input: Optional[torch.Tensor] = None

        for _ in range(max_new_tokens):
            # Only process the last token if we have cache
            if past_key_values is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated

            # Forward pass
            outputs = self.forward(
                current_input,
                past_key_values=past_key_values if past_key_values is not None else None,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :] # (batch_size, vocab_size)
            past_key_values = outputs["past_key_values"]

            # Apply temperature scaling
            if temperature != 1.0:
                probs = logits / temperature

            # Apply top_k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k).values[:, -1:]
                logits[indices_to_remove] = float("-inf")

            # Apply top_p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if do_sample:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

        return generated

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Count model parameters.

        Args:
            non_embedding: Exclude embedding parameters

        Returns:
            Number of parameters in the model
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params

    def get_num_activated_params(self) -> int:
        """
        Count activated parameters per forward pass.

        For MoE layers, only count the experts that are activated
        """
        # Embedding + LM head
        n_params = self.embed_tokens.weight.numel()
        n_params += self.lm_head.weight.numel()

        # Final norm
        n_params += sum(p.numel() for p in self.norm.parameters())

        # Transformer layers
        for layer in self.layers.layers:
            if not isinstance(layer, DeepSeekV3TransformerBlock):
                continue
            # Attention (always activated)
            n_params += sum(p.numel() for p in layer.attn.parameters())
            n_params += sum(p.numel() for p in layer.attn_norm.parameters())
            n_params += sum(p.numel() for p in layer.ffn_norm.parameters())

            if layer.is_moe_layer:
                if not isinstance(layer.ffn, DeepSeekMoE):
                    continue
                # MoE: only count shared experts + top-k routed experts
                moe = layer.ffn
                # Shared experts
                for expert in moe.shared_experts:
                    n_params += sum(p.numel() for p in expert.parameters())
                # Router
                n_params += sum(p.numel() for p in moe.router.parameters())
                # Only top-k routed experts activated per token
                if len(moe.routed_experts) > 0:
                    params_per_expert = sum(
                        p.numel() for p in moe.routed_experts[0].parameters()
                    )
                    n_params += params_per_expert * moe.num_experts_per_tok
            else:
                # Dense FFN (always activated)
                n_params += sum(p.numel() for p in layer.ffn.parameters())

        # MTP (if enabled)
        if self.mtp is not None:
            n_params += sum(p.numel() for p in self.mtp.parameters())

        return n_params

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from configs.model_config import get_nano_config
    
    print("=" * 60)
    print("Testing DeepSeekV3ForCausalLM")
    print("=" * 60)
    
    # Get config
    config = get_nano_config()
    print("\nConfig:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_dim}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_dense_layers: {config.num_dense_layers}")
    print(f"  num_routed_experts: {config.num_routed_experts}")
    print(f"  mtp_depth: {config.mtp_depth}")
    
    # Create model
    model = DeepSeekV3ForCausalLM(config)
    
    # Count parameters
    total_params = model.get_num_params()
    activated_params = model.get_num_activated_params()
    print("\nParameter count:")
    print(f"  Total params: {total_params:,}")
    print(f"  Activated params: {activated_params:,}")
    print(f"  Ratio: {activated_params / total_params:.2%}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print("\nForward pass test:")
    print(f"  Input shape: {input_ids.shape}")
    
    model.train()
    outputs = model(input_ids)
    
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  MTP logits: {len(outputs['mtp_logits'])} tensors")
    if outputs['mtp_logits']:
        for i, mtp_logit in enumerate(outputs['mtp_logits']):
            print(f"    MTP depth {i+1}: {mtp_logit.shape}")
    print(f"  Routing indices: {len(outputs['routing_indices'])} MoE layers")
    
    # Test loss computation
    print("\nLoss computation test:")
    loss_dict = model.compute_loss(input_ids)
    for name, value in loss_dict.items():
        if isinstance(value, torch.Tensor) and value.dim() == 0:
            print(f"  {name}: {value.item():.4f}")
    
    # Test generation
    print("\nGeneration test:")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 5))  # Short prompt
    generated = model.generate(prompt, max_new_tokens=10, do_sample=False)
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    
    # Test with cache (generation mode)
    print("\nCache test (simulating generation):")
    model.eval()
    
    # First pass: full sequence
    outputs1 = model(input_ids[:, :8], use_cache=True)
    cache = outputs1["past_key_values"]
    print("  Initial seq_len: 8")
    print(f"  Cache layers: {len(cache)}")
    print(f"  Cache c_kv shape: {cache[0][0].shape}")
    
    # Second pass: one new token with cache
    outputs2 = model(input_ids[:, 8:9], past_key_values=cache, use_cache=True)
    new_cache = outputs2["past_key_values"]
    print("  New token seq_len: 1")
    print(f"  Updated cache c_kv shape: {new_cache[0][0].shape}")
    
    print("All tests passed!")
