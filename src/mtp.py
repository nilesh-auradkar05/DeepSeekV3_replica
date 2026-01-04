"""
Multi-Token Prediction (MTP)
============================

DeepSeek-V3's approach to improving sample efficiency and representation quality.

Key insight: Instead of predicting just the next token, predict multiple future
tokens simultaneously. This provides:
- Richer training signal (more supervision per forward pass)
- Better long-range planning in representations
- Improved sample efficiency

Architecture per MTP depth:
    hidden_states (from backbone) 
           ↓
    [norm → concat with shifted embeddings → linear → transformer layer → lm_head]
           ↓
    logits for position +d ahead

The transformer layer at each depth is crucial.
It allows each depth to build complex predictions from the backbone representations.

Loss weighting:
    total_loss = main_loss + lambda * sum(mtp_losses)
    Default lambda = 0.3 (from paper)
"""

import torch
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.rms_norm import RMSNorm
from utils.swiglu import SwiGLUFFN

class MTPTransformerLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Self-attention
        self.attn_norm = RMSNorm(hidden_dim)
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        # FFN
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, hidden_dim * 4)

        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_dim)
        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        B, S, H = x.shape

        # self-attention with residual connection
        residual = x
        x = self.attn_norm(x)

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Causal attention (MTP should only attend to previous tokens)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, H)
        attn_out = self.out_proj(attn_out)

        x = residual + self.dropout(attn_out)

        # FFN with residual
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.dropout(self.ffn(x))

        return x

class MTPBlock(torch.nn.Module):
    """
    Single MTP prediction block for one depth level.
    
    Combines backbone hidden states with shifted token embeddings,
    processes through a transformer layer, and predicts logits.
    """
    def __init__(self, config, shared_lm_head: Optional[torch.nn.Linear] = None):
        super().__init__()

        self.dropout_val = config.dropout
        
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size

        # Normalize incoming output from transformer block
        self.input_norm = RMSNorm(self.hidden_dim)

        # Project input to hidden dimension
        self.input_proj = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)

        self.dropout = torch.nn.Dropout(config.dropout)

        # Transformer layer
        num_mtp_heads = max(4, config.num_attention_heads // 2)
        self.transformer = MTPTransformerLayer(hidden_dim=self.hidden_dim, num_heads=num_mtp_heads, dropout=config.dropout)

        # output projection
        if shared_lm_head is not None:
            self.lm_head = shared_lm_head
        else:
            self.lm_head = torch.nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Args:
            hidden_states: The hidden states from the transformer block (batch_size, seq_len, hidden_dim)
            token_embeddings: The token embeddings from the tokenizer (batch_size, seq_len, hidden_dim)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            hidden_states: (batch_size, seq_len, hidden_dim) for next depth of MTP
        """
        assert hidden_states.size(-1) == self.hidden_dim
        assert token_embeddings.size(-1) == self.hidden_dim
        assert hidden_states.size(1) == token_embeddings.size(1), \
            f"Sequence length mismatch: {hidden_states.size(1)} vs {token_embeddings.size(1)}"

        hidden_states = self.input_norm(hidden_states)
        combined = torch.cat([hidden_states, token_embeddings], dim=-1)
        combined = self.input_proj(combined)
        if self.dropout_val > 0.0:
            combined = self.dropout(combined)

        # Process through transformer layer
        hidden_out = self.transformer(combined)

        # Predict logits
        logits = self.lm_head(hidden_out)

        return logits, hidden_out

class MultiTokenPrediction(torch.nn.Module):
    """
    Multi-Token Prediction module.
    
    Creates multiple MTP blocks, each predicting tokens at increasing offsets.
    - Depth 0: Main model predicts position +1 (standard LM)
    - Depth 1: MTP block predicts position +2
    - Depth d: MTP block predicts position +d+1
    """
    def __init__(self, config, shared_embeddings: torch.nn.Embedding, shared_lm_head: Optional[torch.nn.Linear] = None):
        super().__init__()
        self.mtp_depth = config.mtp_depth
        self.hidden_dim = config.hidden_dim

        assert self.mtp_depth > 0, f"mtp_depth must be > 0, got {self.mtp_depth}"

        self.embeddings = shared_embeddings

        # One block per MTP depth
        self.mtp_blocks = torch.nn.ModuleList([
            MTPBlock(config, shared_lm_head=shared_lm_head)
            for _ in range(self.mtp_depth)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Args:
            hidden_states: The hidden states from the transformer block (batch_size, seq_len, hidden_dim)
            input_ids: The input token ids (batch_size, seq_len)

        Returns:
            all_logits: List of logits tensors, one per MTP depth
        """
        B, S, H = hidden_states.shape

        assert hidden_states.size(0) == input_ids.size(0), \
            f"Batch size mismatch: hidden_states {hidden_states.size(0)} vs input_ids {input_ids.size(0)}"

        assert input_ids.size(0) == B
        assert input_ids.size(1) == S

        all_logits = []
        current_hidden = hidden_states

        for depth, mtp_block in enumerate(self.mtp_blocks):
            offset = depth + 1
            
            # Need at least offset+1 positions to make a prediction
            if S <= offset:
                break
            
            # Get shifted token embeddings (tokens at positions offset onwards)
            shifted_ids = input_ids[:, offset:]  # (B, seq_len - offset)
            token_emb = self.embeddings(shifted_ids)
            
            # Truncate hidden states to match
            valid_len = token_emb.size(1)
            hidden_trunc = current_hidden[:, :valid_len]
            
            # Forward through MTP block
            logits, current_hidden = mtp_block(hidden_trunc, token_emb)
            
            all_logits.append(logits)
        
        return all_logits

def compute_mtp_loss(
    main_logits: torch.Tensor,
    mtp_logits: List[torch.Tensor]|torch.Tensor,
    input_ids: torch.Tensor,
    mtp_lambda: float = 0.3,
    ignore_idx: int = -100,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute combined loss for main LM head output + MTP predictions

    Args:
        main_logits: (batch_size, seq_len, vocab_size)
        mtp_logits: List of (batch_size, seq_len, vocab_size) tensors, one per MTP depth
        input_ids: (batch_size, seq_len) ground truth token ids
        mtp_lambda: Weight for MTP losses (default: 0.3 from paper)
        ignore_idx: Toke id to ignore in loss (eg: padding token)

    Returns:
        total_loss: Combined weighted loss
        loss_dict: Dictionary of individual losses for logging
    """
    vocab_size = main_logits.size(-1)
    batch_size, seq_len = input_ids.shape

    loss_dict = {}

    # Depth 0
    # predict token at position i+1 from position i
    loss_main = torch.nn.functional.cross_entropy(
        main_logits.reshape(-1, vocab_size),
        input_ids.reshape(-1),
        ignore_index=ignore_idx,
    )
    loss_dict["loss_depth_0"] = loss_main
    total_loss = loss_main

    # MTP depths: 1, 2, ..., mtp_depth-1
    for depth, logits_d in enumerate(mtp_logits):
        target_offset = depth + 1

        if seq_len <= target_offset:
            break

        targets = input_ids[:, target_offset:]

        # Truncate logits to match targets length
        # (Last few positions don't have targets)
        valid_len = min(logits_d.size(1), targets.size(1))
        if valid_len <= 0:
            break

        logits_truncated = logits_d[:, :valid_len].contiguous()
        targets_truncated = targets[:, :valid_len].contiguous()

        loss_d = torch.nn.functional.cross_entropy(
            logits_truncated.view(-1, vocab_size),
            targets_truncated.view(-1),
            ignore_index=ignore_idx,
        )

        loss_dict[f"loss_depth_{depth + 1}"] = loss_d
        total_loss = total_loss + (mtp_lambda * loss_d)

    loss_dict["total_loss"] = total_loss
    return total_loss, loss_dict

# Test Driver code:
if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        hidden_dim: int = 256
        vocab_size: int = 1000
        mtp_depth: int = 2  # Predict +2 and +3 ahead
    
    config = TestConfig()
    
    # Create shared embedding
    embedding = torch.nn.Embedding(config.vocab_size, config.hidden_dim)
    
    # Create MTP module
    mtp = MultiTokenPrediction(config, embedding)
    
    # Test inputs
    batch_size = 2
    seq_len = 10
    
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_dim)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print("=" * 50)
    print("Testing MultiTokenPrediction")
    print("=" * 50)
    print(f"Input hidden_states: {hidden_states.shape}")
    print(f"Input input_ids: {input_ids.shape}")
    print(f"MTP depth: {config.mtp_depth}")
    
    # Forward pass
    mtp_logits = mtp(hidden_states, input_ids)
    
    print("\nMTP outputs:")
    for i, logits in enumerate(mtp_logits):
        print(f"  Depth {i+1} logits: {logits.shape} (predicts +{i+2} ahead)")
    
    # Test loss computation
    print("\n" + "=" * 50)
    print("Testing MTP Loss")
    print("=" * 50)
    
    # Simulate main model logits
    main_logits = torch.randn(batch_size, seq_len, config.vocab_size)
    
    total_loss, loss_dict = compute_mtp_loss(
        main_logits=main_logits,
        mtp_logits=mtp_logits,
        input_ids=input_ids,
        mtp_lambda=0.3,
    )
    
    print("\nLoss breakdown:")
    for name, loss in loss_dict.items():
        print(f"  {name}: {loss.item():.4f}")
    
    # Verify shapes make sense
    print("\nShape verification:")
    print(f"  main_logits[:, :-1]: {main_logits[:, :-1].shape} -> targets {input_ids[:, 1:].shape}")
    for i, logits in enumerate(mtp_logits):
        target_offset = i + 2
        targets = input_ids[:, target_offset:]
        valid_len = targets.shape[1]
        print(f"  mtp_logits[{i}][:, :{valid_len}]: {logits[:, :valid_len].shape} -> targets {targets.shape}")
    
    print("\nMTP test passed!")