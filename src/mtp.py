import torch
from typing import Tuple, List, Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.rms_norm import RMSNorm

class MTPBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim

        # Normalize incoming output from transformer block
        self.norm = RMSNorm(self.hidden_dim)

        # Project input to hidden dimension
        self.proj = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)

        # Transformer layer
        self.lm_head = torch.nn.Linear(self.hidden_dim, config.vocab_size, bias=False)

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
        hidden_states = self.norm(hidden_states)
        combined = torch.cat([hidden_states, token_embeddings], dim=-1)
        combined = self.proj(combined)
        logits = self.lm_head(hidden_states)
        return logits, hidden_states

class MultiTokenPrediction(torch.nn.Module):
    def __init__(self, config, shared_embeddings: torch.nn.Embedding):
        super().__init__()
        self.mtp_depth = config.mtp_depth
        self.embeddings = shared_embeddings

        # One block per mtp_depth
        self.mtp_blocks = torch.nn.ModuleList([
            MTPBlock(config) for _ in range(self.mtp_depth)
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
        all_logits = []

        for depth in range(self.mtp_depth):
            offset = depth + 1
            
            # Get token embeddings of tokens at offset position
            shifted_ids = input_ids[:, offset:]
            token_emb = self.embeddings(shifted_ids)

            # Truncate hidden states to match offset
            hidden_trunc = hidden_states[:, :token_emb.shape[1]]

            # Forward through MTP block
            logits, hidden_states_new = self.mtp_blocks[depth](hidden_trunc, token_emb)

            all_logits.append(logits)
            hidden_states = hidden_states_new
        
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
    loss_dict = {}
    vocab_size = main_logits.shape[-1]

    # Depth 0
    # predict token at position i+1 from position i
    logits_d0 = main_logits[:, :-1].contiguous()
    targets_d0 = input_ids[:, 1:].contiguous()

    loss_d0 = torch.nn.functional.cross_entropy(
        logits_d0.view(-1, vocab_size),
        targets_d0.view(-1),
        ignore_index=ignore_idx,
    )
    loss_dict["loss_d0"] = loss_d0
    total_loss = loss_d0

    # MTP depths: 1, 2, ..., mtp_depth-1
    for depth, logits in enumerate(mtp_logits):
        target_offset = depth + 2
        targets = input_ids[:, target_offset:].contiguous()

        # Truncate logits to match targets length
        # (Last few positions don't have targets)
        valid_len = targets.shape[1]
        logits = logits[:, :valid_len].contiguous()

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            ignore_index=ignore_idx,
        )

        loss_dict[f"mtp_loss_depth_{depth + 1}"] = loss
        total_loss = total_loss + (mtp_lambda * loss)

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
    
    print("\nMTP test passed! ✓")