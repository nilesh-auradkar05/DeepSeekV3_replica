"""
DeepSeek-V3 Data Module
=======================

Handles data loading and preprocessing for language model training.

Key features:
    - Streaming dataset from HuggingFace (memory efficient)
    - On-the-fly tokenization
    - Proper token buffer management (no gaps between sequences)
    - Configurable token budget for training runs

For validation, we sample real data from the dataset rather than
random tokens, ensuring meaningful validation metrics.
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import lightning as L
from typing import Optional, Iterator, List


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large-scale pretraining.
    
    Streams from HuggingFace datasets, tokenizes on-the-fly, and yields
    fixed-length sequences. Stops after max_tokens to control training budget.
    
    Important: Use num_workers=0 with this dataset to avoid sampler conflicts.
    """
    
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        subset: str = "sample-10BT",
        split: str = "train",
        tokenizer_name: str = "deepseek-ai/DeepSeek-V3",
        max_seq_len: int = 2048,
        max_tokens: int = 1_000_000_000,
        seed: int = 47,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.max_tokens = max_tokens
        self.seed = seed
        
        # Lazy-loaded tokenizer
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer on first access."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from tokenizer."""
        return len(self.tokenizer)
    
    def __iter__(self) -> Iterator[dict]:
        """
        Iterate through dataset, yielding tokenized sequences.
        
        Handles multi-worker sharding if workers are used.
        """
        from datasets import load_dataset
        
        # Handle multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # Load streaming dataset
        dataset = load_dataset(
            self.dataset_name,
            name=self.subset,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )
        
        # Shuffle with worker-specific seed for different data per worker
        dataset = dataset.shuffle(seed=self.seed + worker_id, buffer_size=10_000)
        
        # Token buffer for accumulating partial sequences
        token_buffer: List[int] = []
        tokens_yielded = 0
        tokens_per_worker = self.max_tokens // num_workers
        
        for sample_idx, sample in enumerate(dataset):
            # Shard samples across workers
            if sample_idx % num_workers != worker_id:
                continue
            
            text = sample.get("text", "")
            if not text or not text.strip():
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)
            
            # Yield complete sequences
            # Each sequence needs max_seq_len + 1 tokens (input + target)
            while len(token_buffer) >= self.max_seq_len + 1:
                # Extract sequence
                chunk = token_buffer[:self.max_seq_len + 1]
                
                # IMPORTANT: Remove exactly the tokens we used
                # This ensures no gaps between consecutive sequences
                token_buffer = token_buffer[self.max_seq_len + 1:]
                
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }
                
                tokens_yielded += self.max_seq_len
                
                # Check token budget
                if tokens_yielded >= tokens_per_worker:
                    return


class ValidationDataset(Dataset):
    """
    Fixed validation dataset with real data samples.
    
    Loads a fixed set of samples at initialization for consistent
    validation metrics across training.
    """
    
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        subset: str = "sample-10BT",
        tokenizer_name: str = "deepseek-ai/DeepSeek-V3",
        max_seq_len: int = 2048,
        num_samples: int = 500,
        seed: int = 47,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load and prepare samples
        self.samples = self._prepare_samples(
            dataset_name, subset, tokenizer, seed
        )
    
    def _prepare_samples(
        self,
        dataset_name: str,
        subset: str,
        tokenizer,
        seed: int,
    ) -> List[dict]:
        """Load and tokenize validation samples."""
        from datasets import load_dataset
        
        samples = []
        token_buffer = []
        
        # Load streaming dataset with fixed seed for reproducibility
        dataset = load_dataset(
            dataset_name,
            name=subset,
            split="train",  # Use train split, just different samples
            streaming=True,
            trust_remote_code=True,
        )
        dataset = dataset.shuffle(seed=seed + 1000, buffer_size=1000)
        
        for sample in dataset:
            text = sample.get("text", "")
            if not text or not text.strip():
                continue
            
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)
            
            while len(token_buffer) >= self.max_seq_len + 1 and len(samples) < self.num_samples:
                chunk = token_buffer[:self.max_seq_len + 1]
                token_buffer = token_buffer[self.max_seq_len + 1:]
                
                samples.append({
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                })
            
            if len(samples) >= self.num_samples:
                break
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class DummyDataset(Dataset):
    """
    Random token dataset for testing without real data.
    
    Useful for debugging model architecture and training loop.
    """
    
    def __init__(self, vocab_size: int, max_seq_len: int, num_samples: int):
        assert vocab_size > 0, "vocab_size must be positive"
        assert max_seq_len > 0, "max_seq_len must be positive"
        assert num_samples > 0, "num_samples must be positive"
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        # Use idx as seed for reproducible "random" data
        g = torch.Generator().manual_seed(idx)
        tokens = torch.randint(0, self.vocab_size, (self.max_seq_len + 1,), generator=g)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


class FineWebDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for DeepSeek-V3 training.
    
    Combines streaming training data with fixed validation data.
    """
    
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        subset: str = "sample-10BT",
        tokenizer_name: str = "deepseek-ai/DeepSeek-V3",
        max_seq_len: int = 2048,
        max_tokens: int = 1_000_000_000,
        batch_size: int = 16,
        num_workers: int = 0,  # Must be 0 for streaming
        pin_memory: bool = True,
        seed: int = 47,
        val_samples: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store parameters
        self.dataset_name = dataset_name
        self.subset = subset
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.val_samples = val_samples
        
        # Will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self._vocab_size = None
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from tokenizer."""
        if self._vocab_size is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self._vocab_size = len(tokenizer)
        return self._vocab_size
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for training and validation."""
        if stage == "fit" or stage is None:
            # Training dataset (streaming)
            self.train_dataset = StreamingTextDataset(
                dataset_name=self.dataset_name,
                subset=self.subset,
                tokenizer_name=self.tokenizer_name,
                max_seq_len=self.max_seq_len,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            
            # Validation dataset (fixed samples for consistent metrics)
            self.val_dataset = ValidationDataset(
                dataset_name=self.dataset_name,
                subset=self.subset,
                tokenizer_name=self.tokenizer_name,
                max_seq_len=self.max_seq_len,
                num_samples=self.val_samples,
                seed=self.seed,
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before train_dataloader()")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Streaming handles shuffling
            num_workers=0,  # Required for IterableDataset
            pin_memory=self.pin_memory,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before val_dataloader()")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


class DummyDataModule(L.LightningDataModule):
    """
    Dummy data module for testing without real data.
    
    Generates random token sequences for architecture debugging.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_len: int = 2048,
        batch_size: int = 32,
        train_samples: int = 10000,
        val_samples: int = 1000,
        num_workers: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Set up dummy datasets."""
        self.train_dataset = DummyDataset(
            self.vocab_size,
            self.max_seq_len,
            self.train_samples,
        )
        self.val_dataset = DummyDataset(
            self.vocab_size,
            self.max_seq_len,
            self.val_samples,
        )
    
    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before train_dataloader()")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before val_dataloader()")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )