"""
DeepSeek-V3 Nano: Data Module
=============================

Handles data loading and preprocessing for language model training.
"""
from datasets import IterableDataset
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from typing import Optional, List

class TextDataset(Dataset):
    """
    Dataset for Language Model Training.

    Handles tokenization and chunking of text data.
    """

    def __init__(
        self,
        data: List[str],
        tokenizer,
        max_seq_len: int = 2048,
    ):

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Tokenize all texts and concatenate
        all_tokens = []
        for text in data:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

        # Chunk into sequences of max_seq_len
        self.chunks = []
        for i in range(0, len(all_tokens), max_seq_len):
            chunk = all_tokens[i:i+max_seq_len+1] # +1 for the Next token prediction
            if len(chunk) == max_seq_len + 1:
                self.chunks.append(chunk)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

class DeepSeekDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for DeepSeek-V3 training."""

    def __init__(
        self,
        dataset_name: str = "openwebtext",
        train_split: str = "train",
        val_split: str = "validation",
        tokenizer_name: str = "gpt4",
        max_seq_len: int = 2048,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.train_split = train_split
        self.val_split = val_split
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        """Download and prepare the dataset."""
        from datasets import load_dataset
        from transformers import AutoTokenizer

        # Download dataset
        load_dataset(self.dataset_name, split=self.train_split, trust_remote_code=True)

        # Download tokenizer
        AutoTokenizer.from_pretrained(self.tokenizer_name)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets (called on every GPU)"""
        from datasets import load_dataset
        from transformers import AutoTokenizer

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load datasets
        if stage == "fit" or stage is None:
            train_data = load_dataset(
                self.dataset_name,
                split=self.train_split,
                trust_remote_code=True,
            )

            train_texts: list[str] | None = None
            # Extract text field (handle different dataset formats)
            if "text" in train_data.column_names:
                train_texts = train_data["text"]
            elif "content" in train_data.column_names:
                train_texts = train_data["content"]
            else:
                # Assume first column is text
                for col in train_data.column_names:
                    if isinstance(train_data[0][col], str):
                        train_texts = train_data[col]
                        break

            if train_texts is None:
                raise ValueError(
                    f"No text field found in dataset {self.dataset_name}"
                )

            self.train_dataset = TextDataset(
                train_texts,
                self.tokenizer,
                self.max_seq_len,
            )

            val_texts: list[str] | None = None

            # Validation set
            try:
                val_data = load_dataset(
                    self.dataset_name,
                    split=self.val_split,
                    trust_remote_code=True,
                )
                if "text" in val_data.column_names:
                    val_texts = val_data["text"]
                elif "content" in val_data.column_names:
                    val_texts = val_data["content"]
                else:
                    for col in val_data.column_names:
                        if isinstance(val_data[0][col], str):
                            val_texts = val_data[col]
                            break

                if val_texts is None:
                    raise ValueError(
                        f"No text field found in dataset {self.dataset_name}"
                    )

                self.val_dataset = TextDataset(
                    val_texts,
                    self.tokenizer,
                    self.max_seq_len,
                )

            except Exception:
                # If no validation split, use part of training
                print("No validation split found. Using 5% of training data as validation data.")
                n_val = max(100, len(self.train_dataset) // 20)
                self.val_dataset = torch.utils.data.Subset(
                    self.train_dataset,
                    range(n_val),
                )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("train_dataset is not set.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError("val_dataset is not set.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from tokenizer"""
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            return tokenizer.vocab_size
        return self.tokenizer.vocab_size

class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for pre-training.
    Streams from HuggingFace, tokenizes on-the-fly, and stops at max_tokens.
    """
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        subset: str = "sample-10BT",
        split: str = "train",
        tokenizer_name: str = "gpt4",
        max_seq_len: int = 2048,
        max_tokens: int = 5_000_000_000,
        seed: int = 47,
    ):

        self.dataset_name = dataset_name
        self.subset = subset
        self.data_split = split
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.max_tokens = max_tokens
        self.seed = seed
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def __iter__(self):
        from datasets import load_dataset

        dataset = load_dataset( # type: ignore[call-overload]
            self.dataset_name,
            subset=self.subset,
            split=self.data_split,
            streaming=True,
            trust_remote_code=True,
        )
        dataset = dataset.shuffle(seed=self.seed, buffer_size=50_000)

        token_buffer = []
        tokens_yielded = 0

        for sample in dataset:
            text = sample.get("text", "")
            if not text:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)

            # Yield chunks
            while len(token_buffer) >= self.max_seq_len + 1:
                chunk = token_buffer[:self.max_seq_len + 1]
                token_buffer = token_buffer[self.max_seq_len:]

                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }

                tokens_yielded += self.max_seq_len

                # Stop at token limit
                if tokens_yielded >= self.max_tokens:
                    return


class FineWebDataModule(L.LightningDataModule):
    """
    Lightning DataModule for FineWeb-Edu pre-training.

    Available subsets:
        - sample-10BT: ~10B Tokens (~40GB)
        - sample-100BT: ~100B Tokens (~400GB)
        - sample-350BT: ~350B Tokens (~1.4TB)
    """
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        subset: str = "sample-10BT",
        tokenizer_name: str = "gpt4",
        max_seq_len: int = 2048,
        max_tokens: int = 5_000_000_000,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 47,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.subset = subset
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = StreamingTextDataset(
                dataset_name=self.dataset_name,
                subset=self.subset,
                tokenizer_name=self.tokenizer_name,
                max_seq_len=self.max_seq_len,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )

            self.val_dataset = DummyDataset(100_000, self.max_seq_len, 5000)

    def train_dataloader(self):
        from typing import cast
        
        return DataLoader(
            cast(Dataset, self.train_dataset),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size
        )

class DummyDataModule(L.LightningDataModule):
    """
    Dummy data module for testing without real data.
    Generates random token sequences.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_len: int = 2048,
        batch_size: int = 32,
        train_samples: int = 10000,
        val_samples: int = 1000,
        num_workers: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.num_workers = num_workers
    
    def setup(self, stage: Optional[str] = None):
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
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class DummyDataset(Dataset):
    """Random token dataset for testing."""
    
    def __init__(self, vocab_size: int, max_seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random tokens
        g = torch.Generator().manual_seed(idx)
        tokens = torch.randint(0, self.vocab_size, (self.max_seq_len + 1,), generator=g)
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:]
        }


# ============== Test ==============
if __name__ == "__main__":

    print("Testing Data Modules")

    
    # Test dummy data module
    print("\nTesting DummyDataModule...")
    dm = DummyDataModule(
        vocab_size=32000,
        max_seq_len=256,
        batch_size=4,
        train_samples=100,
        val_samples=20,
    )
    dm.setup()
    
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  Batch labels shape: {batch['labels'].shape}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(dm.val_dataloader())}")
    
    print("\nDummy data module test passed!")        