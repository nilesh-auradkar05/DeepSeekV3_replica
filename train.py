"""
DeepSeekV3 Nano Training Script
===============================

Main entry point for training the DeepSeek-V3 Nano model.

Usage:
    python3/python train.py
    python3/python train.py --optimizer muon
    python3/python train.py --strategy fsdp --num_gpus 4
"""

import argparse
import yaml
from functools import partial
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger

from configs.model_config import get_1b_config, get_nano_config, get_small_config, get_medium_config
from data.data_module import FineWebDataModule, DummyDataModule
from training.lightning_module import DeepSeekV3LightningModule
from lightning.pytorch.callbacks import Callback

from loguru import logger

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Literal, cast
type PrecisionType = Literal["16-mixed", "bf16-mixed", "32-true", "16-true", "bf16-true", 32, 16, 64]

class TrainingLogger(Callback):
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.start_time = datetime.now()
        module = cast(DeepSeekV3LightningModule, pl_module)

        print("=" * 50)
        print("Training Started")
        print("=" * 50)
        print(f"Model: {module.model_config.hidden_dim}d, {module.model_config.num_layers}L")
        print(f"Total parameters: {module.model.get_num_params():,}")
        print(f"Activated parameters: {module.model.get_num_activated_params():,}")
        print(f"Max steps: {trainer.max_steps}")
        print(f"Precision: {trainer.precision}")
        print(f"Devices: {trainer.num_devices}")
        print("=" * 50 + "\n")

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if trainer.global_step % self.log_every_n_steps == 0 and trainer.global_step > 0:
            # Get current metrics
            loss = trainer.callback_metrics.get("train/loss", 0)
            lr = trainer.callback_metrics.get("train/lr", 0)

            # Cacluclat Throughput
            if self.start_time:
                elapsed = (datetime.now() - self.start_time).total_seconds()
                steps_per_sec = trainer.global_step / elapsed
                tokens_per_sec = steps_per_sec * trainer.accumulate_grad_batches * batch["input_ids"].numel()

                print(
                    f"Step {trainer.global_step:6d} |"
                    f"Loss: {loss:.4f} |"
                    f"LR: {lr:.2e} |"
                    f"Tokens/sec: {tokens_per_sec:.0f} |"
                )

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        val_loss = trainer.callback_metrics.get("val/loss", 0)
        val_ppl = trainer.callback_metrics.get("val/perplexity", 0)
        print(f"\nValidation | Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}\n")
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        print("\n" + "=" * 60)
        print("Training Completed")
        print("=" * 60)
        
        if self.start_time:
            total_time = datetime.now() - self.start_time
            print(f"Total time: {total_time}")
        
        if trainer.checkpoint_callback:
            best_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
            if best_path:
                print(f"Best checkpoint: {best_path}")

def get_model_config(preset: str):
    """Get model configuration by preset name."""
    presets = {
        "nano": get_nano_config,
        "small": get_small_config,
        "1b": get_1b_config,
        "medium": get_medium_config,
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset]()

def get_strategy(strategy_name: str, num_gpus: int):
    """Get Lightning training strategy."""
    if num_gpus <= 1:
        return "auto"
    
    if strategy_name in ("auto", "ddp"):
        return "ddp"
    
    if strategy_name == "fsdp":
        from lightning.pytorch.strategies import FSDPStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from src.deepseek_transformer import DeepSeekV3TransformerBlock
        
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={DeepSeekV3TransformerBlock},
        )
        
        return FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing_policy={DeepSeekV3TransformerBlock},
            sharding_strategy="FULL_SHARD",
            cpu_offload=False,
        )
    
    if strategy_name == "deepspeed_stage_2":
        from lightning.pytorch.strategies import DeepSpeedStrategy
        return DeepSpeedStrategy(
            stage=2,
            offload_optimizer=False,
            allgather_bucket_size=int(2e8),
            reduce_bucket_size=int(2e8),
        )
    
    if strategy_name == "deepspeed_stage_3":
        from lightning.pytorch.strategies import DeepSpeedStrategy
        return DeepSpeedStrategy(
            stage=3,
            offload_optimizer=False,
            allgather_bucket_size=int(2e8),
            reduce_bucket_size=int(2e8),
        )
    
    return strategy_name


def setup_loggers(args) -> list:
    """Setup experiment loggers."""
    loggers = []
    
    if args.use_wandb:
        try:
            from lightning.pytorch.loggers import WandbLogger
            loggers.append(WandbLogger(
                project=args.wandb_project or "deepseek-v3-nano",
                name=args.run_name,
            ))
            print("W&B logging enabled")
        except ImportError:
            print("Warning: wandb not installed, skipping W&B logging")
    
    if args.use_mlflow:
        try:
            from lightning.pytorch.loggers import MLFlowLogger
            loggers.append(MLFlowLogger(
                experiment_name=args.mlflow_experiment or "deepseek-v3-nano",
                tracking_uri=args.mlflow_uri or "mlruns",
                run_name=args.run_name,
            ))
            print("MLflow logging enabled")
        except ImportError:
            print("Warning: mlflow not installed, skipping MLflow logging")
    
    return loggers

def main():
    """Main training entry point."""
    # Disable tokenizer parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Enable TF32 for faster matmul on Ampere+ GPUs
    torch.set_float32_matmul_precision("high")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train DeepSeek-V3 Nano")
    
    # Model
    parser.add_argument("--preset", type=str, default="nano",
                        choices=["nano", "small", "1b", "medium"],
                        help="Model size preset")
    
    # Data
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", type=str, default="sample-10BT")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_tokens", type=int, default=1_000_000_000,
                        help="Total tokens to train on")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max training steps (computed from max_tokens if not set)")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="muon",
                        choices=["muon", "adamw"])
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    
    # MoE
    parser.add_argument("--moe_bias_update_speed", type=float, default=0.001)
    
    # Hardware
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs (auto-detect if not set)")
    parser.add_argument("--strategy", type=str, default="auto",
                        choices=["auto", "ddp", "fsdp", "deepspeed_stage_2", "deepspeed_stage_3"])
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    
    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint path")
    
    # Logging
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--val_steps", type=int, default=500)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--use_mlflow", action="store_true")
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    
    # Debug
    parser.add_argument("--debug", action="store_true",
                        help="Quick test with dummy data and 50 steps")
    parser.add_argument("--seed", type=int, default=47)
    
    args = parser.parse_args()
    
    # Set seed
    L.seed_everything(args.seed, workers=True)
    
    # Debug mode overrides
    if args.debug:
        print("\n*** DEBUG MODE ***\n")
        args.max_steps = 50
        args.batch_size = 4
        args.max_seq_len = 256
        args.log_steps = 5
        args.val_steps = 25
        os.environ["WANDB_MODE"] = "disabled"
    
    # Auto-detect GPUs
    if args.num_gpus is None:
        args.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"deepseek-{args.preset}-{timestamp}"
    
    # Get model config
    model_config = get_model_config(args.preset)
    
    # Setup data module
    if args.debug:
        print("Using dummy data for debug mode")
        data_module = DummyDataModule(
            vocab_size=model_config.vocab_size,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            train_samples=1000,
            val_samples=100,
        )
    else:
        # Get vocab size from tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer_vocab_size = len(tokenizer)
        
        # Sync model vocab size
        if model_config.vocab_size != tokenizer_vocab_size:
            print(f"Syncing vocab_size: {model_config.vocab_size} -> {tokenizer_vocab_size}")
            model_config.vocab_size = tokenizer_vocab_size
        
        data_module = FineWebDataModule(
            dataset_name=args.dataset,
            subset=args.subset,
            tokenizer_name=args.tokenizer,
            max_seq_len=args.max_seq_len,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    
    # Compute max_steps from token budget if not specified
    if args.max_steps is None:
        tokens_per_step = args.batch_size * args.gradient_accumulation * args.max_seq_len
        args.max_steps = args.max_tokens // tokens_per_step
        print(f"Computed max_steps: {args.max_steps:,} ({args.max_tokens:,} tokens)")
    
    # Check optimizer availability
    optimizer = args.optimizer
    if optimizer == "muon" and not hasattr(torch.optim, "Muon"):
        print("Warning: Muon optimizer not available (requires PyTorch 2.9+), using AdamW")
        optimizer = "adamw"
    
    # Create Lightning module
    module = DeepSeekV3LightningModule(
        model_config=model_config,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        optimizer=optimizer,
        muon_lr=args.muon_lr,
        moe_bias_update_speed=args.moe_bias_update_speed,
        gradient_clip_val=args.gradient_clip,
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename=f"{args.run_name}" + "-step{step:06d}-loss{val/loss:.4f}",
            every_n_train_steps=args.save_steps,
            save_top_k=args.save_top_k,
            monitor="val/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        TrainingLogger(log_every_n_steps=args.log_steps),
    ]
    
    # Setup loggers
    loggers = setup_loggers(args)
    
    # Setup precision
    precision_map = {
        "bf16": "bf16-mixed",
        "fp16": "16-mixed",
        "fp32": "32-true",
    }
    precision = precision_map[args.precision]
    
    # Setup strategy
    strategy = get_strategy(args.strategy, args.num_gpus)
    
    # Print configuration
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    print(f"Model preset: {args.preset}")
    print(f"Hidden dim: {model_config.hidden_dim}")
    print(f"Layers: {model_config.num_layers} ({model_config.num_dense_layers} dense, {model_config.num_moe_layers} MoE)")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"Max steps: {args.max_steps:,}")
    print(f"Max tokens: {args.max_tokens:,}")
    print(f"Optimizer: {optimizer}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Precision: {precision}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Strategy: {args.strategy}")
    print("=" * 60 + "\n")
    
    # Create trainer
    trainer = L.Trainer(
        accelerator="cuda" if args.num_gpus > 0 else "cpu",
        devices=args.num_gpus if args.num_gpus > 0 else "auto",
        strategy=strategy,
        precision=cast(PrecisionType, precision),
        max_steps=args.max_steps,
        accumulate_grad_batches=args.gradient_accumulation,
        val_check_interval=min(args.val_steps, args.max_steps),
        log_every_n_steps=args.log_steps,
        callbacks=callbacks,
        logger=loggers if loggers else None,
        enable_progress_bar=True,
        gradient_clip_val=None,  # Handled manually in LightningModule
    )
    
    # Train
    trainer.fit(
        module,
        data_module,
        ckpt_path=args.resume_from,
    )
    
    print(f"\nTraining complete! Checkpoints saved to: {args.output_dir}")

if __name__ == "__main__":
    main()