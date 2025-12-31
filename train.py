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

from typing import Literal, Any, cast, Optional
type PrecisionType = Literal["16-mixed", "bf16-mixed", "32-true", "16-true", "bf16-true", 32, 16, 64]

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(log_dir: str = "logs", run_name: Optional[str] = None):
    """Configure loguru for file and console logging."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or "train"
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"

    # Remove default handler
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
        colorize=True,
    )

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="7 days",
    )

    logger.info(f"Logging to: {log_file}")
    return log_file

class LoggingCallback(Callback):
    """Callback to log training progress and model outputs using loguru."""
    
    def __init__(self, log_every_n_steps: int = 100, generate_samples: bool = True):
        self.log_every_n_steps = log_every_n_steps
        self.generate_samples = generate_samples
    
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        pl_module = cast(DeepSeekV3LightningModule, pl_module)
        logger.info("=" * 50)
        logger.info("Training Started")
        logger.info("=" * 50)
        logger.info(f"Total params: {pl_module.model.get_num_params():,}")
        logger.info(f"Activated params: {pl_module.model.get_num_activated_params():,}")
        logger.info(f"Max steps: {trainer.max_steps}")
        logger.info(f"Precision: {trainer.precision}")
    
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int):
        if trainer.global_step % self.log_every_n_steps == 0:
            loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            logger.info(f"Step {trainer.global_step} | Loss: {loss:.4f} | LR: {lr:.2e}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        logger.info(f"Epoch {epoch} completed")
        
        # Log metrics
        metrics = trainer.callback_metrics
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            logger.info(f"  {key}: {value:.4f}")
        
        # Generate sample output
        if self.generate_samples:
            self._generate_sample(pl_module, epoch)
    
    def _generate_sample(self, pl_module, epoch: int):
        """Generate a sample from the model to monitor training progress."""
        logger.info(f"Generating sample output (epoch {epoch})...")
        
        try:
            pl_module.eval()
            device = next(pl_module.parameters()).device
            
            # Simple prompt tokens (using common token IDs)
            prompt = torch.tensor([[464, 1917, 318, 257]], device=device)  # "The world is a"
            
            with torch.no_grad():
                output = pl_module.model.generate(
                    prompt,
                    max_new_tokens=32,
                    temperature=0.8,
                    top_k=50,
                )
            
            # Log token IDs (since we may not have tokenizer)
            generated_tokens = output[0].tolist()
            logger.info(f"Generated tokens: {generated_tokens[:50]}...")
            
            pl_module.train()
            
        except Exception as e:
            logger.warning(f"Sample generation failed: {e}")
    
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val/loss")
        val_ppl = metrics.get("val/perplexity")
        
        if val_loss is not None:
            logger.info(f"Validation | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}" if val_ppl else f"Validation | Loss: {val_loss:.4f}")
    
    def on_train_end(self, trainer, pl_module):
        logger.info("=" * 50)
        logger.info("Training Completed")
        logger.info("=" * 50)
        
        if isinstance(trainer.checkpoint_callback, ModelCheckpoint) and trainer.checkpoint_callback.best_model_path:
            logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")

def setup_mlflow_databricks(cfg: dict) -> "MLFlowLogger":
    """
    Setup MLflow with Databricks backend.

    Required config:
        databricks_host: https://<workspace>.cloud.databricks.com
        databricks_token: dapi....
        experiment_name: /Shared/deepseek-v3-nano

    Optional:
        workspace_id: For multi-workspace setups.
        run_id: Resume existing run
    """
    import mlflow
    from dotenv import load_dotenv
    import os
    from lightning.pytorch.loggers import MLFlowLogger

    load_dotenv()
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    experiment_name = cfg.get("logging", {}).get("experiment_name")
    workspace_id = cfg.get("logging", {}).get("workspace_id")
    run_id = cfg.get("logging", {}).get("run_id")
    mlflow_registry_uri = cfg.get("logging", {}).get("mlflow_registry_uri")

    if not host or not token:
        print("Warning: Databricks host and token are required for MLflow logging.")
        print("Falling back to local MLflow tracking.")
        return MLFlowLogger(experiment_name=experiment_name or "deepseek-v3-nano")

    # Set environment variables for Databricks auth
    os.environ["MLFLOW_TRACKING_URI"] = cfg.get("logging", {}).get("mlflow_tracking_uri", "databricks")
    os.environ["MLFLOW_REGISTRY_URI"] = cfg.get("logging", {}).get("mlflow_registry_uri", "databricks-uc")

    if workspace_id:
        tracking_uri = f"databricks://{workspace_id}"
    else:
        tracking_uri = "databricks"

    mlflow.set_tracking_uri(tracking_uri)

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    print("MLflow Databricks configured:")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Workspace ID: {workspace_id}")
    print(f"  Run ID: {run_id}")

    return MLFlowLogger(
        experiment_name=experiment_name or "deepseek-v3-nano",
        tracking_uri=tracking_uri,
        run_id=run_id
    )

def setup_wandb(cfg: dict) -> "WandbLogger":
    """Setup Weights & Biases logger."""

    from lightning.pytorch.loggers import WandbLogger

    return WandbLogger(
        project=cfg.get("logging", {}).get("wandb_project", "deepseek-v3-nano"),
        name=cfg.get("logging", {}).get("run_name"),
        entity=cfg.get("logging", {}).get("wandb_entity"),
    )

def get_strategy(strategy_name: str, num_gpus: int):
    """
    Get Lightning strategy based on name.

    Strategies:
        - auto: Automatically choose based on num_gpus
        - ddp : Standard DDP
        - fsdp : Fully sharded data parallel
        - deepspeed_stage_2 : DeepSpeed ZeRO-2 (optimizer + gradient sharding)
        - deepspeed_stage_3 : DeepSpeed ZeRO-3 (full parameter sharding)
    """

    if num_gpus == 1:
        return "auto"

    if strategy_name == "auto" or strategy_name == "ddp":
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

def main():
    from typing import cast
    parser = argparse.ArgumentParser(description="Train DeepSeek-V3 Nano")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--num_gpus", type=int)
    parser.add_argument("--strategy", type=str)
    parser.add_argument("--optimizer", type=str, choices=["adamw", "muon"])
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--log_dir", type=str, default="logs")
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    # Apply CLI overrides
    if args.learning_rate:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.max_steps:
        cfg["training"]["max_steps"] = args.max_steps
    if args.num_gpus:
        cfg["training"]["num_gpus"] = args.num_gpus
    if args.strategy: 
        cfg["training"]["strategy"] = args.strategy
    if args.optimizer: 
        cfg["training"]["optimizer"] = args.optimizer
    if args.run_name: 
        cfg["logging"]["run_name"] = args.run_name
    if args.run_id: 
        cfg["logging"]["run_id"] = args.run_id
    
    # Debug mode
    if args.debug:
        cfg["training"]["max_steps"] = 50
        cfg["training"]["batch_size"] = 4
        cfg["training"]["max_seq_len"] = 256
        cfg["logging"]["log_steps"] = 5
    
    # Setup logging
    log_file = setup_logging(args.log_dir, cfg.get("logging", {}).get("run_name"))
    
    # Log config
    logger.info("=" * 50)
    logger.info("DeepSeek-V3 Nano Training")
    logger.info("=" * 50)
    for k, v in cfg["training"].items():
        if "token" in k.lower():
            logger.debug(f"{k}: ***")
        else:
            logger.info(f"{k}: {v}")
    
    # Check optimizer
    optimizer_name = cfg["training"].get("optimizer", "adamw")
    if optimizer_name == "muon":
        if hasattr(torch.optim, "Muon"):
            logger.info("Using Muon optimizer (PyTorch 2.9+)")
        else:
            logger.warning("Muon not available, falling back to AdamW")
            optimizer_name = "adamw"
    
    L.seed_everything(cfg.get("seed", 42), workers=True)
    
    # Model config
    presets = {"nano": get_nano_config, "small": get_small_config, "medium": get_medium_config}
    model_config = presets[cfg["training"].get("preset", "nano")]()
    
    # Data
    if args.debug:
        logger.info("Using dummy data for debug")
        data_module = DummyDataModule(
            vocab_size=model_config.vocab_size,
            max_seq_len=cfg["training"].get("max_seq_len", 2048),
            batch_size=cfg["training"]["batch_size"],
        )
    else:
        max_tokens = cfg["training"].get("max_tokens", 1_000_000_000)
        logger.info(f"Using FineWeb-Edu ({max_tokens/1e9:.1f}B tokens)")
        data_module = FineWebDataModule(
            dataset_name=cfg["training"].get("dataset", "HuggingFaceFW/fineweb-edu"),
            subset=cfg["training"].get("subset", "sample-10BT"),
            tokenizer_name=cfg["training"].get("tokenizer", "gpt2"),
            max_seq_len=cfg["training"].get("max_seq_len", 2048),
            max_tokens=max_tokens,
            batch_size=cfg["training"]["batch_size"],
            num_workers=cfg["training"].get("num_workers", 4),
            seed=cfg["training"].get("seed", 42),
        )
    
    # Model
    module = DeepSeekV3LightningModule(
        model_config=model_config,
        learning_rate=cfg["training"]["learning_rate"],
        min_learning_rate=cfg["training"].get("min_learning_rate", 3e-5),
        weight_decay=cfg["training"].get("weight_decay", 0.1),
        warmup_steps=cfg["training"].get("warmup_steps", 500),
        max_steps=cfg["training"]["max_steps"],
        moe_bias_update_speed=cfg["training"].get("moe_bias_update_speed", 0.001),
        optimizer=optimizer_name,
        muon_lr=cfg["training"].get("muon_lr", 0.02),
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg["training"].get("output_dir", "checkpoints"),
            filename="step-{step:06d}-loss-{val/loss:.4f}",
            every_n_train_steps=cfg["training"].get("save_steps", 1000),
            save_top_k=3,
            monitor="val/loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
        LoggingCallback(
            log_every_n_steps=cfg["logging"].get("log_steps", 100),
            generate_samples=cfg["logging"].get("generate_samples", True),
        ),
    ]
    
    # Loggers
    loggers = []
    if cfg["logging"].get("databricks_host") or cfg["logging"].get("experiment_name"):
        loggers.append(setup_mlflow_databricks(cfg))
    if cfg["logging"].get("wandb_project"):
        from lightning.pytorch.loggers import WandbLogger
        loggers.append(WandbLogger(project=cfg["logging"]["wandb_project"], name=cfg["logging"].get("run_name")))
    
    # Precision and strategy
    precision_map = {"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}
    precision = precision_map.get(cfg["training"].get("precision", "bf16"), "bf16-mixed")
    
    num_gpus = cfg["training"].get("num_gpus", 1) if torch.cuda.is_available() else 0
    strategy = get_strategy(cfg["training"].get("strategy", "auto"), num_gpus)
    
    logger.info(f"Strategy: {cfg["training"].get("strategy", "auto")} | GPUs: {num_gpus} | Optimizer: {optimizer_name}")
    
    # Trainer
    trainer = L.Trainer(
        accelerator="cuda" if num_gpus > 0 else "cpu",
        devices=num_gpus if num_gpus > 0 else "auto",
        strategy=strategy,
        precision=cast(PrecisionType, precision),
        max_steps=cfg["training"]["max_steps"],
        gradient_clip_val=cfg["training"].get("max_grad_norm", 1.0),
        accumulate_grad_batches=cfg["training"].get("gradient_accumulation_steps", 1),
        val_check_interval=min(500, cfg["training"]["max_steps"]),
        log_every_n_steps=cfg["logging"].get("log_steps", 10),
        callbacks=callbacks,
        logger=loggers if loggers else None,
        enable_progress_bar=True,
    )
    
    logger.info("Starting training...")
    trainer.fit(module, data_module)
    
    logger.info(f"Training complete! Log file: {log_file}")


if __name__ == "__main__":
    main()