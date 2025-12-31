"""
DeepSekk-V3 Nano: Lightning Module
==================================

PyTorch Lightning module wrapping the DeepSeek-V3 nano model.

Handles:
    - Training/validation steps
    - Loss computation (main + MTP)
    - MoE load balancing updates
    - Learning rate scheduling
    - Metric logging
"""

from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
import lightning as L
from typing import Dict
import math

import sys
from pathlib import Path

from torch.optim import Optimizer
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deepseekv3 import DeepSeekV3ForCausalLM
from configs.model_config import DeepSeekV3Config

class DeepSeekV3LightningModule(L.LightningModule):
    """
    PyTorch Lightning module for DeepSeek V3

    Encapsulates training logic including:
        - Forward pass and loss computation
        - MoE load balancing updates
        - Optimizer and LR scheduler configuration
        - Metric logging
    """

    def __init__(
        self,
        model_config: DeepSeekV3Config,
        learning_rate: float = 3e-4,
        min_learning_rate: float = 3e-5,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100_000,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
        adam_epsilon: float = 1e-8,
        lr_scheduler: str = "cosine",
        optimizer: str = "muon",
        muon_lr: float = 0.02,
        moe_bias_update_speed: float = 0.001,
    ):
        super().__init__()

        # Save hyperparameters (except model_config)
        self.save_hyperparameters(ignore=["model_config"])

        # Store config
        self.model_config = model_config
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.lr_scheduler_type = lr_scheduler
        self.optimizer_type = optimizer
        self.muon_lr = muon_lr
        self.moe_bias_update_speed = moe_bias_update_speed

        # Create model
        self.model = DeepSeekV3ForCausalLM(model_config)

        # For training
        self.train_losses = []

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(input_ids, **kwargs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Single training step.

        Args:
            batch: Dict with 'input_ids'
            batch_idx: Index of the batch

        Returns:
            Loss Tensor
        """
        input_ids = batch["input_ids"]
        labels = batch.get("labels", None)

        loss_dict = self.model.compute_loss(input_ids, labels)
        total_loss = loss_dict["total_loss"]

        # Update MoE load balancing
        if loss_dict.get("routing_indices") and self.model_config.num_dense_layers < self.model_config.num_layers:
            self.model.update_moe_load_balancing(
                loss_dict["routing_indices"],
                self.moe_bias_update_speed,
            )

        # Log metrics
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss_depth_0", loss_dict["loss_depth_0"], on_step=True, on_epoch=False)

        # Log MPT losses if present
        for key, value in loss_dict.items():
            if key.startswith("loss_depth_") and key != "loss_depth_0":
                if isinstance(value, torch.Tensor):
                    self.log(f"train/{key}", value, on_step=True, on_epoch=False)

        # Log Learning Rate
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Single validation step.
        """
        input_ids = batch["input_ids"]
        labels = batch.get("labels", None)

        # Compute loss
        loss_dict = self.model.compute_loss(input_ids, labels=labels)
        total_loss = loss_dict["total_loss"]

        # Log metrics
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss_depth_0", loss_dict["loss_depth_0"], on_step=False, on_epoch=True, sync_dist=True)

        # Compute Perplexity
        perplexity = torch.exp(loss_dict["loss_depth_0"])
        self.log("val/perplexity", perplexity, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure optimizer and learning rate scheduler.

        Uses Muon optimizer and cosine/linear LR schedule.
        """

        if self.optimizer_type == "muon":
            return self._configure_muon()
        else:
            return self._configure_adamw()

    def _configure_muon(self) -> OptimizerLRScheduler:
        """
        Muon optimizer for 2D hidden weights + AdamW for rest.

        Muon should only optimizer 2D weight matrices.
        Everything else uses AdamW (embeddings, lm_head, biases, norms).
        """
        # Separate parameters
        muon_params = []
        adamw_decay = []
        adamw_no_decay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Skip embeddings and lm_head
            is_embed = "embed" in name
            is_head = "lm_head" in name

            if param.dim() == 2 and not is_embed and not is_head:
                muon_params.append(param)
            elif param.dim() >= 2:
                # other 2D params (embed, head) -> AdamW with decay
                adamw_decay.append(param)
            else:
                # other params -> AdamW without decay
                adamw_no_decay.append(param) 
        
        print(f"Found {len(muon_params)} 2D weight matrices for Muon optimization")
        print(f"Found {len(adamw_decay)} 2D weight matrices for AdamW with decay")
        print(f"Found {len(adamw_no_decay)} non-2D weight matrices for AdamW without decay")

        optimizers = []
        schedulers = []

        # Muon for hidden weights
        if muon_params:
            muon_optimizer = torch.optim.Muon(
                muon_params,
                lr=self.muon_lr,
                momentum=0.95,
                nestrov=True,
                weight_decay=self.weight_decay,
                ns_steps=5,
            )
            optimizers.append(muon_optimizer)
            schedulers.append(self._get_lr_scheduler(muon_optimizer))

        # AdamW for everything else
        adamw_params = []
        if adamw_decay:
            adamw_params.append({"params": adamw_decay, "weight_decay": self.weight_decay})
        if adamw_no_decay:
            adamw_params.append({"params": adamw_no_decay, "weight_decay": 0.0})

        if adamw_params:
            adamw_optimizer = torch.optim.AdamW(
                adamw_params,
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_epsilon,
            )
            optimizers.append(adamw_optimizer)
            schedulers.append(self._get_lr_scheduler(adamw_optimizer))

        if len(optimizers) == 1:
            return {
                "optimizer": optimizers[0],
                "lr_scheduler": {"scheduler": schedulers[0], "interval": "step", "frequency": 1},
            }
        else:
            return (
                [{"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}
                for opt, sched in zip(optimizers, schedulers)]
            )

    def _configure_adamw(self) -> OptimizerLRScheduler:
        """Standard AdamW optimizer."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.learning_rate, betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_epsilon)

        scheduler = self._get_lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
    
    def _get_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ):
        """Cosine schedule with linear warmup."""

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                # Linear warmup
                return current_step / max(1, self.warmup_steps)
            progress = (current_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            return max(
                self.min_learning_rate / self.learning_rate,
                0.5 * (1 + math.cos(math.pi * progress))
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)