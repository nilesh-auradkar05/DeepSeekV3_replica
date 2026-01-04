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

import torch
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from typing import Dict, List, cast
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deepseekv3 import DeepSeekV3ForCausalLM
from configs.model_config import DeepSeekV3Config
from src.MoE import DeepSeekMoE

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
        gradient_clip_val: float = 1.0,
    ):
        super().__init__()
        self.automatic_optimization = False


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
        self.gradient_clip_val = gradient_clip_val

        # Create model
        self.model = DeepSeekV3ForCausalLM(model_config)

        # Track accumulated routing indices for load balancing
        self._accumulated_routing_indices: List[torch.Tensor] = []

        # For training
        self.train_losses = []

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(input_ids, **kwargs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        if schedulers is not None and not isinstance(schedulers, list):
            schedulers = [schedulers]

        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)

        loss_dict = self.model.compute_loss(input_ids, labels=labels)
        loss = loss_dict["total_loss"]

        accumulate_steps = self.trainer.accumulate_grad_batches
        scaled_loss = loss / accumulate_steps

        self.manual_backward(scaled_loss)

        # Accumulate routing indices for load balancing
        if "routing_indices" in loss_dict:
            for idx_tensor in loss_dict["routing_indices"]:
                self._accumulated_routing_indices.append(idx_tensor.detach())

        should_step = (batch_idx + 1) % accumulate_steps == 0

        if should_step:

            if self.gradient_clip_val > 0.0:
                for opt in optimizers:
                    self.clip_gradients(
                        opt.optimizer,
                        gradient_clip_val=self.gradient_clip_val,
                        gradient_clip_algorithm="norm",
                    )

            for opt in optimizers:
                opt.step()
                opt.zero_grad()

            if schedulers:
                for sched in schedulers:
                    sched.step()  # type: ignore

            # Update MoE load balancing ONLY after the step boundary
            self._update_moe_load_balancing()

            # clear accumulated routing indices
            self._accumulated_routing_indices.clear()

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log individual loss
        if "loss_depth_0" in loss_dict:
            self.log("train/loss_lm", loss_dict["loss_depth_0"], on_step=True, sync_dist=True)

        # Log MTP losses
        for key, value in loss_dict.items():
            if key.startswith("loss_depth_") and key != "loss_depth_0":
                self.log(f"train/{key}", value, on_step=True, sync_dist=True)

        # Log learning rate
        if optimizers:
            lr = optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", lr, on_step=True, sync_dist=True)

        return loss

    def _update_moe_load_balancing(self) -> None:
        """Update MoE router biases based on accumulated routing indices."""
        if not self._accumulated_routing_indices:
            return
        
        # Find all MoE layers
        moe_layers = []
        for module in self.model.modules():
            if isinstance(module, DeepSeekMoE):
                moe_layers.append(module)
        
        if not moe_layers:
            return
        
        # Group routing indices by layer
        # Each forward pass produces one tensor per MoE layer
        num_moe_layers = len(moe_layers)
        if len(self._accumulated_routing_indices) < num_moe_layers:
            return
        
        # Process each MoE layer
        for layer_idx, moe_layer in enumerate(moe_layers):
            # Collect routing indices for this layer across accumulation steps
            layer_indices = []
            for step_offset in range(0, len(self._accumulated_routing_indices), num_moe_layers):
                idx = step_offset + layer_idx
                if idx < len(self._accumulated_routing_indices):
                    layer_indices.append(self._accumulated_routing_indices[idx])
            
            if layer_indices:
                # Concatenate and update
                combined_indices = torch.cat(layer_indices, dim=0)
                moe_layer.update_load_balancing(
                    combined_indices,
                    bias_update_speed=self.moe_bias_update_speed,
                )

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Single validation step.
        """
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)

        # Compute loss
        loss_dict = self.model.compute_loss(input_ids, labels=labels)
        total_loss = loss_dict["total_loss"]

        # Log metrics
        self.log("val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Log main LM loss
        if "loss_depth_0" in loss_dict:
            main_loss = loss_dict["loss_depth_0"]
            self.log("val/loss_lm", main_loss, on_step=False, on_epoch=True, sync_dist=True)

            # Compute Perplexity
            perplexity = torch.exp(main_loss.detach().clamp(max=100))
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

        # Check if Muon is available
        if not hasattr(torch.optim, "Muon"):
            print("Warning: Muon optimizer not available, Update to Pytorch 2.9+! Falling back to AdamW!!")
            return self._configure_adamw()

        # Separate parameters
        muon_params = []
        adamw_decay = []
        adamw_no_decay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Skip embeddings and lm_head
            is_embed = "embed" in name.lower()
            is_head = "lm_head" in name.lower()

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
                nesterov=True,
                weight_decay=self.weight_decay,
                ns_steps=5,
            )
            optimizers.append(muon_optimizer)
            schedulers.append(self._get_lr_scheduler(muon_optimizer, base_lr=self.muon_lr))

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
            schedulers.append(self._get_lr_scheduler(adamw_optimizer, base_lr=self.learning_rate))

        return cast(OptimizerLRScheduler, (
            optimizers,
            [{"scheduler": s, "interval": "step", "frequency": 1} for s in schedulers],
        ))

    def _configure_adamw(self) -> OptimizerLRScheduler:
        """Standard AdamW optimizer."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.learning_rate, betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_epsilon)

        scheduler = self._get_lr_scheduler(optimizer, base_lr=self.learning_rate)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            },
        }
    
    def _get_lr_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
    ):
        """Cosine schedule with linear warmup."""

        def lr_lambda(current_step: int) -> float:
            if current_step < self.warmup_steps:
                # Linear warmup
                return current_step / max(1, self.warmup_steps)
            
            progress = (current_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            progress = min(1.0, progress)

            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            # scale to [min_lr/base_lr, 1.0]
            min_ratio = self.min_learning_rate / base_lr
            return min_ratio + (1 - min_ratio) * cosine_decay 

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def on_train_epoch_end(self) -> None:
        """Log MoE statistics at end of epoch."""
        moe_stats = {}
        for i, module in enumerate(self.model.modules()):
            if isinstance(module, DeepSeekMoE):
                stats = module.get_load_balance_stats()
                for key, value in stats.items():
                    moe_stats[f"moe/layer_{i}/{key}"] = value

        for key, value in moe_stats.items():
            self.log(key, value, on_epoch=True, sync_dist=True)