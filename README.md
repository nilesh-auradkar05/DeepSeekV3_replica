# DeepSeek-V3 Nano

A from-scratch implementation of DeepSeek-V3 architecture in PyTorch for learning and experimentation.

## Features

- **Multi-Head Latent Attention (MLA)**: 57x KV cache compression
- **DeepSeekMoE**: Mixture of Experts with auxiliary-loss-free load balancing
- **Multi-Token Prediction (MTP)**: Predicts multiple future tokens
- **Decoupled RoPE**: Separates position encoding from content
- **Muon Optimizer**: PyTorch 2.9+ Newton-Schulz orthogonalization
- **MLflow + Databricks**: 3-stage tracing with Unity Catalog support

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Debug run (dummy data, 50 steps)
python train.py --debug

# Train on FineWeb-Edu (1B tokens)
python train.py

# Train with Muon optimizer (PyTorch 2.9+)
python train.py --optimizer muon

# Multi-GPU with FSDP
python train.py --strategy fsdp --num_gpus 4
```

## MLflow + Databricks Setup

### Option 1: Config file

```yaml
# config.yaml
databricks_host: "https://your-workspace.cloud.databricks.com"
databricks_token: "dapi..."
experiment_name: "/Shared/deepseek-v3-nano"
workspace_id: null           # Optional: for multi-workspace
run_id: null                 # Optional: resume run
unity_catalog: databricks-uc
```

### Option 2: Environment variables

```bash
# .env file
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi_your_token
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT_ID=123456789
```

### Resume a run

```bash
python train.py --run_id abc123def456
```

## Project Structure

```
deepseek_v3_nano/
├── config.yaml              # Training configuration
├── train.py                 # Training script
├── .env.example             # Environment template
├── configs/
│   └── model_config.py      # Model presets
├── models/
│   ├── foundations.py       # RMSNorm, RoPE
│   ├── attention.py         # Multi-Head Latent Attention
│   ├── moe.py               # SwiGLU, Router, DeepSeekMoE
│   ├── transformer_block.py # Transformer blocks
│   ├── mtp.py               # Multi-Token Prediction
│   └── model.py             # Full model
├── training/
│   └── lightning_module.py  # Lightning module
└── data/
    └── data_module.py       # FineWeb-Edu streaming
```

## Model Configurations

| Config | Total Params | Activated | Layers | Experts |
|--------|-------------|-----------|-------- |---------|
| nano   | ~150M       | ~40M      | 12      | 16      |
| small  | ~500M       | ~150M     | 16      | 32      |
| medium | ~2B         | ~500M     | 24      | 64      |

## Optimizers

| Optimizer | Learning Rate | Best For                          |
|-----------|-------------- |-----------------------------------|
| AdamW     | 3e-4          | Default, stable                   |
| Muon      | 0.02          | Faster convergence (PyTorch 2.9+) |

Muon uses Newton-Schulz orthogonalization on 2D weights, AdamW for embeddings/biases.

## Distributed Strategies

| Strategy            | Description |
|----------           |-------------|
| `auto`              | Auto-select |
| `ddp`               | DistributedDataParallel |
| `fsdp`              | Fully Sharded Data Parallel |
| `deepspeed_stage_2` | ZeRO-2 |
| `deepspeed_stage_3` | ZeRO-3 with offload |

## Requirements

- Python 3.10+
- PyTorch 2.9+ (for Muon optimizer)
- Lightning 2.5+
- GPU with 8 to 24GB+ VRAM

## References

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2401.02954)
- [Muon Optimizer](https://kellerjordan.github.io/posts/muon/)