# Flux Core

A minimal implementation of Flow-based diffusion using the Flux architecture.

## Overview

This project implements a **minimal view of Flow-based diffusion** based on the Flux architecture. The core functionality **diffuses sequence condition tokens to another sequence of condition tokens**, where the two token sequences can be from different data spaces.

## Key Features

- **Flux Architecture**: Built on the proven Flux transformer architecture for flow matching
- **Cross-Domain Diffusion**: Supports diffusion between different data spaces
- **Sequence-to-Sequence**: Processes and transforms sequences of condition tokens
- **Minimal Implementation**: Clean, focused codebase without unnecessary complexity
- **Flow Matching**: Uses flow matching for efficient training and sampling

## Architecture

The implementation consists of several key components:

### Core Model (`model.py`)
- `Flux`: Main transformer model for flow matching on sequences
- `FluxParams`: Configuration dataclass for model parameters
- Supports both single and double stream processing

### Layer Components (`layers.py`)
- `DoubleStreamBlock`: Processes image and text streams simultaneously
- `SingleStreamBlock`: Unified processing for concatenated streams
- `SelfAttention`: Multi-head attention with positional encoding
- `MLPEmbedder`: Multi-layer perceptron for embedding various inputs
- `EmbedND`: N-dimensional positional embeddings

### Mathematical Operations (`flux_math.py`)
- `attention`: Scaled dot-product attention with RoPE
- `rope`: Rotary Position Embedding implementation
- `apply_rope`: Applies rotary embeddings to query and key tensors

## Usage

```python
from model import Flux, FluxParams
import torch

# Configure model parameters
params = FluxParams(
    in_channels=2,
    out_channels=2,
    vec_in_dim=4,
    context_in_dim=4,
    hidden_size=1024,
    mlp_ratio=4.0,
    num_heads=8,
    depth=2,
    depth_single_blocks=4,
    axes_dim=[16, 56, 56],  # sum must equal hidden_size // num_heads
    theta=10_000,
    qkv_bias=True,
    guidance_embed=False,
)

# Create model
model = Flux(params)

# Prepare inputs
data = torch.randn(2, 1024, 2)      # Input sequence
cond = torch.randn(2, 1024, 4)      # Condition sequence
data_ids = torch.randint(0, 1024, (2, 1024, 3))  # Position IDs
cond_ids = torch.randint(0, 1024, (2, 1024, 3))  # Condition position IDs
timesteps = torch.randn(2)          # Diffusion timesteps
y = torch.randn(2, 4)              # Additional conditioning

# Forward pass
output = model(data, data_ids, cond, cond_ids, timesteps, y)
```

## Requirements

- PyTorch
- einops
- torch (with CUDA support recommended)

## Installation

```bash
git clone <repository-url>
cd flux_core
pip install torch einops
```

## Model Configuration

The model is highly configurable through `FluxParams`:

- `in_channels` / `out_channels`: Input/output channel dimensions
- `vec_in_dim` / `context_in_dim`: Conditioning vector dimensions
- `hidden_size`: Model hidden dimension
- `num_heads`: Number of attention heads
- `depth`: Number of double-stream blocks
- `depth_single_blocks`: Number of single-stream blocks
- `axes_dim`: Positional embedding dimensions (must sum to `hidden_size // num_heads`)

## Flow-Based Diffusion

This implementation uses flow matching, a continuous-time approach to generative modeling that:

1. Learns to transform noise to data through continuous flows
2. Enables efficient sampling without requiring many denoising steps
3. Provides stable training dynamics
4. Supports conditioning on various input modalities

The Flux architecture specifically excels at handling sequences and cross-modal conditioning, making it ideal for tasks requiring transformation between different data spaces. 