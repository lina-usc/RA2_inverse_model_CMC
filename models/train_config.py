"""
models.train_config

Centralized default training hyperparameters.

This file is intentionally lightweight: it only defines TrainConfig used by
models/train.py. Keeping it here avoids hardcoding hyperparameters in scripts
and prevents import errors when running `python -m models.train`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Optimization / training loop
    epochs: int = 120
    warmup_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 7.5e-5
    patience: int = 12

    # Data augmentation (token space)
    augment_std: float = 0.0

    # Transformer defaults (used by transformer builders)
    d_model: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.10