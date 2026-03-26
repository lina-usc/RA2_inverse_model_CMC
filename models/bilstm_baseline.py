"""
BiLSTM baseline used as a non-attention comparison model.

Important implementation detail (Keras/TF mask + cuDNN):
--------------------------------------------------------
TensorFlow's fused/cuDNN LSTM kernels only support *right-padded* masks.
Our token masks for the TFR-only setting are *left-padded* (ERP tokens are
masked out at the beginning, TFR tokens kept at the end), which triggers:

  InvalidArgumentError: mask does not correspond to right-padded sequences

To keep the baseline working on all hardware backends, we DO NOT pass the mask
into the LSTM layers. Instead we:
  1) Multiply inputs by the mask (so invalid tokens are zeros), and
  2) Use the mask only for the final masked mean pooling.

This preserves the intended ablation (ERP/TFR/Hybrid) without relying on
backend-specific masking constraints.

NOTE (serialization):
---------------------
We register custom layers so tf.keras/keras can load saved .keras models during
evaluation without retraining.
"""

from __future__ import annotations

from keras import Model
from keras import layers, ops
from keras.saving import register_keras_serializable


@register_keras_serializable(package="cmc")
class ExpandDims(layers.Layer):
    """Expand a 2D mask (B,T) to (B,T,1) for broadcast multiplication."""

    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.axis = int(axis)

    def call(self, x):
        return ops.expand_dims(x, axis=self.axis)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg


@register_keras_serializable(package="cmc")
class Sum(layers.Layer):
    """Serializable sum over an axis.

    Backward-compat: older saved models may not include axis/keepdims in config,
    so from_config() fills them in (we infer keepdims from layer name).
    """

    def __init__(self, axis: int = 1, keepdims: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.axis = int(axis)
        self.keepdims = bool(keepdims)

    def call(self, x):
        return ops.sum(x, axis=self.axis, keepdims=self.keepdims)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis, "keepdims": self.keepdims})
        return cfg

    @classmethod
    def from_config(cls, config):
        # Older checkpoints may not have stored these fields.
        if "axis" not in config:
            config["axis"] = 1
        if "keepdims" not in config:
            name = str(config.get("name", ""))
            # In this repo we use: sum_mask (keepdims=True) and sum_h (keepdims=False)
            config["keepdims"] = ("mask" in name)
        return cls(**config)


def get_custom_objects():
    """Expose custom objects for tf.keras.models.load_model(custom_objects=...)."""
    return {"ExpandDims": ExpandDims, "Sum": Sum}


def build_bilstm_baseline(
    n_tokens: int,
    feat_dim: int,
    out_dim: int,
    hidden_size: int = 128,
    depth: int = 2,
    dropout: float = 0.10,
    mlp_units: int = 256,
    name: str = "bilstm_baseline",
) -> Model:
    """Build BiLSTM baseline.

    Inputs:
      x:         (B, T, C)
      token_mask:(B, T)   float32 0/1 token mask

    Output:
      (B, out_dim) posterior head parameters
    """

    x_in = layers.Input(shape=(n_tokens, feat_dim), name="x")
    mask_in = layers.Input(shape=(n_tokens,), name="token_mask")

    # Zero-out masked tokens.
    mask_f = ExpandDims(name="expand_dims")(mask_in)  # (B,T,1)
    x = layers.Multiply(name="multiply")([x_in, mask_f])

    h = x
    for i in range(depth):
        h = layers.Bidirectional(
            layers.LSTM(
                hidden_size,
                return_sequences=True,
                dropout=dropout,
                # NOTE: no mask=... passed (see module docstring).
            ),
            name=f"bilstm_{i}",
        )(h)

    # Masked mean pooling over time:
    #   pooled = sum_t (h_t * m_t) / (sum_t m_t)
    h_masked = layers.Multiply(name="multiply_pool")([h, mask_f])

    denom = Sum(axis=1, keepdims=True, name="sum_mask")(mask_in)   # (B,1)
    numer = Sum(axis=1, keepdims=False, name="sum_h")(h_masked)    # (B,H)

    pooled = layers.Lambda(lambda z: z[0] / (z[1] + 1e-6), name="masked_mean")([numer, denom])

    h2 = layers.LayerNormalization(name="ln")(pooled)
    h2 = layers.Dense(mlp_units, activation="relu", name="mlp")(h2)
    h2 = layers.Dropout(dropout, name="mlp_dropout")(h2)

    out = layers.Dense(out_dim, name="posterior_out")(h2)

    return Model(inputs=[x_in, mask_in], outputs=out, name=name)
