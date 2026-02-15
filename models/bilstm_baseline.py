from __future__ import annotations

import keras
from keras import layers, ops

from models.posterior_fullcov import raw_tril_size


def build_bilstm_baseline(
    *,
    n_tokens: int,
    feature_dim: int,
    n_params: int,
    posterior: str = "fullcov",
    lstm_units: int = 128,
    lstm_layers: int = 2,
    mlp_units: int = 256,
    dropout_rate: float = 0.10,
) -> keras.Model:
    """BiLSTM baseline inverse model.

    Inputs:
      x:    (B, T, C)
      mask: (B, T) float in {0,1}

    Output:
      fullcov: (B, n_params + raw_tril_size(n_params))
      diag:    (B, 2*n_params) [mu + logvar]
    """

    x_in = keras.Input(shape=(n_tokens, feature_dim), name="x")
    mask_in = keras.Input(shape=(n_tokens,), name="mask")

    # Keras-safe boolean mask (Keras 3)
    mask_bool = ops.greater(mask_in, 0.5)  # (B,T) bool

    # Zero invalid tokens explicitly
    x = x_in * ops.expand_dims(mask_in, axis=-1)  # (B,T,C)

    h = x
    for i in range(int(lstm_layers)):
        h = layers.Bidirectional(
            layers.LSTM(
                int(lstm_units),
                return_sequences=True,
                dropout=float(dropout_rate),
            ),
            name=f"bilstm_{i}",
        )(h, mask=mask_bool)

    # Masked mean pooling over tokens
    mask_f = ops.cast(mask_bool, h.dtype)              # (B,T)
    mask_f = ops.expand_dims(mask_f, axis=-1)          # (B,T,1)
    sum_h = ops.sum(h * mask_f, axis=1)                # (B,H)
    denom = ops.sum(mask_f, axis=1) + 1e-6             # (B,1)
    pooled = sum_h / denom                             # (B,H)

    z = layers.LayerNormalization(name="ln")(pooled)
    z = layers.Dense(int(mlp_units), activation="gelu", name="mlp")(z)
    z = layers.Dropout(float(dropout_rate), name="mlp_dropout")(z)

    if posterior == "fullcov":
        out_dim = int(n_params) + raw_tril_size(int(n_params))
    elif posterior == "diag":
        out_dim = int(n_params) * 2
    else:
        raise ValueError(f"Unsupported posterior: {posterior}")

    out = layers.Dense(out_dim, name="posterior_out")(z)
    return keras.Model([x_in, mask_in], out, name="bilstm_baseline")
