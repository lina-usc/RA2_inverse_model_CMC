from __future__ import annotations

from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Dense,
    Dropout,
    Flatten,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Multiply,
    Reshape,
)
from tensorflow.keras.models import Model

try:
    from keras.saving import register_keras_serializable  # type: ignore
except Exception:  # pragma: no cover
    from tensorflow.keras.utils import register_keras_serializable  # type: ignore

from models.posterior_fullcov import raw_tril_size


@register_keras_serializable(package="ra2_cmc")
class TokenMaskPreprocess(tf.keras.layers.Layer):
    """
    Keras-safe preprocessing of token_mask.

    Input:  mask_in (B, tokens) float32 in {0,1}
    Output: mask_bool (B, tokens) bool
            mask_f    (B, tokens, 1) float32
    """

    def call(self, mask_in: tf.Tensor):
        m = tf.cast(mask_in, tf.float32)
        mask_bool = tf.cast(m > 0.5, tf.bool)
        mask_f = m[..., tf.newaxis]
        return mask_bool, mask_f

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()


@register_keras_serializable(package="ra2_cmc")
class ExpandAxis1(tf.keras.layers.Layer):
    """Expand dims at axis=1 -> (B,1,T) attention mask."""

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.expand_dims(x, axis=1)

    def get_config(self) -> Dict[str, Any]:
        return super().get_config()


@register_keras_serializable(package="ra2_cmc")
class CrossMask(tf.keras.layers.Layer):
    """
    Tile a (B, T) boolean mask to (B, Q, T) for cross-attention.
    """

    def __init__(self, n_query: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_query = int(n_query)

    def call(self, mask_bool: tf.Tensor) -> tf.Tensor:
        m = tf.expand_dims(mask_bool, axis=1)  # (B,1,T)
        return tf.tile(m, [1, self.n_query, 1])  # (B,Q,T)

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"n_query": self.n_query})
        return cfg


@register_keras_serializable(package="ra2_cmc")
class HybridPositionalEncoding(tf.keras.layers.Layer):
    """
    Token layout:
      - first n_tokens_erp are ERP time patches
      - remaining are TFR patches ordered time-major then freq-minor

    Adds learnable embeddings:
      - time embedding for all tokens
      - type embedding (ERP vs TFR)
      - freq embedding for TFR tokens only
    """

    def __init__(self, n_time: int, n_freq: int, d_model: int, n_tokens_erp: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_time = int(n_time)
        self.n_freq = int(n_freq)
        self.d_model = int(d_model)
        self.n_tokens_erp = int(n_tokens_erp)

        self.time_emb = self.add_weight(
            shape=(self.n_time, self.d_model),
            initializer="random_normal",
            trainable=True,
            name="time_emb",
        )
        self.freq_emb = self.add_weight(
            shape=(self.n_freq, self.d_model),
            initializer="random_normal",
            trainable=True,
            name="freq_emb",
        )
        self.type_emb = self.add_weight(
            shape=(2, self.d_model),  # 0=ERP, 1=TFR
            initializer="random_normal",
            trainable=True,
            name="type_emb",
        )

        n_tokens_tfr = self.n_time * self.n_freq
        n_tokens = self.n_tokens_erp + n_tokens_tfr

        time_idx = np.zeros((n_tokens,), dtype=np.int32)
        freq_idx = np.zeros((n_tokens,), dtype=np.int32)
        type_idx = np.zeros((n_tokens,), dtype=np.int32)

        # ERP tokens
        k = np.arange(self.n_tokens_erp)
        time_idx[k] = k
        #freq_idx[k] = 0  # No need, already initialized to 0
        #type_idx[k] = 0  # No need, already initialized to 0

        # TFR tokens
        # Vectorized assignment for TFR tokens
        tfr_range = np.arange(n_tokens_tfr)
        time_idx[self.n_tokens_erp:self.n_tokens_erp + n_tokens_tfr] = tfr_range // self.n_freq
        freq_idx[self.n_tokens_erp:self.n_tokens_erp + n_tokens_tfr] = tfr_range % self.n_freq
        type_idx[self.n_tokens_erp:self.n_tokens_erp + n_tokens_tfr] = 1

        self._time_idx = tf.constant(time_idx, dtype=tf.int32)
        self._freq_idx = tf.constant(freq_idx, dtype=tf.int32)
        self._type_idx = tf.constant(type_idx, dtype=tf.int32)


    def call(self, x: tf.Tensor) -> tf.Tensor:
        t = tf.gather(self.time_emb, self._time_idx)     # (tokens, d_model)
        typ = tf.gather(self.type_emb, self._type_idx)   # (tokens, d_model)
        f = tf.gather(self.freq_emb, self._freq_idx)     # (tokens, d_model)
        is_tfr = tf.cast(tf.equal(self._type_idx, 1), tf.float32)[:, tf.newaxis]
        pe = t + typ + is_tfr * f
        return x + pe[tf.newaxis, :, :]

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update(
            {"n_time": self.n_time, "n_freq": self.n_freq, "d_model": self.d_model, "n_tokens_erp": self.n_tokens_erp}
        )
        return cfg


@register_keras_serializable(package="ra2_cmc")
class ParameterTokenLayer(tf.keras.layers.Layer):
    """One learned query token per parameter (broadcast to batch). Output: (B, P, d_model)."""

    def __init__(self, n_params: int, d_model: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_params = int(n_params)
        self.d_model = int(d_model)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.param_tokens = self.add_weight(
            shape=(self.n_params, self.d_model),
            initializer="random_normal",
            trainable=True,
            name="param_tokens",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        b = tf.shape(x)[0]
        return tf.tile(self.param_tokens[tf.newaxis, :, :], [b, 1, 1])

    def get_config(self) -> Dict[str, Any]:
        cfg = super().get_config()
        cfg.update({"n_params": self.n_params, "d_model": self.d_model})
        return cfg


def transformer_encoder_block(
    x: tf.Tensor,
    self_attn_mask: tf.Tensor,  # (B,1,tokens) bool
    d_model: int,
    num_heads: int,
    ff_dim: int,
    dropout_rate: float,
    block_id: int,
) -> tf.Tensor:
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        output_shape=d_model,
        name=f"self_attn_{block_id}",
    )(x, x, attention_mask=self_attn_mask)

    attn = Dropout(dropout_rate, name=f"self_attn_drop_{block_id}")(attn)
    x = Add(name=f"self_attn_add_{block_id}")([x, attn])
    x = LayerNormalization(epsilon=1e-6, name=f"self_attn_norm_{block_id}")(x)

    ff = Dense(ff_dim, activation="gelu", name=f"ff1_{block_id}")(x)
    ff = Dense(d_model, name=f"ff2_{block_id}")(ff)
    ff = Dropout(dropout_rate, name=f"ff_drop_{block_id}")(ff)

    x = Add(name=f"ff_add_{block_id}")([x, ff])
    x = LayerNormalization(epsilon=1e-6, name=f"ff_norm_{block_id}")(x)
    return x


def build_paramtoken_transformer(
    n_tokens: int,
    feature_dim: int,
    n_params: int,
    n_time_patches: int,
    n_freq_patches: int,
    n_tokens_erp: int,
    d_model: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    ff_dim: int = 256,
    dropout_rate: float = 0.10,
    posterior: str = "fullcov",
    return_attention: bool = False,
) -> Model:
    tokens_in = Input(shape=(n_tokens, feature_dim), name="tokens_input")
    mask_in = Input(shape=(n_tokens,), name="token_mask")

    mask_bool, mask_f = TokenMaskPreprocess(name="mask_preprocess")(mask_in)  # bool (B,T), float (B,T,1)
    self_attn_mask = ExpandAxis1(name="self_attn_mask")(mask_bool)            # (B,1,T)

    x = Dense(d_model, name="token_proj")(tokens_in)
    x = HybridPositionalEncoding(
        n_time=n_time_patches,
        n_freq=n_freq_patches,
        d_model=d_model,
        n_tokens_erp=n_tokens_erp,
        name="hybrid_posenc",
    )(x)

    x = Multiply(name="apply_mask_pre")([x, mask_f])

    for li in range(num_layers):
        x = transformer_encoder_block(
            x=x,
            self_attn_mask=self_attn_mask,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            block_id=li,
        )
        x = Multiply(name=f"apply_mask_postblock_{li}")([x, mask_f])

    q = ParameterTokenLayer(n_params, d_model, name="param_tokens")(x)

    mha = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        output_shape=d_model,
        name="param_cross_attn",
    )

    cross_mask = CrossMask(n_query=n_params, name="cross_mask")(mask_bool)  # (B,P,T) bool

    if return_attention:
        ctx, scores = mha(q, x, attention_mask=cross_mask, return_attention_scores=True)
    else:
        ctx = mha(q, x, attention_mask=cross_mask)
        scores = None

    ctx = Dropout(dropout_rate, name="param_ctx_drop")(ctx)
    ctx = LayerNormalization(epsilon=1e-6, name="param_ctx_norm")(ctx)

    h = Dense(ff_dim, activation="gelu", name="param_head_fc")(ctx)
    h = Dropout(dropout_rate, name="param_head_drop")(h)

    mu_z = Dense(1, activation="linear", name="mu_head")(h)        # (B,P,1)
    mu_z = Reshape((n_params,), name="mu_z")(mu_z)                 # (B,P)

    if posterior == "diag":
        logvar_z = Dense(1, activation="linear", name="logvar_head")(h)
        logvar_z = Reshape((n_params,), name="logvar_z")(logvar_z)
        pred = Concatenate(axis=1, name="pred_mu_logvar")([mu_z, logvar_z])

    elif posterior == "fullcov":
        n_tril = raw_tril_size(n_params)
        ctx_flat = Flatten(name="ctx_flat")(ctx)  # (B, P*d_model)
        raw_tril = Dense(n_tril, activation="linear", name="raw_tril_head")(ctx_flat)
        pred = Concatenate(axis=1, name="pred_mu_rawtril")([mu_z, raw_tril])

    else:
        raise ValueError("posterior must be 'diag' or 'fullcov'")

    if return_attention:
        return Model([tokens_in, mask_in], [pred, scores], name="CMC_ParamTokenTransformer")
    return Model([tokens_in, mask_in], pred, name="CMC_ParamTokenTransformer")


def get_custom_objects() -> Dict[str, Any]:
    return {
        "TokenMaskPreprocess": TokenMaskPreprocess,
        "ExpandAxis1": ExpandAxis1,
        "CrossMask": CrossMask,
        "HybridPositionalEncoding": HybridPositionalEncoding,
        "ParameterTokenLayer": ParameterTokenLayer,
        "ra2_cmc>TokenMaskPreprocess": TokenMaskPreprocess,
        "ra2_cmc>ExpandAxis1": ExpandAxis1,
        "ra2_cmc>CrossMask": CrossMask,
        "ra2_cmc>HybridPositionalEncoding": HybridPositionalEncoding,
        "ra2_cmc>ParameterTokenLayer": ParameterTokenLayer,
    }
