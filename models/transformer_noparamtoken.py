from __future__ import annotations
import numpy as np

def build_noparamtoken_transformer(
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
):
    """
    Transformer encoder WITHOUT per-parameter tokens.
    Readout = masked mean pooling over tokens -> posterior head.

    Inputs:
      x: (B, n_tokens, feature_dim)
      m: (B, n_tokens) in {0,1}

    Output:
      fullcov: (B, n_params + n_tril)
      diag   : (B, 2*n_params)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from models.posterior_fullcov import raw_tril_size

    if posterior not in ("fullcov", "diag"):
        raise ValueError("posterior must be 'fullcov' or 'diag'")

    T = int(n_tokens)
    D = int(d_model)
    n_tokens_erp = int(n_tokens_erp)
    n_time_patches = int(n_time_patches)
    n_freq_patches = int(n_freq_patches)

    x_in = layers.Input(shape=(T, int(feature_dim)), name="x")
    m_in = layers.Input(shape=(T,), dtype="float32", name="m")

    # Project features
    x = layers.Dense(D, name="proj")(x_in)

    # Build structured token indices: type/time/freq (constants, then tiled to batch)
    type_idx = np.zeros((T,), dtype=np.int32)
    type_idx[n_tokens_erp:] = 1

    time_idx = np.zeros((T,), dtype=np.int32)
    freq_idx = np.zeros((T,), dtype=np.int32)

    # ERP tokens: time index 0..n_tokens_erp-1
    time_idx[:n_tokens_erp] = np.arange(n_tokens_erp, dtype=np.int32)

    # TFR tokens: flattened row-major after ERP tokens
    n_tfr = max(0, T - n_tokens_erp)
    if n_tfr > 0:
        k = np.arange(n_tfr, dtype=np.int32)
        time_idx[n_tokens_erp:] = (k // n_freq_patches).astype(np.int32)
        freq_idx[n_tokens_erp:] = (k % n_freq_patches).astype(np.int32)

    type_const = tf.constant(type_idx[None, :], dtype=tf.int32)  # (1,T)
    time_const = tf.constant(time_idx[None, :], dtype=tf.int32)  # (1,T)
    freq_const = tf.constant(freq_idx[None, :], dtype=tf.int32)  # (1,T)

    def tile_const(const, name):
        # Output shape excludes batch dim => (T,)
        return layers.Lambda(
            lambda xb: tf.tile(const, [tf.shape(xb)[0], 1]),
            output_shape=(T,),
            dtype=tf.int32,
            name=name,
        )(x_in)

    type_ids = tile_const(type_const, "type_ids")
    time_ids = tile_const(time_const, "time_ids")
    freq_ids = tile_const(freq_const, "freq_ids")

    type_emb = layers.Embedding(2, D, name="type_emb")(type_ids)
    time_emb = layers.Embedding(n_time_patches, D, name="time_emb")(time_ids)
    freq_emb = layers.Embedding(n_freq_patches, D, name="freq_emb")(freq_ids)

    x = layers.Add(name="add_pos")([x, type_emb, time_emb, freq_emb])
    x = layers.Dropout(float(dropout_rate), name="in_drop")(x)

    # Boolean token mask (B,T)
    m_bool = layers.Lambda(
        lambda m: tf.cast(m, tf.bool),
        output_shape=(T,),
        dtype=tf.bool,
        name="m_bool",
    )(m_in)

    # Attention mask (B,T,T)
    attn_mask = layers.Lambda(
        lambda mb: tf.logical_and(mb[:, :, None], mb[:, None, :]),
        output_shape=(T, T),
        dtype=tf.bool,
        name="attn_mask",
    )(m_bool)

    # Encoder blocks
    for i in range(int(num_layers)):
        attn = layers.MultiHeadAttention(
            num_heads=int(num_heads),
            key_dim=int(D // int(num_heads)),
            dropout=float(dropout_rate),
            name=f"mha_{i}",
        )(x, x, attention_mask=attn_mask)
        attn = layers.Dropout(float(dropout_rate), name=f"attn_drop_{i}")(attn)
        x = layers.LayerNormalization(epsilon=1e-5, name=f"ln1_{i}")(layers.Add()([x, attn]))

        ff = layers.Dense(int(ff_dim), activation=tf.nn.gelu, name=f"ff1_{i}")(x)
        ff = layers.Dropout(float(dropout_rate), name=f"ff_drop1_{i}")(ff)
        ff = layers.Dense(D, name=f"ff2_{i}")(ff)
        ff = layers.Dropout(float(dropout_rate), name=f"ff_drop2_{i}")(ff)
        x = layers.LayerNormalization(epsilon=1e-5, name=f"ln2_{i}")(layers.Add()([x, ff]))

    # Masked mean pooling over tokens => (B,D)
    x_pool = layers.Lambda(
        lambda args: tf.reduce_sum(args[0] * tf.cast(args[1], tf.float32)[:, :, None], axis=1)
        / (tf.reduce_sum(tf.cast(args[1], tf.float32), axis=1, keepdims=True) + 1e-6),
        output_shape=(D,),
        dtype=tf.float32,
        name="masked_mean_pool",
    )([x, m_bool])

    # Posterior head
    if posterior == "fullcov":
        n_tril = int(raw_tril_size(int(n_params)))
        out_dim = int(n_params) + n_tril
    else:
        out_dim = int(2 * n_params)

    y = layers.Dense(int(out_dim), name="posterior_head")(x_pool)

    model = Model(inputs=[x_in, m_in], outputs=y, name="noparamtoken_transformer")
    return model
