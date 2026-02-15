from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def _raw_to_tril(raw_tril: tf.Tensor, n_params: int, min_diag: float = 1e-3) -> tf.Tensor:
    """
    Convert raw_tril vector (B, n_tril) to lower-triangular matrix L (B, P, P).
    Diagonal is forced positive via softplus + min_diag.

    n_tril = P*(P+1)/2
    """
    raw_tril = tf.convert_to_tensor(raw_tril)
    B = tf.shape(raw_tril)[0]
    P = int(n_params)

    # Build L using small Python loops (P<=10, cheap and stable)
    rows = []
    idx = 0
    for i in range(P):
        cols = []
        for j in range(P):
            if j <= i:
                cols.append(raw_tril[:, idx])
                idx += 1
            else:
                cols.append(tf.zeros((B,), dtype=raw_tril.dtype))
        rows.append(tf.stack(cols, axis=1))  # (B,P)
    L = tf.stack(rows, axis=1)  # (B,P,P)

    # Positive diagonal
    diag = tf.linalg.diag_part(L)
    diag = tf.nn.softplus(diag) + tf.cast(min_diag, raw_tril.dtype)
    L = tf.linalg.set_diag(L, diag)
    return L


def mvn_tril_nll(
    y: tf.Tensor,
    mu: tf.Tensor,
    raw_tril: tf.Tensor,
    include_const: bool = False,
    min_diag: float = 1e-3,
) -> tf.Tensor:
    """
    Negative log-likelihood for multivariate normal with covariance Σ = L L^T,
    where L is lower triangular derived from raw_tril.

    Returns
    -------
    nll : (B,) tensor
    """
    y = tf.convert_to_tensor(y)
    mu = tf.convert_to_tensor(mu)
    raw_tril = tf.convert_to_tensor(raw_tril)

    P = int(mu.shape[-1])
    L = _raw_to_tril(raw_tril, n_params=P, min_diag=min_diag)  # (B,P,P)

    diff = (y - mu)[..., tf.newaxis]  # (B,P,1)
    solve = tf.linalg.triangular_solve(L, diff, lower=True)  # (B,P,1)
    maha = tf.reduce_sum(tf.square(solve), axis=[1, 2])  # (B,)

    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)  # (B,)

    nll = 0.5 * (maha + logdet)
    if include_const:
        nll = nll + 0.5 * tf.cast(P, nll.dtype) * tf.math.log(tf.constant(2.0 * 3.141592653589793, dtype=nll.dtype))
    return nll


def raw_tril_size(n_params: int) -> int:
    P = int(n_params)
    return P * (P + 1) // 2
