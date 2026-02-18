from __future__ import annotations

import argparse
import os
import sys
import pickle
import logging
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_OUT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
os.makedirs(MODELS_OUT, exist_ok=True)

from data.splits import ensure_splits  # noqa: E402
from models.param_transforms import theta_to_z  # noqa: E402
from models.posterior_fullcov import mvn_tril_nll, raw_tril_size  # noqa: E402
from models.transformer_paramtoken import build_paramtoken_transformer  # noqa: E402


def _import_tensorflow_or_die():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except ModuleNotFoundError as e:
        msg = (
            "\nERROR: TensorFlow is not installed in this environment.\n\n"
            f"Python executable:\n  {sys.executable}\n\n"
            "Fix (example):\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install -r requirements.txt\n"
        )
        raise SystemExit(msg) from e


@dataclass
class TrainConfig:
    # optimization
    batch_size: int = 64
    epochs: int = 150
    warmup_epochs: int = 15

    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    clipnorm: float = 1.0
    weight_decay: float = 1e-5

    dropout_rate: float = 0.10
    augment_std: float = 0.01  # after scaling; applied only on valid tokens

    # model
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 256

    # baseline (BiLSTM) hyperparams
    lstm_units: int = 128
    lstm_layers: int = 2
    mlp_units: int = 256

    # splits
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15


def _feature_path(features: str) -> str:
    if features == "hybrid":
        return os.path.join(DATA_OUT, "features.npy")
    if features == "erp":
        return os.path.join(DATA_OUT, "features_erp.npy")
    if features == "tfr":
        return os.path.join(DATA_OUT, "features_tfr.npy")
    raise ValueError("features must be one of: hybrid, erp, tfr")


def _build_token_mask(features: str, n_tokens: int, n_tokens_erp: int) -> np.ndarray:
    m = np.zeros((n_tokens,), dtype=np.float32)
    if features == "hybrid":
        m[:] = 1.0
    elif features == "erp":
        m[:n_tokens_erp] = 1.0
    elif features == "tfr":
        m[n_tokens_erp:] = 1.0
    else:
        raise ValueError("features must be one of: hybrid, erp, tfr")
    return m


def main() -> None:
    tf = _import_tensorflow_or_die()
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
    from tensorflow.keras.optimizers import Adam

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default="cmc")
    ap.add_argument("--arch", choices=["paramtoken", "bilstm"], default="paramtoken")
    ap.add_argument("--features", choices=["hybrid", "erp", "tfr"], default="hybrid")
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None, help="Alias for --train-seed.")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--posterior", choices=["diag", "fullcov"], default="fullcov")
    args = ap.parse_args()

    if args.seed is not None:
        args.train_seed = args.seed


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("train")

    cfg = TrainConfig()

    # Repro
    np.random.seed(args.train_seed)
    tf.random.set_seed(args.train_seed)

    arch_suffix = "" if args.arch == "paramtoken" else f"_{args.arch}"
    run_dir = os.path.join(MODELS_OUT, f"{args.run_name}_{args.features}{arch_suffix}_seed{args.train_seed}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info("Run dir: %s", run_dir)
    logger.info("arch=%s features=%s posterior=%s train_seed=%d split_seed=%d", args.arch, args.features, args.posterior, args.train_seed, args.split_seed)

    # Load data
    X_path = _feature_path(args.features)
    X_mem = np.load(X_path, mmap_mode="r")  # (N,tokens,feat)
    y_theta = np.load(os.path.join(DATA_OUT, "params.npy"), mmap_mode="r")  # (N,P)

    meta = np.load(os.path.join(DATA_OUT, "tfr_meta.npz"))
    param_meta = np.load(os.path.join(DATA_OUT, "param_meta.npz"))

    param_names = [x.decode("utf-8") for x in param_meta["param_names"]]
    low = param_meta["prior_low"].astype(np.float32)
    high = param_meta["prior_high"].astype(np.float32)

    N, n_tokens, feature_dim = X_mem.shape
    P = int(y_theta.shape[1])
    n_tril = raw_tril_size(P)

    n_time_patches = int(meta["n_time_patches"])
    n_freq_patches = int(meta["n_freq_patches"])
    n_tokens_erp = int(meta["n_tokens_erp"])

    logger.info("X=%s | y=%s | params=%s", X_mem.shape, y_theta.shape, param_names)
    logger.info("tokens=%d feat_dim=%d | ERP tokens=%d | TFR=%dx%d", n_tokens, feature_dim, n_tokens_erp, n_time_patches, n_freq_patches)

    # Splits (fixed across feature sets)
    splits = ensure_splits(
        data_out_dir=DATA_OUT,
        n_samples=N,
        seed=args.split_seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        overwrite=False,
    )
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]
    logger.info("Split sizes: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    np.savez(
        os.path.join(run_dir, "split_indices_used.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        split_seed=np.array(args.split_seed),
    )

    # Target transform (theta -> z)
    y_z = theta_to_z(np.asarray(y_theta, dtype=np.float32), low, high)  # (N,P)

    # Token mask (constant pattern per run)
    token_mask_1d = _build_token_mask(args.features, n_tokens=n_tokens, n_tokens_erp=n_tokens_erp)  # (tokens,)
    valid_idx = np.where(token_mask_1d > 0.5)[0]

    # Load train/val into RAM
    logger.info("Loading train/val arrays into RAM...")
    X_train = np.asarray(X_mem[train_idx], dtype=np.float32)
    X_val = np.asarray(X_mem[val_idx], dtype=np.float32)
    y_train = np.asarray(y_z[train_idx], dtype=np.float32)
    y_val = np.asarray(y_z[val_idx], dtype=np.float32)

    # Fit scaler ONLY on valid tokens for this condition
    logger.info("Fitting StandardScaler on VALID tokens only...")
    scaler = StandardScaler()
    scaler.fit(X_train[:, valid_idx, :].reshape(-1, feature_dim))

    mu = scaler.mean_.astype(np.float32)
    sd = scaler.scale_.astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)

    # Apply scaling
    X_train = (X_train - mu[None, None, :]) / sd[None, None, :]
    X_val = (X_val - mu[None, None, :]) / sd[None, None, :]

    # Build mask arrays for dataset (B,tokens)
    m_train = np.tile(token_mask_1d[None, :], (X_train.shape[0], 1)).astype(np.float32)
    m_val = np.tile(token_mask_1d[None, :], (X_val.shape[0], 1)).astype(np.float32)

    # Force invalid tokens to exactly zero after scaling (clean ablation)
    X_train *= m_train[:, :, None]
    X_val *= m_val[:, :, None]

    np.savez(os.path.join(run_dir, "scaler_stats.npz"), mu=mu, sd=sd)
    with open(os.path.join(run_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # tf.data
    def augment(inputs, y):
        x, m = inputs
        if cfg.augment_std <= 0:
            return (x, m), y
        noise = tf.random.normal(tf.shape(x), stddev=cfg.augment_std, dtype=x.dtype)
        x = x + noise * m[:, :, tf.newaxis]
        return (x, m), y

    train_ds = (
        tf.data.Dataset.from_tensor_slices(((X_train, m_train), y_train))
        .shuffle(buffer_size=min(8192, len(X_train)), seed=args.train_seed, reshuffle_each_iteration=True)
        .batch(cfg.batch_size)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(((X_val, m_val), y_val))
        .batch(cfg.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Losses
    def mse_mu_only(y_true, y_pred):
        mu_z = y_pred[:, :P]
        return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - mu_z), axis=1))

    def diag_gaussian_nll_z(y_true, y_pred):
        mu_z = y_pred[:, :P]
        logvar_z = y_pred[:, P:]
        logvar_z = tf.clip_by_value(logvar_z, -10.0, 10.0)
        inv_var = tf.exp(-logvar_z)
        nll = 0.5 * (inv_var * tf.square(y_true - mu_z) + logvar_z)
        return tf.reduce_mean(tf.reduce_sum(nll, axis=1))

    def fullcov_gaussian_nll_z(y_true, y_pred):
        mu_z = y_pred[:, :P]
        raw = y_pred[:, P : P + n_tril]
        raw = tf.clip_by_value(raw, -10.0, 10.0)
        nll = mvn_tril_nll(y_true, mu_z, raw, include_const=False)  # (B,)
        return tf.reduce_mean(nll)

    # Build model
    model = build_paramtoken_transformer(
        n_tokens=n_tokens,
        feature_dim=feature_dim,
        n_params=P,
        n_time_patches=n_time_patches,
        n_freq_patches=n_freq_patches,
        n_tokens_erp=n_tokens_erp,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        ff_dim=cfg.ff_dim,
        dropout_rate=cfg.dropout_rate,
        posterior=args.posterior,
        return_attention=False,
    )
    model.summary()

    # Optimizer (AdamW if available)
    try:
        from tensorflow.keras.optimizers import AdamW
        opt = AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay, clipnorm=cfg.clipnorm)
        logger.info("Using AdamW.")
    except Exception:
        opt = Adam(learning_rate=cfg.learning_rate, clipnorm=cfg.clipnorm)
        logger.info("Using Adam (AdamW unavailable).")

    best_path = os.path.join(run_dir, "paramtoken_best.keras")
    final_path = os.path.join(run_dir, "paramtoken_final.keras")

    # Phase 1: mean warmup
    logger.info("Phase 1: warmup_epochs=%d with MSE(mu) in z-space", cfg.warmup_epochs)
    model.compile(optimizer=opt, loss=mse_mu_only)
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.warmup_epochs,
        callbacks=[CSVLogger(os.path.join(run_dir, "train_log_phase1.csv"))],
        verbose=1,
    )

    # Phase 2: NLL
    logger.info("Phase 2: NLL training for up to %d epochs", cfg.epochs)
    loss_fn = fullcov_gaussian_nll_z if args.posterior == "fullcov" else diag_gaussian_nll_z

    callbacks2 = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=cfg.min_lr, verbose=1),
        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=1),
        CSVLogger(os.path.join(run_dir, "train_log_phase2.csv")),
    ]

    model.compile(optimizer=opt, loss=loss_fn)
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks2,
        verbose=1,
    )

    model.save(final_path)
    logger.info("Saved model: %s", final_path)
    logger.info("Best checkpoint: %s", best_path)

    # Save bounds + config (paper provenance)
    np.savez(
        os.path.join(run_dir, "param_bounds.npz"),
        param_names=np.array(param_names, dtype="S"),
        prior_low=low,
        prior_high=high,
    )
    np.savez(
        os.path.join(run_dir, "model_config.npz"),
        run_name=np.array(args.run_name, dtype="S"),
        arch=np.array(args.arch, dtype="S"),
        features=np.array(args.features, dtype="S"),
        posterior=np.array(args.posterior, dtype="S"),
        n_tokens=int(n_tokens),
        feature_dim=int(feature_dim),
        n_params=int(P),
        n_tril=int(n_tril),
        n_time_patches=int(n_time_patches),
        n_freq_patches=int(n_freq_patches),
        n_tokens_erp=int(n_tokens_erp),
        d_model=int(cfg.d_model),
        num_layers=int(cfg.num_layers),
        num_heads=int(cfg.num_heads),
        ff_dim=int(cfg.ff_dim),
        dropout_rate=float(cfg.dropout_rate),
        lstm_units=int(cfg.lstm_units),
        lstm_layers=int(cfg.lstm_layers),
        mlp_units=int(cfg.mlp_units),
        train_seed=int(args.train_seed),
        split_seed=int(args.split_seed),
        token_mask_1d=token_mask_1d.astype(np.float32),
    )
    np.savez(os.path.join(run_dir, "train_config.npz"), **asdict(cfg))

    logger.info("DONE: %s", run_dir)


if __name__ == "__main__":
    main()
