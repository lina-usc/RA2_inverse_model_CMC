"""Evaluate an ensemble of trained inverse models and write paper-ready outputs.

Intended to be run from the repo root as:

  python -m eval.evaluate_ensemble \
    --data-out data_out_morlet \
    --model-dirs models_out_morlet/run1 models_out_morlet/run2 models_out_morlet/run3 \
    --features hybrid \
    --arch paramtoken \
    --split test \
    --n-eval 1500 \
    --n-post-samples 200 \
    --seed 0 \
    --out-dir plots_morlet/eval_hybrid_ens

Key design goals:
  * Works for paramtoken / noparamtoken Transformers and the BiLSTM baseline.
  * Works for fullcov and diag Gaussian heads (infers which from output dim).
  * Writes outputs compatible with the provided plotting scripts.

Outputs to --out-dir:
  - eval_<split>_outputs.npz
      theta_true               (N,P)
      theta_post_mean          (N,P)
      theta_post_samples       (N,S,P)
      plus convenience aliases: theta_mean/theta_samples
      plus z_true, nll_z, eval_idx, param_names, etc.

  - metrics_<split>.csv
      param, pearson_mean, rmse_mean, rmse_norm_mean

  - nllz_hist_<split>.png

Notes:
  * We compute metrics in *theta space* using the posterior mean estimated by
    Monte Carlo samples from the ensemble mixture.
  * We compute NLL in *z space* using the exact per-member Gaussian heads and
    an equal-weight mixture.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Non-interactive backend before importing pyplot.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from scipy.stats import pearsonr

# -----------------------------------------------------------------------------
# Repo-relative imports
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import tensorflow as tf

from models.param_transforms import theta_to_z, z_to_theta
from models.posterior_fullcov import mvn_tril_nll, raw_tril_size


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _decode_str_array(arr: np.ndarray) -> List[str]:
    out: List[str] = []
    for x in arr.tolist():
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _pick(npz: "np.lib.npyio.NpzFile", keys: Sequence[str]) -> np.ndarray:
    for k in keys:
        if k in npz:
            return npz[k]
    raise KeyError(f"None of keys found in npz: {keys} ; available={list(npz.keys())}")


def _feature_path(data_out: str, features: str) -> str:
    if features == "hybrid":
        return os.path.join(data_out, "features.npy")
    if features == "erp":
        return os.path.join(data_out, "features_erp.npy")
    if features == "tfr":
        return os.path.join(data_out, "features_tfr.npy")
    raise ValueError(f"Unknown features='{features}'")


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    res = pearsonr(a, b)
    r = getattr(res, "statistic", None)
    if r is None:
        r = res[0]
    if np.isnan(r):
        return 0.0
    return float(r)


def _assert_non_degenerate_theta(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    std = arr.std(axis=0)
    mn = float(arr.min())
    mx = float(arr.max())
    if abs(mx - mn) < 1e-12 or np.all(std < 1e-12):
        raise ValueError(
            f"Degenerate {name}: all selected targets have zero variance. "
            f"min={mn} max={mx} std={std}. "
            f"Check data_out/params.npy and split indices before evaluating."
        )

def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _softplus_inv(y: np.ndarray) -> np.ndarray:
    """Inverse of softplus for y>0, stable for small/large y."""
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, 1e-12, None)
    # For large y, softplus(x) ~= x, so inv(y) ~= y.
    return np.where(y > 20.0, y, np.log(np.expm1(y))).astype(np.float32)


def _logmeanexp(logp: np.ndarray, axis: int = 0) -> np.ndarray:
    """Stable log(mean(exp(logp))) along axis."""
    m = np.max(logp, axis=axis, keepdims=True)
    return (m + np.log(np.mean(np.exp(logp - m), axis=axis, keepdims=True))).squeeze(axis)


@dataclass
class PackedTril:
    P: int
    diag_eps: float = 1e-3

    def __post_init__(self) -> None:
        tri = np.tril_indices(self.P)
        self._tri = tri

        diag_idx = []
        for d in range(self.P):
            pos = np.where((tri[0] == d) & (tri[1] == d))[0]
            if pos.size != 1:
                raise RuntimeError("Could not locate diagonal indices in tril packing")
            diag_idx.append(int(pos[0]))
        self.diag_idx = np.array(diag_idx, dtype=np.int32)

    def raw_to_L(self, raw: np.ndarray) -> np.ndarray:
        """raw (..., n_tril) -> L (..., P, P) using softplus(diag)+eps."""
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape[-1] != raw_tril_size(self.P):
            raise ValueError(f"raw last dim {raw.shape[-1]} != n_tril {raw_tril_size(self.P)}")

        flat = raw.reshape((-1, raw.shape[-1]))
        L = np.zeros((flat.shape[0], self.P, self.P), dtype=np.float32)
        L[:, self._tri[0], self._tri[1]] = flat

        # Softplus on diag (TF softplus for parity).
        diag_vals = tf.nn.softplus(tf.convert_to_tensor(L[:, np.arange(self.P), np.arange(self.P)])).numpy()
        L[:, np.arange(self.P), np.arange(self.P)] = diag_vals + self.diag_eps

        return L.reshape(raw.shape[:-1] + (self.P, self.P))

    def diag_std_to_raw(self, std: np.ndarray) -> np.ndarray:
        """Convert desired diagonal std (sigma) to raw packed vector (diag only)."""
        std = np.asarray(std, dtype=np.float32)
        n_tril = raw_tril_size(self.P)
        raw = np.zeros(std.shape[:-1] + (n_tril,), dtype=np.float32)

        # We want: softplus(raw_diag) + eps = std  => raw_diag = softplus^{-1}(std - eps)
        target = np.clip(std - self.diag_eps, 1e-12, None)
        raw_diag = _softplus_inv(target)

        for i in range(self.P):
            raw[..., self.diag_idx[i]] = raw_diag[..., i]
        return raw


def _get_custom_objects() -> Dict[str, object]:
    custom: Dict[str, object] = {}
    try:
        from models.transformer_paramtoken import get_custom_objects as _go

        custom.update(_go())
    except Exception:
        pass

    try:
        from models.transformer_noparamtoken import get_custom_objects as _go2

        custom.update(_go2())
    except Exception:
        pass

    try:
        from models.bilstm_baseline import get_custom_objects as _go3

        custom.update(_go3())
    except Exception:
        pass

    return custom



def _repair_lambda_tf(model) -> None:
    """Fix Keras Lambda layers where `tf` was deserialized as a dict.

    Some saved Lambda layers end up with function globals like {'tf': tf.__dict__}
    which breaks calls like tf.tile(...) at inference time.
    """
    import tensorflow as _tf

    try:
        from keras.layers import Lambda as _Lambda
    except Exception:
        try:
            from tensorflow.keras.layers import Lambda as _Lambda  # type: ignore
        except Exception:
            _Lambda = None

    if _Lambda is None:
        return

    seen = set()

    def _walk(obj):
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        # Patch Lambda layers
        if isinstance(obj, _Lambda):
            fn = getattr(obj, "function", None) or getattr(obj, "_function", None)
            if fn is not None and hasattr(fn, "__globals__"):
                g = fn.__globals__
                if isinstance(g.get("tf"), dict):
                    g["tf"] = _tf

        # Recurse into nested models/layers
        for sub in getattr(obj, "layers", []) or []:
            _walk(sub)

    _walk(model)

def _find_checkpoint(model_dir: str, arch: Optional[str], use_final: bool) -> str:
    suffix = "final" if use_final else "best"

    # Best effort: arch-specific name.
    if arch:
        p = os.path.join(model_dir, f"{arch}_{suffix}.keras")
        if os.path.isfile(p):
            return p

    # Single match.
    matches = sorted(glob.glob(os.path.join(model_dir, f"*_{suffix}.keras")))
    if len(matches) == 1:
        return matches[0]

    # Priority list.
    for name in [
        f"paramtoken_{suffix}.keras",
        f"noparamtoken_{suffix}.keras",
        f"bilstm_{suffix}.keras",
    ]:
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p

    # Any .keras.
    any_keras = sorted(glob.glob(os.path.join(model_dir, "*.keras")))
    if any_keras:
        return any_keras[0]

    raise FileNotFoundError(f"No .keras checkpoints found in: {model_dir}")


def _load_scaler_stats(model_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load (mu, sd) used for StandardScaler in training."""

    npz_path = os.path.join(model_dir, "scaler_stats.npz")
    if os.path.isfile(npz_path):
        s = np.load(npz_path)
        if "mu" in s and "sd" in s:
            return s["mu"].astype(np.float32), s["sd"].astype(np.float32)
        if "mean" in s and "std" in s:
            return s["mean"].astype(np.float32), s["std"].astype(np.float32)

    pkl_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.isfile(pkl_path):
        import joblib

        scaler = joblib.load(pkl_path)
        mu = scaler.mean_.astype(np.float32)
        sd = np.sqrt(scaler.var_).astype(np.float32)
        return mu, sd

    raise FileNotFoundError(f"Missing scaler_stats.npz (or scaler.pkl) in {model_dir}")


def _load_split_indices(model_dir: str, data_out: str, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefer split indices saved by training; fall back to data_out/splits.npz."""

    cand = os.path.join(model_dir, "split_indices_used.npz")
    if os.path.isfile(cand):
        z = np.load(cand)
        tr = _pick(z, ["train_idx", "idx_train"])
        va = _pick(z, ["val_idx", "idx_val"])
        te = _pick(z, ["test_idx", "idx_test"])
        return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)

    splits_path = os.path.join(data_out, "splits.npz")
    if os.path.isfile(splits_path):
        z = np.load(splits_path)
        tr = _pick(z, ["train_idx", "idx_train"])
        va = _pick(z, ["val_idx", "idx_val"])
        te = _pick(z, ["test_idx", "idx_test"])
        return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64)

    # Last-resort: call ensure_splits if it exists.
    try:
        from data.splits import ensure_splits

        out = ensure_splits(data_out_dir=data_out, n_samples=N, seed=42)
        if isinstance(out, dict):
            return out["train_idx"], out["val_idx"], out["test_idx"]
        return out  # assume tuple
    except Exception as e:
        raise FileNotFoundError(
            "Could not locate split indices (split_indices_used.npz or data_out/splits.npz) "
            f"and ensure_splits fallback failed: {e}"
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def _extract_rank2_const_from_lambda_fn(fn):
    import numpy as np
    import tensorflow as tf

    candidates = []

    for cell in getattr(fn, "__closure__", ()) or ():
        try:
            candidates.append(cell.cell_contents)
        except Exception:
            pass

    for x in getattr(fn, "__defaults__", ()) or ():
        candidates.append(x)

    kwd = getattr(fn, "__kwdefaults__", None) or {}
    candidates.extend(kwd.values())

    for obj in candidates:
        tens = None

        if tf.is_tensor(obj):
            tens = obj
        elif isinstance(obj, np.ndarray):
            tens = tf.convert_to_tensor(obj)
        elif isinstance(obj, (list, tuple)):
            arr = np.asarray(obj)
            if arr.ndim in (1, 2):
                tens = tf.convert_to_tensor(arr)

        if tens is None:
            continue

        if tens.shape.rank == 1:
            tens = tens[None, :]

        if tens.shape.rank == 2 and int(tens.shape[0]) == 1:
            return tens

    raise RuntimeError("Could not extract rank-2 constant from Lambda function")


def _looks_like_bool_cast_lambda(layer):
    fn = getattr(layer, "function", None)
    code = getattr(fn, "__code__", None)
    names = set(getattr(code, "co_names", ()) or ())
    dtype_str = str(getattr(layer, "compute_dtype", getattr(layer, "dtype", "")))
    return ("cast" in names and "bool" in names) or (dtype_str.endswith("bool") and "cast" in names)


def _repair_lambda_tf_v3(model):
    import tensorflow as tf

    def _iter_layers(obj):
        try:
            for sub in obj._flatten_layers(include_self=True, recursive=True):
                yield sub
            return
        except Exception:
            pass

        seen = set()
        stack = [obj]
        while stack:
            cur = stack.pop()
            oid = id(cur)
            if oid in seen:
                continue
            seen.add(oid)
            yield cur
            for sub in getattr(cur, "layers", []) or []:
                stack.append(sub)

    def _patch_fn_globals(fn):
        try:
            g = getattr(fn, "__globals__", None)
            if not isinstance(g, dict):
                return
            cur = g.get("tf", None)
            if cur is None or isinstance(cur, dict):
                g["tf"] = tf
            cur2 = g.get("tensorflow", None)
            if isinstance(cur2, dict):
                g["tensorflow"] = tf
        except Exception:
            pass

    for layer in _iter_layers(model):
        fn = getattr(layer, "function", None)
        if not callable(fn):
            continue

        _patch_fn_globals(fn)

        if getattr(layer, "name", None) in {"type_ids", "time_ids", "freq_ids"}:
            try:
                const = tf.cast(_extract_rank2_const_from_lambda_fn(fn), tf.int32)

                def _tile_const(xb, const=const):
                    import tensorflow as tf
                    return tf.tile(const, [tf.shape(xb)[0], 1])

                layer.function = _tile_const
                layer._fn_expects_mask_arg = False
                layer._fn_expects_training_arg = False
            except Exception:
                pass

        elif _looks_like_bool_cast_lambda(layer):
            def _bool_cast(m):
                import tensorflow as tf
                return tf.cast(m, tf.bool)

            layer.function = _bool_cast
            layer._fn_expects_mask_arg = False
            layer._fn_expects_training_arg = False

    return model


def _install_lambda_call_repair_v3():
    import tensorflow as tf
    try:
        import keras
    except Exception:
        keras = None

    def _patch_class(cls):
        if cls is None:
            return
        if getattr(cls.call, "_ra2_lambda_call_patched_v3", False):
            return

        orig_call = cls.call

        def _wrapped_call(self, inputs, mask=None, training=None):
            _repair_lambda_tf_v3(self)

            if getattr(self, "name", None) in {"type_ids", "time_ids", "freq_ids"}:
                const = tf.cast(_extract_rank2_const_from_lambda_fn(getattr(self, "function", None)), tf.int32)
                return tf.tile(const, [tf.shape(inputs)[0], 1])

            if _looks_like_bool_cast_lambda(self):
                return tf.cast(inputs, tf.bool)

            try:
                return orig_call(self, inputs, mask=mask, training=training)
            except AttributeError as e:
                if "'dict' object has no attribute" not in str(e):
                    raise

                _repair_lambda_tf_v3(self)

                if getattr(self, "name", None) in {"type_ids", "time_ids", "freq_ids"}:
                    const = tf.cast(_extract_rank2_const_from_lambda_fn(getattr(self, "function", None)), tf.int32)
                    return tf.tile(const, [tf.shape(inputs)[0], 1])

                if _looks_like_bool_cast_lambda(self):
                    return tf.cast(inputs, tf.bool)

                raise

        _wrapped_call._ra2_lambda_call_patched_v3 = True
        cls.call = _wrapped_call

    _patch_class(getattr(tf.keras.layers, "Lambda", None))
    if keras is not None:
        _patch_class(getattr(keras.layers, "Lambda", None))

def _rebuild_noparamtoken_model(
    loaded_model,
    *,
    n_tokens: int,
    feat_dim: int,
    n_params: int,
    n_tokens_erp: int,
):
    from models.posterior_fullcov import raw_tril_size
    from models.transformer_noparamtoken import build_noparamtoken_transformer

    def _get_int(layer, attr_name: str, config_key: str | None = None) -> int:
        v = getattr(layer, attr_name, None)
        if v is None:
            cfg = layer.get_config()
            key = config_key or attr_name
            v = cfg.get(key, None)
        if v is None:
            raise ValueError(f"Could not infer {attr_name} from layer {layer.name}")
        return int(v)

    proj = loaded_model.get_layer("proj")
    time_emb = loaded_model.get_layer("time_emb")
    freq_emb = loaded_model.get_layer("freq_emb")
    in_drop = loaded_model.get_layer("in_drop")
    ff1_0 = loaded_model.get_layer("ff1_0")
    mha_0 = loaded_model.get_layer("mha_0")

    d_model = _get_int(proj, "units")
    n_time_patches = _get_int(time_emb, "input_dim")
    n_freq_patches = _get_int(freq_emb, "input_dim")
    ff_dim = _get_int(ff1_0, "units")

    num_heads = getattr(mha_0, "num_heads", None)
    if num_heads is None:
        num_heads = getattr(mha_0, "_num_heads", None)
    if num_heads is None:
        num_heads = mha_0.get_config().get("num_heads", None)
    if num_heads is None:
        raise ValueError("Could not infer num_heads from mha_0")
    num_heads = int(num_heads)

    dropout_rate = getattr(in_drop, "rate", None)
    if dropout_rate is None:
        dropout_rate = in_drop.get_config().get("rate", None)
    if dropout_rate is None:
        dropout_rate = getattr(mha_0, "dropout", None)
    if dropout_rate is None:
        dropout_rate = getattr(mha_0, "_dropout", None)
    if dropout_rate is None:
        dropout_rate = mha_0.get_config().get("dropout", None)
    if dropout_rate is None:
        raise ValueError("Could not infer dropout_rate")
    dropout_rate = float(dropout_rate)

    mha_ids = []
    for layer in loaded_model.layers:
        name = getattr(layer, "name", "")
        if name.startswith("mha_"):
            try:
                mha_ids.append(int(name.split("_", 1)[1]))
            except Exception:
                pass
    if not mha_ids:
        raise ValueError("Could not infer num_layers from mha_* layers")
    num_layers = max(mha_ids) + 1

    out_dim = int(loaded_model.output_shape[-1])
    n_tril = raw_tril_size(int(n_params))
    if out_dim == int(n_params) + n_tril:
        posterior = "fullcov"
    elif out_dim == 2 * int(n_params):
        posterior = "diag"
    else:
        raise ValueError(
            f"Unexpected noparamtoken output dim {out_dim}; "
            f"expected {int(n_params) + n_tril} or {2 * int(n_params)}"
        )

    fresh = build_noparamtoken_transformer(
        n_tokens=int(n_tokens),
        feature_dim=int(feat_dim),
        n_params=int(n_params),
        n_time_patches=int(n_time_patches),
        n_freq_patches=int(n_freq_patches),
        n_tokens_erp=int(n_tokens_erp),
        d_model=int(d_model),
        num_layers=int(num_layers),
        num_heads=int(num_heads),
        ff_dim=int(ff_dim),
        dropout_rate=float(dropout_rate),
        posterior=posterior,
        return_attention=False,
    )
    fresh.set_weights(loaded_model.get_weights())
    return fresh


def _maybe_rebuild_loaded_model(
    loaded_model,
    *,
    ckpt: str,
    arch: Optional[str],
    n_tokens: int,
    feat_dim: int,
    n_params: int,
    n_tokens_erp: int,
):
    base = os.path.basename(str(ckpt)).lower()
    if arch == "noparamtoken" or "noparamtoken" in base:
        return _rebuild_noparamtoken_model(
            loaded_model,
            n_tokens=n_tokens,
            feat_dim=feat_dim,
            n_params=n_params,
            n_tokens_erp=n_tokens_erp,
        )
    return loaded_model

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-out", required=True)
    ap.add_argument("--model-dirs", nargs="+", required=True)

    ap.add_argument("--features", choices=["erp", "tfr", "hybrid"], required=True)
    ap.add_argument("--arch", choices=["paramtoken", "noparamtoken", "bilstm"], default=None)

    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--n-eval", type=int, default=1500)
    ap.add_argument("--n-post-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)

    ap.add_argument("--use-final", action="store_true")
    ap.add_argument("--out-dir", required=True)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Make sure fonts are readable even if user forgets matplotlibrc.
    plt.rcParams.update(
        {
            "font.size": max(9.0, float(plt.rcParams.get("font.size", 10.0))),
            "axes.titlesize": max(9.0, float(plt.rcParams.get("axes.titlesize", 10.0))),
            "axes.labelsize": max(9.0, float(plt.rcParams.get("axes.labelsize", 10.0))),
            "xtick.labelsize": max(8.0, float(plt.rcParams.get("xtick.labelsize", 10.0))),
            "ytick.labelsize": max(8.0, float(plt.rcParams.get("ytick.labelsize", 10.0))),
            "legend.fontsize": max(8.0, float(plt.rcParams.get("legend.fontsize", 10.0))),
        }
    )

    # ------------------------
    # Load data + meta
    # ------------------------

    X_path = _feature_path(args.data_out, args.features)
    if not os.path.isfile(X_path):
        raise FileNotFoundError(f"Missing {X_path}")

    X_all = np.load(X_path, mmap_mode="r")
    theta_all = np.load(os.path.join(args.data_out, "params.npy"), mmap_mode="r")

    N, n_tokens, feat_dim = X_all.shape
    P = theta_all.shape[1]

    tfr_meta = np.load(os.path.join(args.data_out, "tfr_meta.npz"))
    n_tokens_erp = int(tfr_meta["n_tokens_erp"])

    param_meta = np.load(os.path.join(args.data_out, "param_meta.npz"), allow_pickle=True)
    param_names = _decode_str_array(param_meta["param_names"])
    prior_low = param_meta["prior_low"].astype(np.float32)
    prior_high = param_meta["prior_high"].astype(np.float32)

    # Splits
    idx_train, idx_val, idx_test = _load_split_indices(args.model_dirs[0], args.data_out, N=N)
    if args.split == "train":
        idx = idx_train
    elif args.split == "val":
        idx = idx_val
    else:
        idx = idx_test

    idx = np.asarray(idx, dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    if args.n_eval > 0 and args.n_eval < idx.size:
        idx = rng.choice(idx, size=args.n_eval, replace=False)

    idx = np.sort(idx)

    X = np.asarray(X_all[idx], dtype=np.float32)
    theta_true = np.asarray(theta_all[idx], dtype=np.float32)
    _assert_non_degenerate_theta("theta_true", theta_true)

    # Token mask (shape N x T)
    if args.features == "hybrid":
        mask_1d = np.ones((n_tokens,), dtype=np.float32)
    elif args.features == "erp":
        mask_1d = np.zeros((n_tokens,), dtype=np.float32)
        mask_1d[:n_tokens_erp] = 1.0
    else:  # tfr
        mask_1d = np.zeros((n_tokens,), dtype=np.float32)
        mask_1d[n_tokens_erp:] = 1.0

    mask = np.broadcast_to(mask_1d[None, :], (X.shape[0], n_tokens)).astype(np.float32)

    # Standardize (mu/sd are per-channel)
    mu, sd = _load_scaler_stats(args.model_dirs[0])
    if mu.shape != (feat_dim,) or sd.shape != (feat_dim,):
        raise ValueError(f"Scaler stats shape mismatch: mu {mu.shape} sd {sd.shape} expected {(feat_dim,)}")

    Xs = (X - mu[None, None, :]) / (sd[None, None, :] + 1e-8)
    Xs = Xs * mask[:, :, None]

    # Ground-truth z
    z_true = theta_to_z(theta_true, prior_low, prior_high)

    # ------------------------
    # Load models + predict
    # ------------------------

    custom = _get_custom_objects()

    youts: List[np.ndarray] = []
    for d in args.model_dirs:
        ckpt = _find_checkpoint(d, arch=args.arch, use_final=args.use_final)
        try:
            model = tf.keras.models.load_model(ckpt, custom_objects=custom, compile=False, safe_mode=False)
        except TypeError:
            model = tf.keras.models.load_model(ckpt, custom_objects=custom, compile=False)

        model = _maybe_rebuild_loaded_model(
            model,
            ckpt=ckpt,
            arch=args.arch,
            n_tokens=n_tokens,
            feat_dim=feat_dim,
            n_params=P,
            n_tokens_erp=n_tokens_erp,
        )
        _repair_lambda_tf(model)
        y = model.predict([Xs, mask], batch_size=args.batch_size, verbose=0)
        youts.append(np.asarray(y, dtype=np.float32))

    K = len(youts)

    out_dim = youts[0].shape[1]
    n_tril = raw_tril_size(P)

    if out_dim == P + n_tril:
        posterior = "fullcov"
        mu_z_members = np.stack([y[:, :P] for y in youts], axis=0)  # (K,N,P)
        raw_tril_members = np.stack([y[:, P : P + n_tril] for y in youts], axis=0)  # (K,N,n_tril)

    elif out_dim == 2 * P:
        posterior = "diag"
        mu_z_members = np.stack([y[:, :P] for y in youts], axis=0)  # (K,N,P)
        logvar = np.stack([y[:, P:] for y in youts], axis=0)  # (K,N,P)
        std = np.exp(0.5 * logvar).astype(np.float32)

        pack = PackedTril(P=P, diag_eps=1e-3)
        raw_tril_members = pack.diag_std_to_raw(std)

    else:
        raise ValueError(
            f"Unexpected model output dim {out_dim}. Expected {2*P} (diag) or {P+n_tril} (fullcov)."
        )

    # ------------------------
    # Mixture posterior sampling
    # ------------------------

    pack = PackedTril(P=P, diag_eps=1e-3)

    S = int(args.n_post_samples)
    eps = rng.normal(size=(K, Xs.shape[0], S, P)).astype(np.float32)

    z_samps_members = np.empty((K, Xs.shape[0], S, P), dtype=np.float32)
    for k in range(K):
        L = pack.raw_to_L(raw_tril_members[k])  # (N,P,P)
        # Sample z = mu + eps @ L^T  (eps is row-vector; L is lower-triangular)
        # so that Cov(z) = L L^T.
        z_samps_members[k] = mu_z_members[k][:, None, :] + np.einsum("nsp,nqp->nsq", eps[k], L)

    comp = rng.integers(0, K, size=(Xs.shape[0], S))
    z_samps = np.empty((Xs.shape[0], S, P), dtype=np.float32)
    for k in range(K):
        sel = comp == k
        if np.any(sel):
            z_samps[sel] = z_samps_members[k][sel]

    theta_samps = z_to_theta(z_samps, prior_low, prior_high)
    theta_mean = np.mean(theta_samps, axis=1)

    # ------------------------
    # Mixture NLL in z-space
    # ------------------------

    # nll_k(z_true) using training-consistent mvn_tril_nll
    z_true_tf = tf.convert_to_tensor(z_true, dtype=tf.float32)

    logp = []
    for k in range(K):
        mu_tf = tf.convert_to_tensor(mu_z_members[k], dtype=tf.float32)
        raw_tf = tf.convert_to_tensor(raw_tril_members[k], dtype=tf.float32)
        nll_k = mvn_tril_nll(z_true_tf, mu_tf, raw_tf).numpy().astype(np.float32)  # (N,)
        logp.append(-nll_k)

    logp = np.stack(logp, axis=0)  # (K,N)
    logp_mix = _logmeanexp(logp, axis=0)
    nll_z = (-logp_mix).astype(np.float32)

    # ------------------------
    # Metrics
    # ------------------------

    rows = []
    for i, name in enumerate(param_names):
        r = _safe_pearson(theta_true[:, i], theta_mean[:, i])
        rmse = _rmse(theta_true[:, i], theta_mean[:, i])
        rng_i = float(prior_high[i] - prior_low[i])
        rows.append(
            {
                "param": name,
                "pearson_mean": r,
                "rmse_mean": rmse,
                "rmse_norm_mean": rmse / (rng_i + 1e-12),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics_path = os.path.join(args.out_dir, f"metrics_{args.split}.csv")
    metrics.to_csv(metrics_path, index=False)
    print("Wrote", metrics_path)

    # ------------------------
    # NLL histogram
    # ------------------------

    plt.figure(figsize=(7.0, 4.0))
    plt.hist(nll_z, bins=40)
    plt.xlabel("Per-example NLL in z-space")
    plt.ylabel("Count")
    plt.title(f"NLL(z) histogram ({args.features}, {args.split})")
    plt.tight_layout()
    nll_fig = os.path.join(args.out_dir, f"nllz_hist_{args.split}.png")
    plt.savefig(nll_fig, dpi=300)
    plt.close()
    print("Wrote", nll_fig)

    # ------------------------
    # Save outputs (.npz)
    # ------------------------

    out_npz = os.path.join(args.out_dir, f"eval_{args.split}_outputs.npz")

    np.savez(
        out_npz,
        eval_idx=idx,
        param_names=np.array(param_names, dtype="S"),
        features=np.array(args.features, dtype="S"),
        arch=np.array((args.arch or "auto"), dtype="S"),
        posterior=np.array(posterior, dtype="S"),
        theta_true=theta_true,
        z_true=z_true,
        # samples/means (both naming conventions)
        theta_post_mean=theta_mean,
        theta_post_samples=theta_samps,
        theta_mean=theta_mean,
        theta_samples=theta_samps,
        # diagnostics
        nll_z=nll_z,
        # optional: store member params for debugging
        mu_z_members=mu_z_members,
        raw_tril_members=raw_tril_members,
    )

    print("Wrote", out_npz)


if __name__ == "__main__":
    main()