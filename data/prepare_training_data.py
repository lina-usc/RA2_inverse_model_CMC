# data/prepare_training_data.py

from __future__ import annotations

import argparse
import json
import os
import inspect
from typing import Any, Dict, Optional

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")

from data.feature_tokens import TokenConfig, compute_erp_tokens, compute_tfr_tokens


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"ERROR: {msg}")


def _as_str_list(x: np.ndarray) -> list[str]:
    # supports dtype 'S' bytes arrays
    if x.dtype.kind in ("S", "O"):
        return [s.decode("utf-8") if isinstance(s, (bytes, np.bytes_)) else str(s) for s in x]
    return [str(s) for s in x]


def _token_cfg_fields() -> set[str]:
    # dataclass fields if present
    if hasattr(TokenConfig, "__dataclass_fields__"):
        return set(TokenConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    # fallback: signature inspection
    sig = inspect.signature(TokenConfig)  # type: ignore[arg-type]
    return set(sig.parameters.keys())


def _check_theta_probe(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise SystemExit(f"ERROR: {name} must be rank-2, got shape={arr.shape}")
    if arr.shape[0] == 0:
        raise SystemExit(f"ERROR: {name} is empty")
    if not np.all(np.isfinite(arr)):
        raise SystemExit(f"ERROR: {name} contains non-finite values")

    std = arr.std(axis=0)
    mn = float(arr.min())
    mx = float(arr.max())

    print(f"[prepare_training_data] {name} probe shape={arr.shape} min={mn:.6g} max={mx:.6g}")
    print(f"[prepare_training_data] {name} probe std per param={std}")

    if abs(mx - mn) < 1e-12 or np.all(std < 1e-12):
        raise SystemExit(
            f"ERROR: degenerate {name}: values are constant or zero-variance. "
            f"min={mn} max={mx} std={std}"
        )

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--in", dest="in_path", type=str, default="data/synthetic_cmc_dataset.h5")
    ap.add_argument("--out-dir", type=str, default="data_out")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-n", type=int, default=0, help="For debugging only. 0 = use all.")

    # -------- NEW: choose TFR backend + params --------
    ap.add_argument(
        "--tfr-method",
        choices=["stft", "morlet"],
        default="stft",
        help="TFR backend. 'stft' reproduces original behavior; 'morlet' uses MNE Morlet then bins to the SAME 25x15 patch grid.",
    )
    ap.add_argument("--tfr-fmin", type=float, default=2.0)
    ap.add_argument("--tfr-fmax", type=float, default=40.0)

    # STFT knobs (used when --tfr-method stft)
    ap.add_argument("--stft-nperseg", type=int, default=128)
    ap.add_argument("--stft-noverlap", type=int, default=112)
    ap.add_argument("--stft-nfft", type=int, default=256)

    # Morlet knobs (used when --tfr-method morlet)
    ap.add_argument("--morlet-n-freqs", type=int, default=48)
    ap.add_argument("--morlet-cycles-low", type=float, default=3.0)
    ap.add_argument("--morlet-cycles-high", type=float, default=10.0)
    ap.add_argument("--morlet-decim", type=int, default=1)
    ap.add_argument("--morlet-n-jobs", type=int, default=1)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Outputs (we keep compatibility with older training scripts by writing features.npy = hybrid)
    feat_hybrid_path = os.path.join(args.out_dir, "features.npy")
    feat_erp_path = os.path.join(args.out_dir, "features_erp.npy")
    feat_tfr_path = os.path.join(args.out_dir, "features_tfr.npy")
    params_path = os.path.join(args.out_dir, "params.npy")

    for p in [feat_hybrid_path, feat_erp_path, feat_tfr_path, params_path]:
        if os.path.exists(p) and not args.overwrite:
            raise SystemExit(f"ERROR: {p} exists. Use --overwrite.")

    # ---- compatibility guard: refuse "morlet" if feature_tokens doesn't support it
    cfg_fields = _token_cfg_fields()
    if args.tfr_method != "stft" and "tfr_method" not in cfg_fields:
        raise SystemExit(
            "ERROR: --tfr-method morlet requested, but "
            "data.feature_tokens.TokenConfig does not have field 'tfr_method'.\n"
            "You must update data/feature_tokens.py to add Morlet support "
            "(TokenConfig.tfr_method + compute_tfr_tokens backend).\n"
            "If you want to keep current behavior, omit --tfr-method (default: stft)."
        )

    with h5py.File(args.in_path, "r") as f:
        eeg = f["eeg"]  # (N, C, T)
        theta = f["theta"]  # (N, P)

        N = int(eeg.shape[0])
        C = int(eeg.shape[1])
        T = int(eeg.shape[2])
        P = int(theta.shape[1])

        if args.max_n and args.max_n > 0:
            N = min(N, int(args.max_n))

        probe_n = min(N, 2048)
        theta_probe = np.asarray(theta[:probe_n], dtype=np.float32)
        _check_theta_probe("input H5 theta", theta_probe)

        # Read forward meta from file attrs
        fs = int(f.attrs["fs"])
        duration = float(f.attrs["duration_sec"])
        stim_onset = float(f.attrs["stim_onset_sec"])

        # Confirm consistency
        _require(C == 16, f"Expected 16 channels, got {C}")
        _require(T == int(round(duration * fs)), f"EEG length mismatch: T={T}, duration*fs={int(round(duration*fs))}")

        # Build TokenConfig (filter kwargs so the script works with older TokenConfig too)
        cfg_kwargs: Dict[str, Any] = dict(
            fs=fs,
            duration=duration,
            n_channels=C,
            stim_onset=stim_onset,

            n_time_patches=25,
            n_freq_patches=15,

            f_min=float(args.tfr_fmin),
            f_max=float(args.tfr_fmax),

            # STFT params (kept for traceability even if morlet is selected)
            nperseg=int(args.stft_nperseg),
            noverlap=int(args.stft_noverlap),
            nfft=int(args.stft_nfft),

            # Morlet backend params (only used if supported and method==morlet)
            tfr_method=str(args.tfr_method),
            morlet_n_freqs=int(args.morlet_n_freqs),
            morlet_n_cycles_low=float(args.morlet_cycles_low),
            morlet_n_cycles_high=float(args.morlet_cycles_high),
            morlet_decim=int(args.morlet_decim),
            morlet_n_jobs=int(args.morlet_n_jobs),
        )
        cfg = TokenConfig(**{k: v for k, v in cfg_kwargs.items() if k in cfg_fields})

        n_tokens_erp = int(cfg.n_time_patches)
        n_tokens_tfr = int(cfg.n_time_patches * cfg.n_freq_patches)
        n_tokens_total = int(n_tokens_erp + n_tokens_tfr)

        # memmaps
        X_hybrid = np.lib.format.open_memmap(
            feat_hybrid_path, mode="w+", dtype=np.float32, shape=(N, n_tokens_total, C)
        )
        X_erp = np.lib.format.open_memmap(
            feat_erp_path, mode="w+", dtype=np.float32, shape=(N, n_tokens_total, C)
        )
        X_tfr = np.lib.format.open_memmap(
            feat_tfr_path, mode="w+", dtype=np.float32, shape=(N, n_tokens_total, C)
        )
        Y = np.lib.format.open_memmap(params_path, mode="w+", dtype=np.float32, shape=(N, P))

        # store meta from first example
        first_tfr_meta: Optional[Dict[str, Any]] = None

        print(f"[prepare_training_data] reading: {args.in_path}")
        print(f"[prepare_training_data] out_dir: {args.out_dir}")
        print(f"[prepare_training_data] N={N} | channels={C} | time={T} | params={P}")
        print(f"[prepare_training_data] tokens: ERP={n_tokens_erp} TFR={n_tokens_tfr} total={n_tokens_total}")
        print(f"[prepare_training_data] TFR backend: {args.tfr_method}")
        print(f"[prepare_training_data] f_min={float(args.tfr_fmin)} f_max={float(args.tfr_fmax)}")

        if args.tfr_method == "stft":
            print(
                f"[prepare_training_data] STFT: nperseg={int(args.stft_nperseg)} "
                f"noverlap={int(args.stft_noverlap)} nfft={int(args.stft_nfft)}"
            )
        else:
            print(
                f"[prepare_training_data] Morlet: n_freqs={int(args.morlet_n_freqs)} "
                f"cycles_low={float(args.morlet_cycles_low)} cycles_high={float(args.morlet_cycles_high)} "
                f"decim={int(args.morlet_decim)} n_jobs={int(args.morlet_n_jobs)}"
            )

        for i in range(N):
            x = np.asarray(eeg[i], dtype=np.float32)   # (C,T)
            th = np.asarray(theta[i], dtype=np.float32)

            erp_tok = compute_erp_tokens(x, cfg)              # (25,16)
            tfr_tok, tfr_meta = compute_tfr_tokens(x, cfg)    # (375,16) + meta

            if first_tfr_meta is None:
                first_tfr_meta = tfr_meta
                # optional consistency check if feature_tokens exposes backend in meta
                if "tfr_backend" in tfr_meta:
                    b = tfr_meta["tfr_backend"]
                    if isinstance(b, (bytes, np.bytes_)):
                        b = b.decode("utf-8")
                    elif isinstance(b, np.ndarray) and b.dtype.kind == "S":
                        b = b.astype(str)[()]
                    b = str(b)
                    if b and (b != args.tfr_method):
                        raise SystemExit(
                            f"ERROR: Requested --tfr-method {args.tfr_method}, but "
                            f"compute_tfr_tokens produced backend '{b}'. "
                            "Check your data/feature_tokens.py implementation."
                        )

            hybrid = np.concatenate([erp_tok, tfr_tok], axis=0).astype(np.float32)  # (400,16)

            # padded ablations (same shape, same architecture later)
            erp_only = np.zeros_like(hybrid, dtype=np.float32)
            tfr_only = np.zeros_like(hybrid, dtype=np.float32)
            erp_only[:n_tokens_erp] = erp_tok
            tfr_only[n_tokens_erp:] = tfr_tok

            X_hybrid[i] = hybrid
            X_erp[i] = erp_only
            X_tfr[i] = tfr_only
            Y[i] = th

            if (i + 1) % 500 == 0:
                print(f"  processed {i+1}/{N}")

        X_hybrid.flush()
        X_erp.flush()
        X_tfr.flush()
        Y.flush()

        Y_check = np.load(params_path, mmap_mode="r")
        _check_theta_probe("written params.npy", np.asarray(Y_check[:probe_n], dtype=np.float32))

        # Write param_meta.npz from H5 contents (paper-ready)
        param_names = _as_str_list(np.asarray(f["param_names"]))
        prior_low = np.asarray(f["prior_low"], dtype=np.float32)
        prior_high = np.asarray(f["prior_high"], dtype=np.float32)
        prior_dist = _as_str_list(np.asarray(f["prior_dist"]))
        prior_params = np.asarray(f["prior_params"], dtype=np.float32)
        prior_spec_json = np.asarray(f["prior_spec_json"]).astype(str)[()]  # stored as a scalar string

        np.savez(
            os.path.join(args.out_dir, "param_meta.npz"),
            param_names=np.array(param_names, dtype="S"),
            prior_low=prior_low,
            prior_high=prior_high,
            prior_dist=np.array(prior_dist, dtype="S"),
            prior_params=prior_params,
            prior_spec_json=np.array(prior_spec_json, dtype="S"),
        )

        # Write tfr_meta.npz from cfg + first_tfr_meta
        _require(first_tfr_meta is not None, "Failed to capture TFR meta from first sample.")

        # store channel names
        ch_names = _as_str_list(np.asarray(f["ch_names"]))

        # nice human-readable method label for meta
        tfr_method_label = (
            "STFT_power_db_ratio_pre_stim" if args.tfr_method == "stft" else "Morlet_power_db_ratio_pre_stim"
        )

        # Some fields may not exist in older feature_tokens meta; enforce required ones
        for k in ["stft_f", "stft_t", "time_edges", "freq_edges"]:
            _require(k in first_tfr_meta, f"Missing key '{k}' in tfr_meta returned by compute_tfr_tokens().")

        np.savez(
            os.path.join(args.out_dir, "tfr_meta.npz"),
            fs=np.array(fs, dtype=np.int32),
            duration=np.array(duration, dtype=np.float32),
            n_channels=np.array(C, dtype=np.int32),
            stim_onset=np.array(stim_onset, dtype=np.float32),
            stim_sigma=np.array(float(f.attrs["stim_sigma_sec"]), dtype=np.float32),
            warmup_sec=np.array(float(f.attrs["warmup_sec"]), dtype=np.float32),
            bandpass=np.array([float(f.attrs["bandpass_lo_hz"]), float(f.attrs["bandpass_hi_hz"])], dtype=np.float32),

            n_time_patches=np.array(cfg.n_time_patches, dtype=np.int32),
            n_freq_patches=np.array(cfg.n_freq_patches, dtype=np.int32),
            n_tokens_erp=np.array(n_tokens_erp, dtype=np.int32),
            n_tokens_tfr=np.array(n_tokens_tfr, dtype=np.int32),
            n_tokens_total=np.array(n_tokens_total, dtype=np.int32),
            feature_dim=np.array(C, dtype=np.int32),

            token_order=np.array("ERP_then_TFR_time_major_freq_minor", dtype="S"),

            # method identity + band
            tfr_backend=np.array(str(args.tfr_method), dtype="S"),
            tfr_method=np.array(tfr_method_label, dtype="S"),
            f_min=np.array(float(args.tfr_fmin), dtype=np.float32),
            f_max=np.array(float(args.tfr_fmax), dtype=np.float32),

            # keep STFT params for traceability (even if morlet)
            nperseg=np.array(int(args.stft_nperseg), dtype=np.int32),
            noverlap=np.array(int(args.stft_noverlap), dtype=np.int32),
            nfft=np.array(int(args.stft_nfft), dtype=np.int32),

            # morlet params (always written, harmless if stft)
            morlet_n_freqs=np.array(int(args.morlet_n_freqs), dtype=np.int32),
            morlet_cycles_low=np.array(float(args.morlet_cycles_low), dtype=np.float32),
            morlet_cycles_high=np.array(float(args.morlet_cycles_high), dtype=np.float32),
            morlet_decim=np.array(int(args.morlet_decim), dtype=np.int32),
            morlet_n_jobs=np.array(int(args.morlet_n_jobs), dtype=np.int32),

            ch_names=np.array(ch_names, dtype="S"),

            # backend meta returned by tokenizer
            stft_f=np.asarray(first_tfr_meta["stft_f"], dtype=np.float32),
            stft_t=np.asarray(first_tfr_meta["stft_t"], dtype=np.float32),
            time_edges=np.asarray(first_tfr_meta["time_edges"], dtype=np.float32),
            freq_edges=np.asarray(first_tfr_meta["freq_edges"], dtype=np.float32),
        )

        # Save a small JSON provenance log
        log = {
            "in_h5": args.in_path,
            "out_dir": args.out_dir,
            "N": N,
            "channels": ch_names,
            "params": param_names,
            "features_files": {
                "hybrid": os.path.basename(feat_hybrid_path),
                "erp_only_padded": os.path.basename(feat_erp_path),
                "tfr_only_padded": os.path.basename(feat_tfr_path),
                "params": os.path.basename(params_path),
            },
            "cli": vars(args),
            "token_cfg_effective": cfg.__dict__,
        }
        with open(os.path.join(args.out_dir, "prepare_training_data_log.json"), "w") as fp:
            json.dump(log, fp, indent=2)

    print("[prepare_training_data] DONE")
    print(f"  wrote: {feat_hybrid_path}")
    print(f"  wrote: {feat_erp_path}")
    print(f"  wrote: {feat_tfr_path}")
    print(f"  wrote: {params_path}")
    print(f"  wrote: {os.path.join(args.out_dir, 'tfr_meta.npz')}")
    print(f"  wrote: {os.path.join(args.out_dir, 'param_meta.npz')}")


if __name__ == "__main__":
    main()
