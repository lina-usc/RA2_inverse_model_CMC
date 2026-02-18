from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=str, default="data/synthetic_cmc_dataset.h5")
    ap.add_argument("--out-dir", type=str, default="data_out")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-n", type=int, default=0, help="For debugging only. 0 = use all.")
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

    with h5py.File(args.in_path, "r") as f:
        eeg = f["eeg"]  # (N, C, T)
        theta = f["theta"]  # (N, P)

        N = int(eeg.shape[0])
        C = int(eeg.shape[1])
        T = int(eeg.shape[2])
        P = int(theta.shape[1])

        if args.max_n and args.max_n > 0:
            N = min(N, int(args.max_n))

        # Read forward meta from file attrs
        fs = int(f.attrs["fs"])
        duration = float(f.attrs["duration_sec"])
        stim_onset = float(f.attrs["stim_onset_sec"])

        # Confirm consistency
        _require(C == 16, f"Expected 16 channels, got {C}")
        _require(T == int(round(duration * fs)), f"EEG length mismatch: T={T}, duration*fs={int(round(duration*fs))}")

        # TokenConfig from dataset meta (paper-traceable)
        cfg = TokenConfig(
            fs=fs,
            duration=duration,
            n_channels=C,
            stim_onset=stim_onset,
            # keep your current explicit TFR settings
            n_time_patches=25,
            n_freq_patches=15,
            f_min=2.0,
            f_max=40.0,
            nperseg=128,
            noverlap=112,
            nfft=256,
        )

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
        print(f"[prepare_training_data] N={N} | channels={C} | time={T} | params={P}")
        print(f"[prepare_training_data] tokens: ERP={n_tokens_erp} TFR={n_tokens_tfr} total={n_tokens_total}")

        for i in range(N):
            x = np.asarray(eeg[i], dtype=np.float32)   # (C,T)
            th = np.asarray(theta[i], dtype=np.float32)

            erp_tok = compute_erp_tokens(x, cfg)              # (25,16)
            tfr_tok, tfr_meta = compute_tfr_tokens(x, cfg)    # (375,16) + meta

            if first_tfr_meta is None:
                first_tfr_meta = tfr_meta

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
            tfr_method=np.array("STFT_power_db_ratio_pre_stim", dtype="S"),
            f_min=np.array(cfg.f_min, dtype=np.float32),
            f_max=np.array(cfg.f_max, dtype=np.float32),
            nperseg=np.array(cfg.nperseg, dtype=np.int32),
            noverlap=np.array(cfg.noverlap, dtype=np.int32),
            nfft=np.array(cfg.nfft, dtype=np.int32),
            ch_names=np.array(ch_names, dtype="S"),
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
            "token_cfg": cfg.__dict__,
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
