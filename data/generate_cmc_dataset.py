#data/generate_cmc_dataset.py

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np

from sim import simulate_eeg
from data.feature_tokens import TokenConfig, compute_erp_tokens, compute_tfr_tokens, regime_reject
from data.priors import build_prior_spec, sample_theta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="data_out")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--stim-causal", action="store_true", help="Use causal (truncated) Gaussian stimulus: stim(t)=0 for t<onset.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Fixed, paper-explicit settings ----------
    cfg = TokenConfig()

    # Forward sim settings (explicit, saved to JSON + NPZ)
    sim_cfg: Dict[str, object] = {
        "fs": cfg.fs,
        "duration": cfg.duration,
        "n_channels": cfg.n_channels,
        "stim_onset": cfg.stim_onset,
        "stim_sigma": 0.05,     # Gaussian bump width (seconds)
        "stim_causal": bool(getattr(args, "stim_causal", False)),
        "warmup_sec": 3.0,      # MUST be > 0 (steady state)
        "n_sources": 3,
        "n_trials": 10,
        "input_noise_std": 0.2,
        "sensor_noise_std": 2.0,
        "internal_fs": 1000,
        "bandpass": (0.5, 40.0),
        "baseline_correct": True,
        "baseline_window": None,    # baseline = [0, stim_onset)
        "downsample_method": "slice",
        "uV_scale": 100.0,
    }

    # Priors
    names, prior_low, prior_high, prior_dist, prior_params, prior_json = build_prior_spec()
    P = len(names)

    n_time = int(round(cfg.duration * cfg.fs))
    n_tokens_erp = cfg.n_time_patches
    n_tokens_tfr = cfg.n_time_patches * cfg.n_freq_patches
    n_tokens = n_tokens_erp + n_tokens_tfr
    feature_dim = cfg.n_channels

    feat_path = os.path.join(args.out_dir, "features.npy")
    param_path = os.path.join(args.out_dir, "params.npy")

    if (os.path.exists(feat_path) or os.path.exists(param_path)) and not args.overwrite:
        raise SystemExit(
    "ERROR: data_out/features.npy or params.npy exists. "
    "Use --overwrite to regenerate."
    )
    X = np.lib.format.open_memmap(
    feat_path,
    mode="w+",
    dtype=np.float32,
    shape=(args.n_samples, n_tokens, feature_dim),
    )

    Y = np.lib.format.open_memmap(param_path, mode="w+", dtype=np.float32, shape=(args.n_samples, P))

    # Save metadata (priors reportable)
    np.savez(
        os.path.join(args.out_dir, "param_meta.npz"),
        param_names=np.array(names, dtype="S"),
        prior_low=prior_low.astype(np.float32),
        prior_high=prior_high.astype(np.float32),
        prior_dist=np.array(prior_dist, dtype="S"),
        prior_params=prior_params.astype(np.float32),
        prior_spec_json=np.array(prior_json, dtype="S"),
    )

    # Save meta for plotting/reporting
    np.savez(
        os.path.join(args.out_dir, "tfr_meta.npz"),
        fs=np.array(cfg.fs, dtype=np.int32),
        duration=np.array(cfg.duration, dtype=np.float32),
        n_channels=np.array(cfg.n_channels, dtype=np.int32),
        stim_onset=np.array(cfg.stim_onset, dtype=np.float32),
        stim_sigma=np.array(sim_cfg["stim_sigma"], dtype=np.float32),
        warmup_sec=np.array(sim_cfg["warmup_sec"], dtype=np.float32),
        n_time_patches=np.array(cfg.n_time_patches, dtype=np.int32),
        n_freq_patches=np.array(cfg.n_freq_patches, dtype=np.int32),
        n_tokens_erp=np.array(n_tokens_erp, dtype=np.int32),
        f_min=np.array(cfg.f_min, dtype=np.float32),
        f_max=np.array(cfg.f_max, dtype=np.float32),
        nperseg=np.array(cfg.nperseg, dtype=np.int32),
        noverlap=np.array(cfg.noverlap, dtype=np.int32),
        nfft=np.array(cfg.nfft, dtype=np.int32),
        bandpass=np.array(sim_cfg["bandpass"], dtype=np.float32),
    )

    rng = np.random.default_rng(args.seed)
    reject_counts = {}
    qc_eegs = []
    qc_thetas = []
    max_qc_store = 8

    i = 0
    tries = 0
    print(f"[generate_cmc_dataset] target N={args.n_samples} | tokens={n_tokens} feat_dim={feature_dim}")

    while i < args.n_samples:
        tries += 1

        theta, p = sample_theta(rng, names, prior_low, prior_high, prior_dist, prior_params)

        sim_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        eeg = simulate_eeg(
            params=p,
            seed=sim_seed,
            fs=int(sim_cfg["fs"]),
            duration=float(sim_cfg["duration"]),
            n_channels=int(sim_cfg["n_channels"]),
            stim_onset=float(sim_cfg["stim_onset"]),
            stim_sigma=float(sim_cfg["stim_sigma"]),
            stim_causal=bool(sim_cfg.get("stim_causal", False)),
            warmup_sec=float(sim_cfg["warmup_sec"]),
            n_sources=int(sim_cfg["n_sources"]),
            n_trials=int(sim_cfg["n_trials"]),
            input_noise_std=float(sim_cfg["input_noise_std"]),
            sensor_noise_std=float(sim_cfg["sensor_noise_std"]),
            internal_fs=int(sim_cfg["internal_fs"]),
            bandpass=tuple(sim_cfg["bandpass"]),
            baseline_correct=bool(sim_cfg["baseline_correct"]),
            baseline_window=None,
            downsample_method=str(sim_cfg["downsample_method"]),
            uV_scale=float(sim_cfg["uV_scale"]),
        )

        ok, reason = regime_reject(eeg, cfg)
        reject_counts[reason] = reject_counts.get(reason, 0) + 1
        if not ok:
            continue

        erp_tok = compute_erp_tokens(eeg, cfg)           # (25,16)
        tfr_tok, _ = compute_tfr_tokens(eeg, cfg)        # (375,16)
        tokens = np.concatenate([erp_tok, tfr_tok], axis=0).astype(np.float32)

        X[i] = tokens
        Y[i] = theta

        if len(qc_eegs) < max_qc_store:
            qc_eegs.append(eeg.astype(np.float32))
            qc_thetas.append(theta.astype(np.float32))

        i += 1
        if i % 500 == 0:
            acc_rate = i / tries
            print(f"  accepted {i}/{args.n_samples} | tries={tries} | accept_rate={acc_rate:.3f}")

    X.flush()
    Y.flush()

    # Save a few raw examples for ERP sanity plots
    t = (np.arange(n_time, dtype=np.float32) / float(cfg.fs)).astype(np.float32)
    np.savez(
        os.path.join(args.out_dir, "qc_examples.npz"),
        eeg=np.stack(qc_eegs, axis=0),
        theta=np.stack(qc_thetas, axis=0),
        t=t,
        param_names=np.array(names, dtype="S"),
    )

    gen_log = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "tries": tries,
        "accept_rate": float(args.n_samples / tries),
        "reject_counts": reject_counts,
        "sim_cfg": {**sim_cfg, "bandpass": list(sim_cfg["bandpass"])},
        "token_cfg": cfg.__dict__,
        "param_names": names,
    }
    with open(os.path.join(args.out_dir, "generation_log.json"), "w") as f:
        json.dump(gen_log, f, indent=2)

    print("[generate_cmc_dataset] DONE")
    print("Reject counts:", reject_counts)


if __name__ == "__main__":
    main()
