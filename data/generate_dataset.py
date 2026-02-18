from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sim import simulate_eeg
from sim.leadfield_mne import make_leadfield, DEFAULT_16_CH_NAMES
from sim.regime_filter import RegimeFilterConfig, regime_reject

from data.priors import build_prior_spec, sample_theta


def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="Number of ACCEPTED samples to store.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/synthetic_cmc_dataset.h5")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    qc_dir = os.path.join("plots", "qc_dataset")
    os.makedirs(qc_dir, exist_ok=True)

    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"ERROR: {args.out} exists. Use --overwrite.")

    # ---------------- Forward config (paper-explicit) ----------------
    fs = 250
    duration = 2.0
    n_channels = 16

    stim_onset = 0.5
    stim_sigma = 0.05
    warmup_sec = 3.0  # MUST be > 0

    n_sources = 3
    n_trials = 10

    input_noise_std = 0.2
    sensor_noise_std = 2.0

    internal_fs = 1000
    bandpass = (0.5, 40.0)

    baseline_correct = True
    baseline_window = None
    downsample_method = "slice"
    uV_scale = 100.0

    # deterministic leadfield (stored in file)
    L, info, lf_meta, ch_pos, src_pos, src_ori = make_leadfield(fs=fs, n_sources=n_sources, seed=0)
    ch_names = list(DEFAULT_16_CH_NAMES)

    # regime filter config
    rf_cfg = RegimeFilterConfig(fs=fs, duration=duration, stim_onset=stim_onset)

    # ---------------- Priors (paper-explicit) ----------------
    names, prior_low, prior_high, prior_dist, prior_params, prior_json = build_prior_spec()
    P = len(names)

    # ---------------- H5 allocation ----------------
    n_time = int(round(duration * fs))

    dt_str = h5py.string_dtype(encoding="utf-8")

    with h5py.File(args.out, "w") as f:
        # core datasets (accepted-only)
        d_eeg = f.create_dataset(
            "eeg",
            shape=(args.n, n_channels, n_time),
            dtype="float32",
            chunks=(1, n_channels, n_time),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        d_theta = f.create_dataset("theta", shape=(args.n, P), dtype="float32")
        d_sim_seed = f.create_dataset("sim_seed", shape=(args.n,), dtype="int64")

        # store leadfield + channel names
        f.create_dataset("leadfield", data=L.astype(np.float32))
        f.create_dataset("ch_names", data=np.array(ch_names, dtype="S"))

        # store prior meta inside H5 for provenance
        f.create_dataset("param_names", data=np.array(names, dtype="S"))
        f.create_dataset("prior_low", data=prior_low.astype(np.float32))
        f.create_dataset("prior_high", data=prior_high.astype(np.float32))
        f.create_dataset("prior_dist", data=np.array(prior_dist, dtype="S"))
        f.create_dataset("prior_params", data=prior_params.astype(np.float32))
        f.create_dataset("prior_spec_json", data=np.array(prior_json, dtype=dt_str), dtype=dt_str)

        # forward settings (attributes)
        f.attrs["fs"] = fs
        f.attrs["duration_sec"] = float(duration)
        f.attrs["n_channels"] = n_channels
        f.attrs["stim_shape"] = "gaussian"
        f.attrs["stim_onset_sec"] = float(stim_onset)
        f.attrs["stim_sigma_sec"] = float(stim_sigma)
        f.attrs["warmup_sec"] = float(warmup_sec)
        f.attrs["n_sources"] = int(n_sources)
        f.attrs["n_trials"] = int(n_trials)
        f.attrs["input_noise_std"] = float(input_noise_std)
        f.attrs["sensor_noise_std"] = float(sensor_noise_std)
        f.attrs["internal_fs"] = int(internal_fs)
        f.attrs["bandpass_lo_hz"] = float(bandpass[0])
        f.attrs["bandpass_hi_hz"] = float(bandpass[1])
        f.attrs["baseline_correct"] = int(bool(baseline_correct))
        f.attrs["downsample_method"] = str(downsample_method)
        f.attrs["uV_scale"] = float(uV_scale)

        # leadfield meta JSON
        lf_meta_json = json.dumps({
            "montage_name": lf_meta.montage_name,
            "head_radius_m": lf_meta.head_radius_m,
            "source_radius_m": lf_meta.source_radius_m,
            "n_sources": lf_meta.n_sources,
            "seed": lf_meta.seed,
        }, indent=2)
        f.create_dataset("leadfield_meta_json", data=np.array(lf_meta_json, dtype=dt_str), dtype=dt_str)

        # ---------------- Generation loop ----------------
        rng = np.random.default_rng(args.seed)

        reject_counts: Dict[str, int] = {}
        tries = 0
        i = 0

        attempted_theta: List[np.ndarray] = []
        attempted_ok: List[bool] = []

        print(f"[generate_dataset] target accepted N={args.n} -> {args.out}")
        while i < args.n:
            tries += 1

            theta, p = sample_theta(rng, names, prior_low, prior_high, prior_dist, prior_params)

            sim_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            eeg = simulate_eeg(
                params=p,
                fs=fs,
                duration=duration,
                n_channels=n_channels,
                seed=sim_seed,
                bandpass=bandpass,
                stim_onset=stim_onset,
                stim_sigma=stim_sigma,
                n_sources=n_sources,
                leadfield=L,
                sensor_noise_std=sensor_noise_std,
                n_trials=n_trials,
                input_noise_std=input_noise_std,
                internal_fs=internal_fs,
                baseline_correct=baseline_correct,
                baseline_window=baseline_window,
                warmup_sec=warmup_sec,
                downsample_method=downsample_method,
                uV_scale=uV_scale,
            )

            ok, reason = regime_reject(eeg, rf_cfg)
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

            attempted_theta.append(theta.copy())
            attempted_ok.append(bool(ok))

            if not ok:
                continue

            d_eeg[i] = eeg.astype(np.float32)
            d_theta[i] = theta.astype(np.float32)
            d_sim_seed[i] = np.int64(sim_seed)

            i += 1
            if i % 500 == 0:
                acc_rate = i / tries
                print(f"  accepted {i}/{args.n} | tries={tries} | accept_rate={acc_rate:.3f}")

        # end while
        accept_rate = float(args.n / tries)
        print("[generate_dataset] DONE")
        print("reject_counts:", reject_counts)
        print("accept_rate:", accept_rate)

        # store generation summary inside file too
        f.attrs["tries"] = int(tries)
        f.attrs["accept_rate"] = accept_rate
        f.create_dataset("reject_counts_json", data=np.array(json.dumps(reject_counts, indent=2), dtype=dt_str), dtype=dt_str)

    # ---------------- QC plots after file close ----------------
    attempted_theta_arr = np.stack(attempted_theta, axis=0).astype(np.float32)  # (tries,P)
    attempted_ok_arr = np.array(attempted_ok, dtype=bool)
    accepted_theta_arr = attempted_theta_arr[attempted_ok_arr]

    # save arrays for paper/provenance
    np.save(os.path.join(qc_dir, "attempted_theta.npy"), attempted_theta_arr)
    np.save(os.path.join(qc_dir, "accepted_theta.npy"), accepted_theta_arr)
    np.save(os.path.join(qc_dir, "attempted_ok.npy"), attempted_ok_arr)

    # plot: rejection reasons
    plt.figure(figsize=(7, 4))
    keys = list(reject_counts.keys())
    vals = [reject_counts[k] for k in keys]
    plt.bar(keys, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("count")
    plt.title(f"Regime filter outcomes (accept_rate={accept_rate:.3f})")
    _savefig(os.path.join(qc_dir, "00_reject_reasons.png"))

    # plot: prior vs accepted hist (grid)
    ncols = 3
    nrows = int(np.ceil(P / ncols))
    plt.figure(figsize=(12, 3.5 * nrows))
    for j in range(P):
        plt.subplot(nrows, ncols, j + 1)
        plt.hist(attempted_theta_arr[:, j], bins=40, alpha=0.6, density=True, label="prior draws (attempted)")
        plt.hist(accepted_theta_arr[:, j], bins=40, alpha=0.6, density=True, label="accepted (effective prior)")
        plt.title(names[j])
        plt.xlabel("value")
        plt.ylabel("density")
        plt.legend(fontsize=8)
    _savefig(os.path.join(qc_dir, "01_prior_vs_effective_prior_hist.png"))

    # plot: acceptance rate vs parameter (binned)
    plt.figure(figsize=(12, 3.5 * nrows))
    for j in range(P):
        x = attempted_theta_arr[:, j]
        ok = attempted_ok_arr.astype(np.float32)

        bins = np.linspace(float(np.min(x)), float(np.max(x)), 21)
        centers = 0.5 * (bins[:-1] + bins[1:])
        acc = np.zeros_like(centers, dtype=np.float32)

        for b in range(len(centers)):
            m = (x >= bins[b]) & (x < bins[b + 1])
            if np.sum(m) < 10:
                acc[b] = np.nan
            else:
                acc[b] = float(np.mean(ok[m]))

        plt.subplot(nrows, ncols, j + 1)
        plt.plot(centers, acc, marker="o", linewidth=2)
        plt.ylim(0.0, 1.0)
        plt.title(names[j])
        plt.xlabel("value")
        plt.ylabel("P(accept | value)")
    _savefig(os.path.join(qc_dir, "02_acceptance_rate_vs_param.png"))

    # save a JSON log for the paper
    log = {
        "out": args.out,
        "seed": args.seed,
        "n_accepted": args.n,
        "tries": int(tries),
        "accept_rate": float(accept_rate),
        "reject_counts": reject_counts,
        "forward_cfg": {
            "fs": fs,
            "duration": duration,
            "n_channels": n_channels,
            "stim_onset": stim_onset,
            "stim_sigma": stim_sigma,
            "warmup_sec": warmup_sec,
            "n_sources": n_sources,
            "n_trials": n_trials,
            "input_noise_std": input_noise_std,
            "sensor_noise_std": sensor_noise_std,
            "internal_fs": internal_fs,
            "bandpass": list(bandpass),
            "baseline_correct": baseline_correct,
            "downsample_method": downsample_method,
            "uV_scale": uV_scale,
        },
        "param_names": names,
        "prior_spec_json": json.loads(prior_json),
        "qc_dir": qc_dir,
    }
    with open(os.path.join(qc_dir, "generation_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"[generate_dataset] QC plots/logs written to: {qc_dir}")


if __name__ == "__main__":
    main()
