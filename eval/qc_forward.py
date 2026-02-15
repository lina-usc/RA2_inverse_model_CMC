from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from sim import simulate_eeg
from sim.stimulus import gaussian_bump
from sim.leadfield_mne import make_leadfield, DEFAULT_16_CH_NAMES

from data.priors import build_prior_spec, sample_theta
from data.feature_tokens import TokenConfig
from sim.regime_filter import RegimeFilterConfig, regime_reject



def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="plots/qc_forward")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = TokenConfig()

    # deterministic leadfield used (plot + reuse)
    L, info, meta, ch_pos, src_pos, src_ori = make_leadfield(fs=cfg.fs, n_sources=3, seed=0)

    # priors (for sampling test params)
    names, low, high, dist, prior_params, prior_json = build_prior_spec()

    rng = np.random.default_rng(args.seed)

    counts: Dict[str, int] = {}
    accepted = []
    rejected = []

    t0 = time.perf_counter()

    for _ in range(args.n):
        theta, p = sample_theta(rng, names, low, high, dist, prior_params)
        sim_seed = int(rng.integers(0, np.iinfo(np.int32).max))

        eeg = simulate_eeg(
            params=p,
            fs=cfg.fs,
            duration=cfg.duration,
            n_channels=cfg.n_channels,
            seed=sim_seed,
            stim_onset=cfg.stim_onset,
            stim_sigma=0.05,
            n_sources=3,
            leadfield=L,
            sensor_noise_std=2.0,
            n_trials=10,
            input_noise_std=0.2,
            internal_fs=1000,
            bandpass=(0.5, 40.0),
            baseline_correct=True,
            baseline_window=None,
            warmup_sec=3.0,
            downsample_method="slice",
            uV_scale=100.0,
        )

        ok, reason = regime_reject(eeg, cfg)
        counts[reason] = counts.get(reason, 0) + 1

        if ok and len(accepted) < 6:
            accepted.append(eeg)
        if (not ok) and len(rejected) < 6:
            rejected.append(eeg)

    t1 = time.perf_counter()
    sec_per_sim = (t1 - t0) / max(1, args.n)

    n_ok = counts.get("ok", 0)
    n_total = sum(counts.values())

    summary = {
        "n": args.n,
        "seed": args.seed,
        "accepted": int(n_ok),
        "total": int(n_total),
        "accept_rate": float(n_ok / max(1, n_total)),
        "seconds_total": float(t1 - t0),
        "seconds_per_sim": float(sec_per_sim),
        "reject_counts": counts,
        "stimulus": {"shape": "gaussian", "stim_onset_sec": cfg.stim_onset, "stim_sigma_sec": 0.05},
        "warmup_sec": 3.0,
        "fs": cfg.fs,
        "duration_sec": cfg.duration,
        "channels": list(DEFAULT_16_CH_NAMES),
        "leadfield_meta": {
            "montage_name": meta.montage_name,
            "head_radius_m": meta.head_radius_m,
            "source_radius_m": meta.source_radius_m,
            "n_sources": meta.n_sources,
            "seed": meta.seed,
        },
        "prior_spec_json": prior_json,
    }
    with open(os.path.join(args.out, "qc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ----- time axis -----
    t = np.arange(int(round(cfg.duration * cfg.fs)), dtype=np.float32) / float(cfg.fs)

    # ----- plot 1: stimulus -----
    stim = gaussian_bump(t, onset=cfg.stim_onset, sigma=0.05, amp=1.0)
    plt.figure(figsize=(8, 3))
    plt.plot(t, stim, linewidth=2)
    plt.axvline(cfg.stim_onset, linestyle="--", linewidth=2, label="stim onset")
    plt.xlabel("Time (s)")
    plt.ylabel("Stim amplitude (a.u.)")
    plt.title("Stimulus definition: Gaussian bump (amp=1.0)")
    plt.legend()
    _savefig(os.path.join(args.out, "00_stimulus.png"))

    # ----- plot 2: leadfield heatmap -----
    plt.figure(figsize=(7, 5))
    plt.imshow(L, aspect="auto")
    plt.colorbar(label="Leadfield weight (a.u.)")
    plt.yticks(np.arange(cfg.n_channels), list(DEFAULT_16_CH_NAMES), fontsize=8)
    plt.xticks(np.arange(L.shape[1]), [f"src{j}" for j in range(L.shape[1])])
    plt.title("Deterministic 16-ch leadfield (MNE montage positions)")
    _savefig(os.path.join(args.out, "01_leadfield.png"))

    # ----- plot 3: sensor/source geometry -----
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(ch_pos[:, 0], ch_pos[:, 1], ch_pos[:, 2], s=30, label="EEG sensors")
    ax.scatter(src_pos[:, 0], src_pos[:, 1], src_pos[:, 2], s=80, marker="^", label="sources")
    ax.set_title("Sensor + source geometry (meters)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend()
    _savefig(os.path.join(args.out, "02_geometry.png"))

    # ----- plot 4: ERP + GFP accepted -----
    if len(accepted) > 0:
        eeg0 = accepted[0]
        gfp = np.std(eeg0, axis=0)

        plt.figure(figsize=(10, 4))
        for ch in range(min(cfg.n_channels, eeg0.shape[0])):
            plt.plot(t, eeg0[ch], linewidth=0.8, alpha=0.6)
        plt.plot(t, gfp, linewidth=2.5, label="GFP (std across channels)")
        plt.axvline(cfg.stim_onset, linestyle="--", linewidth=2, label="stim onset")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (uV)")
        plt.title("ERP sanity check (accepted example): channels + GFP")
        plt.legend(loc="upper right")
        _savefig(os.path.join(args.out, "03_erp_gfp_accepted.png"))

    # ----- plot 5: regime filter counts -----
    plt.figure(figsize=(7, 4))
    keys = list(counts.keys())
    vals = [counts[k] for k in keys]
    plt.bar(keys, vals)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("count")
    plt.title(f"Regime filter outcomes (accept_rate={summary['accept_rate']:.3f})")
    _savefig(os.path.join(args.out, "04_regime_counts.png"))

    # ----- plot 6: GFP accepted vs rejected -----
    plt.figure(figsize=(10, 4))
    for i, eeg in enumerate(accepted[:5]):
        plt.plot(t, np.std(eeg, axis=0), linewidth=1.8, alpha=0.8, label="accepted" if i == 0 else None)
    for i, eeg in enumerate(rejected[:5]):
        plt.plot(t, np.std(eeg, axis=0), linewidth=1.2, alpha=0.6, linestyle="--", label="rejected" if i == 0 else None)
    plt.axvline(cfg.stim_onset, linestyle="--", linewidth=2, color="k", label="stim onset")
    plt.xlabel("Time (s)")
    plt.ylabel("GFP (uV)")
    plt.title("GFP traces: accepted (solid) vs rejected (dashed)")
    plt.legend(loc="upper right")
    _savefig(os.path.join(args.out, "05_gfp_acc_vs_rej.png"))

    # ----- plot 7: PSD sanity -----
    if len(accepted) > 0:
        eeg0 = accepted[0]
        stim_idx = int(round(cfg.stim_onset * cfg.fs))
        pre = eeg0[:, :stim_idx]
        post = eeg0[:, stim_idx : min(eeg0.shape[1], stim_idx + int(round(0.6 * cfg.fs)))]

        def avg_psd(x):
            psds = []
            for ch in range(x.shape[0]):
                nperseg = min(256, x.shape[1])
                f, pxx = welch(x[ch], fs=cfg.fs, nperseg=nperseg)
                psds.append(pxx)
            return f, np.mean(np.stack(psds, axis=0), axis=0)

        f_pre, p_pre = avg_psd(pre)
        f_post, p_post = avg_psd(post)

        plt.figure(figsize=(8, 4))
        plt.semilogy(f_pre, p_pre, linewidth=2, label="pre (baseline)")
        plt.semilogy(f_post, p_post, linewidth=2, label="post (0–0.6s after onset)")
        plt.xlim(0, 60)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (uV^2/Hz)")
        plt.title("Power spectrum sanity (accepted example)")
        plt.legend()
        _savefig(os.path.join(args.out, "06_psd_pre_vs_post.png"))

    print(f"[qc_forward] wrote plots to: {args.out}")
    print(f"[qc_forward] seconds_per_sim: {sec_per_sim:.4f}")
    print("reject_counts:", counts)


if __name__ == "__main__":
    main()
