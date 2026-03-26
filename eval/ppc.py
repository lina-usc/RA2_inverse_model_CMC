from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def _find_repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    cands = [
        os.getcwd(),
        here,
        os.path.dirname(here),
        os.path.dirname(os.path.dirname(here)),
    ]
    seen = []
    for c in cands:
        c = os.path.abspath(c)
        if c not in seen:
            seen.append(c)
    for c in seen:
        if os.path.isdir(os.path.join(c, "sim")) and os.path.isdir(os.path.join(c, "data")):
            return c
    return os.getcwd()


BASE_DIR = _find_repo_root()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sim import simulate_eeg  # noqa: E402


def _set_plot_defaults() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _decode(arr):
    out = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _find_ch(ch_names, want="Cz"):
    for i, n in enumerate(ch_names):
        if n == want:
            return i
    return 0


def _tfr_1d(x, fs, fmin=0.5, fmax=40.0):
    f, t, Sxx = spectrogram(
        x,
        fs=fs,
        nperseg=128,
        noverlap=96,
        scaling="density",
        mode="psd",
    )
    m = (f >= fmin) & (f <= fmax)
    f = f[m]
    S = np.log10(Sxx[m, :] + 1e-12)
    return f, t, S


def _theta_to_params(theta, param_names):
    return {param_names[i]: float(theta[i]) for i in range(len(param_names))}


def _save_pair_figure(examples: List[Dict[str, Any]], cfg: Dict[str, Any], out_png: str) -> None:
    nrows = len(examples)
    fig, axes = plt.subplots(nrows, 4, figsize=(18.0, 4.6 * nrows), squeeze=False)

    for r, ex in enumerate(examples):
        ax1, ax2, ax3, ax4 = axes[r]
        t = ex["t"]
        ax1.plot(t, ex["x_obs"], linewidth=1.6, label="Observed")
        ax1.plot(t, ex["mean"], linewidth=1.6, label="PPC mean")
        ax1.fill_between(t, ex["lo"], ex["hi"], alpha=0.22, label="PPC 5–95%")
        ax1.axvline(cfg["stim_onset"], linestyle="--", linewidth=1.1, color="black")
        ax1.set_title(f"ERP @ {ex['channel']} (idx={ex['idx']})")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("uV")
        ax1.grid(True, alpha=0.25)
        if r == 0:
            ax1.legend(loc="upper right")

        vmin = ex["tfr_vmin"]
        vmax = ex["tfr_vmax"]
        dlim = ex["diff_lim"]

        im2 = ax2.imshow(
            ex["S_obs"], aspect="auto", origin="lower",
            extent=[ex["tt"].min(), ex["tt"].max(), ex["f_obs"].min(), ex["f_obs"].max()],
            vmin=vmin, vmax=vmax,
        )
        ax2.set_title("Observed spectrogram")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Hz")
        c2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
        c2.ax.tick_params(labelsize=8)

        im3 = ax3.imshow(
            ex["S_mean"], aspect="auto", origin="lower",
            extent=[ex["tt"].min(), ex["tt"].max(), ex["f_obs"].min(), ex["f_obs"].max()],
            vmin=vmin, vmax=vmax,
        )
        ax3.set_title("PPC mean spectrogram")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Hz")
        c3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)
        c3.ax.tick_params(labelsize=8)

        im4 = ax4.imshow(
            ex["S_mean"] - ex["S_obs"], aspect="auto", origin="lower",
            extent=[ex["tt"].min(), ex["tt"].max(), ex["f_obs"].min(), ex["f_obs"].max()],
            vmin=-dlim, vmax=dlim,
        )
        ax4.set_title("PPC − observed")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Hz")
        c4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.02)
        c4.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


def _save_individual_figure(ex: Dict[str, Any], cfg: Dict[str, Any], out_png: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0))
    ax1, ax2, ax3, ax4 = axes.reshape(-1)

    ax1.plot(ex["t"], ex["x_obs"], linewidth=1.6, label="Observed")
    ax1.plot(ex["t"], ex["mean"], linewidth=1.6, label="PPC mean")
    ax1.fill_between(ex["t"], ex["lo"], ex["hi"], alpha=0.22, label="PPC 5–95%")
    ax1.axvline(cfg["stim_onset"], linestyle="--", linewidth=1.1, color="black")
    ax1.set_title(f"ERP @ {ex['channel']} (idx={ex['idx']})")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("uV")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    im2 = ax2.imshow(
        ex["S_obs"], aspect="auto", origin="lower",
        extent=[ex["tt"].min(), ex["tt"].max(), ex["f_obs"].min(), ex["f_obs"].max()],
        vmin=ex["tfr_vmin"], vmax=ex["tfr_vmax"],
    )
    ax2.set_title("Observed spectrogram")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Hz")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    im3 = ax3.imshow(
        ex["S_mean"], aspect="auto", origin="lower",
        extent=[ex["tt"].min(), ex["tt"].max(), ex["f_obs"].min(), ex["f_obs"].max()],
        vmin=ex["tfr_vmin"], vmax=ex["tfr_vmax"],
    )
    ax3.set_title("PPC mean spectrogram")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Hz")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.02)

    im4 = ax4.imshow(
        ex["S_mean"] - ex["S_obs"], aspect="auto", origin="lower",
        extent=[ex["tt"].min(), ex["tt"].max(), ex["f_obs"].min(), ex["f_obs"].max()],
        vmin=-ex["diff_lim"], vmax=ex["diff_lim"],
    )
    ax4.set_title("PPC − observed")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Hz")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.replace(".png", ".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-npz", required=True, help="eval_test_outputs.npz (must contain theta_samples).")
    ap.add_argument("--h5", default=os.path.join(BASE_DIR, "data", "synthetic_cmc_dataset.h5"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", default="Cz")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--pair-size", type=int, default=2)
    ap.add_argument("--n-ppc-sims", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _set_plot_defaults()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ev = np.load(args.eval_npz, allow_pickle=True)
    theta_true = ev["theta_true"].astype(np.float32)
    theta_samples = ev["theta_samples"].astype(np.float32) if "theta_samples" in ev.files else ev["theta_post_samples"].astype(np.float32)
    param_names = _decode(ev["param_names"])

    if "eval_idx" in ev.files:
        eval_idx = ev["eval_idx"].astype(np.int64)
    else:
        splits = np.load(os.path.join(BASE_DIR, "data_out", "splits.npz"))
        eval_idx = splits["test_idx"].astype(np.int64)

    with h5py.File(args.h5, "r") as f:
        eeg = f["eeg"]
        theta_h5 = f["theta"]
        sim_seed = f["sim_seed"]
        ch_names = _decode(f["ch_names"][:])
        L = f["leadfield"][:].astype(np.float32)

        cfg = dict(
            fs=int(f.attrs["fs"]),
            duration=float(f.attrs["duration_sec"]),
            n_channels=int(f.attrs["n_channels"]),
            stim_onset=float(f.attrs["stim_onset_sec"]),
            stim_sigma=float(f.attrs["stim_sigma_sec"]),
            warmup_sec=float(f.attrs["warmup_sec"]),
            n_sources=int(f.attrs["n_sources"]),
            n_trials=int(f.attrs["n_trials"]),
            input_noise_std=float(f.attrs["input_noise_std"]),
            sensor_noise_std=float(f.attrs["sensor_noise_std"]),
            internal_fs=int(f.attrs["internal_fs"]),
            bandpass=(float(f.attrs["bandpass_lo_hz"]), float(f.attrs["bandpass_hi_hz"])),
            baseline_correct=bool(int(f.attrs["baseline_correct"])),
            downsample_method=str(f.attrs["downsample_method"]),
            uV_scale=float(f.attrs["uV_scale"]),
        )

        max_diff = float(np.max(np.abs(theta_h5[eval_idx] - theta_true)))
        if max_diff > 1e-6:
            raise SystemExit(
                f"Eval/H5 alignment failed (max |theta| diff={max_diff}).\n"
                "Fix: ensure evaluate_ensemble saves eval_idx."
            )

        ch_i = _find_ch(ch_names, args.channel)
        T = eeg.shape[-1]
        t = np.arange(T, dtype=np.float32) / cfg["fs"]
        n_eval = len(eval_idx)
        picks = rng.choice(n_eval, size=min(args.n_examples, n_eval), replace=False)

        examples: List[Dict[str, Any]] = []
        for k, j in enumerate(picks):
            idx = int(eval_idx[j])
            eeg_obs = eeg[idx].astype(np.float32)
            x_obs = eeg_obs[ch_i]

            S = theta_samples.shape[1]
            sel = rng.choice(S, size=min(args.n_ppc_sims, S), replace=False)
            thetas = theta_samples[j, sel, :]

            sims = []
            base_seed = int(sim_seed[idx])
            for s_i, th in enumerate(thetas):
                p = _theta_to_params(th, param_names)
                sim = simulate_eeg(
                    params=p,
                    fs=cfg["fs"],
                    duration=cfg["duration"],
                    n_channels=cfg["n_channels"],
                    seed=base_seed + 1000 * (s_i + 1),
                    bandpass=cfg["bandpass"],
                    stim_onset=cfg["stim_onset"],
                    stim_sigma=cfg["stim_sigma"],
                    n_sources=cfg["n_sources"],
                    leadfield=L,
                    sensor_noise_std=cfg["sensor_noise_std"],
                    n_trials=cfg["n_trials"],
                    input_noise_std=cfg["input_noise_std"],
                    internal_fs=cfg["internal_fs"],
                    baseline_correct=cfg["baseline_correct"],
                    baseline_window=None,
                    warmup_sec=cfg["warmup_sec"],
                    downsample_method=cfg["downsample_method"],
                    uV_scale=cfg["uV_scale"],
                )
                sims.append(sim.astype(np.float32))
            sims = np.stack(sims, axis=0)
            x_sim = sims[:, ch_i, :]

            mean = x_sim.mean(axis=0)
            lo = np.percentile(x_sim, 5, axis=0)
            hi = np.percentile(x_sim, 95, axis=0)

            f_obs, tt, S_obs = _tfr_1d(x_obs, fs=cfg["fs"], fmin=cfg["bandpass"][0], fmax=cfg["bandpass"][1])
            S_sim = np.stack([
                _tfr_1d(xx, fs=cfg["fs"], fmin=cfg["bandpass"][0], fmax=cfg["bandpass"][1])[2]
                for xx in x_sim
            ], axis=0)
            S_mean = S_sim.mean(axis=0)
            tfr_vmin = float(min(S_obs.min(), S_mean.min()))
            tfr_vmax = float(max(S_obs.max(), S_mean.max()))
            diff_lim = float(max(1e-6, np.max(np.abs(S_mean - S_obs))))

            ex = {
                "idx": idx,
                "channel": ch_names[ch_i],
                "t": t,
                "x_obs": x_obs,
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "f_obs": f_obs,
                "tt": tt,
                "S_obs": S_obs,
                "S_mean": S_mean,
                "tfr_vmin": tfr_vmin,
                "tfr_vmax": tfr_vmax,
                "diff_lim": diff_lim,
            }
            examples.append(ex)
            out_single = os.path.join(args.out, f"ppc_{k:02d}_idx{idx}.png")
            _save_individual_figure(ex, cfg, out_single)

    pair_files = []
    pair_size = max(1, int(args.pair_size))
    for pi, start in enumerate(range(0, len(examples), pair_size)):
        subset = examples[start : start + pair_size]
        out_pair = os.path.join(args.out, f"ppc_pair_{pi:02d}.png")
        _save_pair_figure(subset, cfg, out_pair)
        pair_files.append(os.path.basename(out_pair))

    manifest = {
        "eval_npz": args.eval_npz,
        "h5": args.h5,
        "n_examples": len(examples),
        "pair_size": pair_size,
        "pair_files": pair_files,
        "example_indices": [int(ex["idx"]) for ex in examples],
    }
    with open(os.path.join(args.out, "ppc_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.out, "ppc_forward_cfg.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("[ppc] wrote individual and paired PPC figures to:", args.out)


if __name__ == "__main__":
    main()
