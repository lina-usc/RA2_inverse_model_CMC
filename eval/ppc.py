
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.signal import spectrogram  # noqa: E402

from sim import simulate_eeg


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
    # Simple spectrogram for PPC plots (not token-exact; good for qualitative checks)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-npz", required=True, help="eval_test_outputs.npz (must contain theta_samples).")
    ap.add_argument("--h5", default="data/synthetic_cmc_dataset.h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--channel", default="Cz")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--n-ppc-sims", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ev = np.load(args.eval_npz, allow_pickle=True)
    if "theta_samples" not in ev.files:
        raise SystemExit(f"theta_samples not found in {args.eval_npz}. keys={ev.files}")

    theta_true = ev["theta_true"].astype(np.float32)          # (N,P)
    theta_samples = ev["theta_samples"].astype(np.float32)    # (N,S,P)
    param_names = _decode(ev["param_names"])

    # Use eval_idx if present; else assume standard test split order
    if "eval_idx" in ev.files:
        eval_idx = ev["eval_idx"].astype(np.int64)
    else:
        splits = np.load("data_out/splits.npz")
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

        # sanity: alignment (eval ordering == H5 indexing)
        max_diff = float(np.max(np.abs(theta_h5[eval_idx] - theta_true)))
        if max_diff > 1e-6:
            raise SystemExit(
                f"Eval/H5 alignment failed (max |theta| diff={max_diff}).\n"
                "Fix: ensure evaluate_ensemble saves eval_idx."
            )

        ch_i = _find_ch(ch_names, args.channel)
        T = eeg.shape[-1]
        t = np.arange(T) / cfg["fs"]

        n_eval = len(eval_idx)
        picks = rng.choice(n_eval, size=min(args.n_examples, n_eval), replace=False)

        for k, j in enumerate(picks):
            idx = int(eval_idx[j])
            eeg_obs = eeg[idx].astype(np.float32)  # (C,T)
            x_obs = eeg_obs[ch_i]

            # choose posterior draws
            S = theta_samples.shape[1]
            sel = rng.choice(S, size=min(args.n_ppc_sims, S), replace=False)
            thetas = theta_samples[j, sel, :]  # (K,P)

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
            sims = np.stack(sims, axis=0)  # (K,C,T)
            x_sim = sims[:, ch_i, :]

            mean = x_sim.mean(axis=0)
            lo = np.percentile(x_sim, 5, axis=0)
            hi = np.percentile(x_sim, 95, axis=0)

            f_obs, tt, S_obs = _tfr_1d(x_obs, fs=cfg["fs"], fmin=cfg["bandpass"][0], fmax=cfg["bandpass"][1])
            S_sim = np.stack([_tfr_1d(xx, fs=cfg["fs"], fmin=cfg["bandpass"][0], fmax=cfg["bandpass"][1])[2] for xx in x_sim], axis=0)
            S_mean = S_sim.mean(axis=0)

            fig = plt.figure(figsize=(12, 7))

            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(t, x_obs, label="observed")
            ax1.plot(t, mean, label="ppc mean")
            ax1.fill_between(t, lo, hi, alpha=0.2, label="ppc 5–95%")
            ax1.axvline(cfg["stim_onset"], linestyle="--", linewidth=1)
            ax1.set_title(f"ERP @ {ch_names[ch_i]} (idx={idx})")
            ax1.set_xlabel("time (s)")
            ax1.set_ylabel("uV")
            ax1.legend(fontsize=8)

            ax2 = plt.subplot(2, 2, 2)
            im2 = ax2.imshow(S_obs, aspect="auto", origin="lower",
                             extent=[tt.min(), tt.max(), f_obs.min(), f_obs.max()])
            ax2.set_title("Observed TFR (log10 PSD)")
            ax2.set_xlabel("time (s)")
            ax2.set_ylabel("Hz")
            plt.colorbar(im2, ax=ax2, fraction=0.046)

            ax3 = plt.subplot(2, 2, 3)
            im3 = ax3.imshow(S_mean, aspect="auto", origin="lower",
                             extent=[tt.min(), tt.max(), f_obs.min(), f_obs.max()])
            ax3.set_title("PPC mean TFR (log10 PSD)")
            ax3.set_xlabel("time (s)")
            ax3.set_ylabel("Hz")
            plt.colorbar(im3, ax=ax3, fraction=0.046)

            ax4 = plt.subplot(2, 2, 4)
            im4 = ax4.imshow(S_mean - S_obs, aspect="auto", origin="lower",
                             extent=[tt.min(), tt.max(), f_obs.min(), f_obs.max()])
            ax4.set_title("PPC − Observed (log10 PSD)")
            ax4.set_xlabel("time (s)")
            ax4.set_ylabel("Hz")
            plt.colorbar(im4, ax=ax4, fraction=0.046)

            plt.tight_layout()
            out_png = os.path.join(args.out, f"ppc_{k:02d}_idx{idx}.png")
            plt.savefig(out_png, dpi=200)
            plt.close(fig)

    with open(os.path.join(args.out, "ppc_forward_cfg.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("[ppc] wrote:", args.out)


if __name__ == "__main__":
    main()
