from __future__ import annotations

import argparse
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _savefig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _summ(x: np.ndarray, name: str) -> None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    frac0 = float(np.mean(x == 0.0))
    frac_small = float(np.mean(np.abs(x) < 1e-6))
    qs = np.quantile(x, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    print(
        f"[qc_tokens] {name}: mean={x.mean():.4g} std={x.std():.4g} "
        f"min={x.min():.4g} max={x.max():.4g} frac(x==0)={frac0:.3f} frac(|x|<1e-6)={frac_small:.3f} "
        f"q=[{', '.join(f'{v:.4g}' for v in qs)}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-out", type=str, default="data_out")
    ap.add_argument("--out", type=str, default="plots/qc_tokens")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    meta = np.load(os.path.join(args.data_out, "tfr_meta.npz"))
    ch_names = [s.decode("utf-8") for s in meta["ch_names"]]
    stim_onset = float(meta["stim_onset"])

    n_time_patches = int(meta["n_time_patches"])
    n_freq_patches = int(meta["n_freq_patches"])
    n_tokens_erp = int(meta["n_tokens_erp"])
    n_tokens_total = int(meta["n_tokens_total"])

    time_edges = meta["time_edges"].astype(np.float32)
    # patch index that CONTAINS stim_onset
    stim_patch = int(np.searchsorted(time_edges, stim_onset, side="right") - 1)
    stim_patch = int(np.clip(stim_patch, 0, n_time_patches - 1))

    X = np.load(os.path.join(args.data_out, "features.npy"), mmap_mode="r")  # hybrid tokens
    N = int(X.shape[0])

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, size=min(args.n, N), replace=False)

    # pick a channel to visualize
    ch_pick = "Cz" if "Cz" in ch_names else ch_names[len(ch_names) // 2]
    ch_i = ch_names.index(ch_pick)

    erp = np.asarray(X[idx, :n_tokens_erp, :], dtype=np.float32)  # (n, time_patches, ch)
    tfr = np.asarray(X[idx, n_tokens_erp:n_tokens_total, :], dtype=np.float32)  # (n, 375, ch)
    tfr = tfr.reshape(len(idx), n_time_patches, n_freq_patches, len(ch_names))  # (n, time, freq, ch)

    _summ(erp, "ERP tokens (subset)")
    _summ(tfr, "TFR tokens (subset)")

    # --- plot 1: ERP token heatmap (mean over subset) ---
    erp_mean = erp.mean(axis=0)  # (time_patches, ch)
    plt.figure(figsize=(10, 4))
    plt.imshow(erp_mean.T, aspect="auto", origin="lower")
    plt.colorbar(label="ERP token mean")
    plt.yticks(np.arange(len(ch_names)), ch_names, fontsize=8)
    plt.xticks(np.arange(n_time_patches), [f"{time_edges[i]:.2f}" for i in range(n_time_patches)], rotation=90, fontsize=7)
    plt.title("ERP tokens (mean over subset): channels × time_patches")
    plt.xlabel("time patch start (s)")
    plt.ylabel("channel")
    plt.axvline(stim_patch, linestyle="--", linewidth=2)
    _savefig(os.path.join(args.out, "00_erp_tokens_mean_heatmap.png"))

    # --- plot 1b: ERP token heatmap (std over subset) ---
    erp_std = erp.std(axis=0)
    plt.figure(figsize=(10, 4))
    plt.imshow(erp_std.T, aspect="auto", origin="lower")
    plt.colorbar(label="ERP token std")
    plt.yticks(np.arange(len(ch_names)), ch_names, fontsize=8)
    plt.xticks(np.arange(n_time_patches), [f"{time_edges[i]:.2f}" for i in range(n_time_patches)], rotation=90, fontsize=7)
    plt.title("ERP tokens (std over subset): channels × time_patches")
    plt.xlabel("time patch start (s)")
    plt.ylabel("channel")
    plt.axvline(stim_patch, linestyle="--", linewidth=2)
    _savefig(os.path.join(args.out, "00b_erp_tokens_std_heatmap.png"))

    # --- plot 2: TFR token heatmap for one channel (mean over subset) ---
    tfr_mean = tfr.mean(axis=0)  # (time, freq, ch)
    plt.figure(figsize=(8, 5))
    img = tfr_mean[:, :, ch_i].T  # (freq, time)
    plt.imshow(img, aspect="auto", origin="lower")
    plt.colorbar(label="TFR token mean")
    plt.title(f"TFR tokens (mean over subset) | channel={ch_pick}")
    plt.xlabel("time patch")
    plt.ylabel("freq patch")
    _savefig(os.path.join(args.out, "01_tfr_tokens_mean_heatmap_channel.png"))

    # --- plot 2b: TFR token heatmap for one channel (std over subset) ---
    tfr_std = tfr.std(axis=0)
    plt.figure(figsize=(8, 5))
    img = tfr_std[:, :, ch_i].T
    plt.imshow(img, aspect="auto", origin="lower")
    plt.colorbar(label="TFR token std")
    plt.title(f"TFR tokens (std over subset) | channel={ch_pick}")
    plt.xlabel("time patch")
    plt.ylabel("freq patch")
    _savefig(os.path.join(args.out, "01b_tfr_tokens_std_heatmap_channel.png"))

    # --- plot 3: token value distributions (ERP vs TFR) ---
    erp_vals = erp.reshape(-1)
    tfr_vals = tfr.reshape(-1)
    plt.figure(figsize=(8, 4))
    plt.hist(erp_vals, bins=80, alpha=0.6, density=True, label="ERP tokens")
    plt.hist(tfr_vals, bins=80, alpha=0.6, density=True, label="TFR tokens")
    plt.title("Token value distribution (subset)")
    plt.xlabel("token value")
    plt.ylabel("density")
    plt.legend()
    _savefig(os.path.join(args.out, "02_token_value_hist.png"))

    print(f"[qc_tokens] wrote plots to: {args.out}")
    print(f"[qc_tokens] N={N} | subset={len(idx)} | tokens_total={n_tokens_total} | ch_pick={ch_pick} | stim_patch={stim_patch}")


if __name__ == "__main__":
    main()