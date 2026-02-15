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
    fs = int(meta["fs"])
    duration = float(meta["duration"])
    stim_onset = float(meta["stim_onset"])
    n_time = int(round(duration * fs))

    n_time_patches = int(meta["n_time_patches"])
    n_freq_patches = int(meta["n_freq_patches"])
    n_tokens_erp = int(meta["n_tokens_erp"])
    n_tokens_total = int(meta["n_tokens_total"])

    time_edges = meta["time_edges"].astype(np.float32)
    freq_edges = meta["freq_edges"].astype(np.float32)

    X = np.load(os.path.join(args.data_out, "features.npy"), mmap_mode="r")  # hybrid tokens
    N = int(X.shape[0])

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(N, size=min(args.n, N), replace=False)

    # pick a channel to visualize
    ch_pick = "Cz" if "Cz" in ch_names else ch_names[len(ch_names) // 2]
    ch_i = ch_names.index(ch_pick)

    # --- plot 1: ERP token heatmap (mean over subset) ---
    erp = np.asarray(X[idx, :n_tokens_erp, :], dtype=np.float32)  # (n, time_patches, ch)
    erp_mean = erp.mean(axis=0)  # (time_patches, ch)

    plt.figure(figsize=(10, 4))
    plt.imshow(erp_mean.T, aspect="auto", origin="lower")
    plt.colorbar(label="ERP token mean (uV)")
    plt.yticks(np.arange(len(ch_names)), ch_names, fontsize=8)
    plt.xticks(np.arange(n_time_patches), [f"{time_edges[i]:.2f}" for i in range(n_time_patches)], rotation=90, fontsize=7)
    plt.title("ERP tokens (mean over subset): channels × time_patches")
    plt.xlabel("time patch start (s)")
    plt.ylabel("channel")
    plt.axvline(int(round(stim_onset / (duration / n_time_patches))), linestyle="--", linewidth=2)
    _savefig(os.path.join(args.out, "00_erp_tokens_mean_heatmap.png"))

    # --- plot 2: TFR token heatmap for one channel (mean over subset) ---
    tfr = np.asarray(X[idx, n_tokens_erp:n_tokens_total, :], dtype=np.float32)  # (n, 375, ch)
    tfr = tfr.reshape(len(idx), n_time_patches, n_freq_patches, len(ch_names))  # (n, time, freq, ch)
    tfr_mean = tfr.mean(axis=0)  # (time, freq, ch)

    plt.figure(figsize=(8, 5))
    img = tfr_mean[:, :, ch_i].T  # (freq, time)
    plt.imshow(img, aspect="auto", origin="lower")
    plt.colorbar(label="TFR token mean (dB rel baseline)")
    plt.title(f"TFR tokens (mean over subset) | channel={ch_pick}")
    plt.xlabel("time patch")
    plt.ylabel("freq patch")
    _savefig(os.path.join(args.out, "01_tfr_tokens_mean_heatmap_channel.png"))

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
    print(f"[qc_tokens] N={N} | subset={len(idx)} | tokens_total={n_tokens_total} | ch_pick={ch_pick}")


if __name__ == "__main__":
    main()
