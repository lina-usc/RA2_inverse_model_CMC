from __future__ import annotations
import argparse, os, json, importlib, inspect, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.feature_tokens import TokenConfig, compute_erp_tokens, compute_tfr_tokens  # noqa

def import_callable(spec: str):
    if ":" not in spec:
        raise ValueError("CMC_SIMULATOR must look like 'module.submodule:function'")
    mod, fn = spec.split(":", 1)
    m = importlib.import_module(mod)
    f = getattr(m, fn)
    if not callable(f):
        raise TypeError(f"{spec} is not callable")
    return f

def relpath_to_module(p: Path) -> str:
    rel = p.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)

def autodetect_simulate_eeg():
    # Search sim/ first (highest priority), then whole repo.
    search_roots = []
    if (REPO_ROOT / "sim").exists():
        search_roots.append(REPO_ROOT / "sim")
    search_roots.append(REPO_ROOT)

    candidates: list[Path] = []
    pat = re.compile(r"def\s+simulate_eeg\s*\(")

    seen = set()
    for root in search_roots:
        for p in root.rglob("*.py"):
            if p in seen:
                continue
            seen.add(p)
            try:
                txt = p.read_text(errors="ignore")
            except Exception:
                continue
            if pat.search(txt):
                candidates.append(p)

    if not candidates:
        raise RuntimeError(
            "Could not auto-detect simulate_eeg(). Set CMC_SIMULATOR to the correct module:function.\n"
            "Try: python -c \"import sim; print(dir(sim))\""
        )

    # Prefer sim/* and shorter module paths.
    def score(p: Path):
        rel = str(p.relative_to(REPO_ROOT))
        s = 0
        if rel.startswith("sim/"):
            s -= 100
        s += len(rel)
        return s

    candidates.sort(key=score)

    last_err = None
    for p in candidates:
        modname = relpath_to_module(p)
        if not modname:
            continue
        try:
            m = importlib.import_module(modname)
            fn = getattr(m, "simulate_eeg", None)
            if callable(fn):
                return fn, f"{modname}:simulate_eeg"
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Found simulate_eeg definitions but could not import them. Last error: {repr(last_err)}")

def resolve_simulator():
    spec = os.environ.get("CMC_SIMULATOR", "").strip()
    if spec:
        try:
            fn = import_callable(spec)
            return fn, spec
        except Exception as e:
            print(f"[WARN] CMC_SIMULATOR='{spec}' failed ({repr(e)}). Falling back to auto-detect.")
    fn, spec2 = autodetect_simulate_eeg()
    return fn, spec2

def call_sim(sim_fn, theta: np.ndarray, seed: int):
    # Try a few common calling conventions
    try:
        return sim_fn(theta)
    except Exception:
        pass
    try:
        return sim_fn(theta, seed=seed)
    except Exception:
        pass
    try:
        rng = np.random.default_rng(seed)
        return sim_fn(theta, rng=rng)
    except Exception:
        pass
    try:
        return sim_fn(theta, random_state=seed)
    except Exception as e:
        raise RuntimeError(
            "Could not call simulator. Tried sim(theta), sim(theta, seed=?), sim(theta, rng=?), sim(theta, random_state=?). "
            f"Last error: {repr(e)}"
        )

def normalize_sim_output(out) -> np.ndarray:
    if isinstance(out, (tuple, list)):
        out = out[0]
    eeg = np.asarray(out, dtype=np.float32)
    if eeg.ndim != 2:
        raise ValueError(f"Simulator output must be 2D, got shape {eeg.shape}")
    if eeg.shape[0] == 16 and eeg.shape[1] >= 500:
        eeg = eeg[:, :500]
    elif eeg.shape[1] == 16 and eeg.shape[0] >= 500:
        eeg = eeg[:500, :].T
    else:
        raise ValueError(f"Simulator output shape not understood: {eeg.shape} (expected (16,500) or (500,16))")
    return eeg.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snpe-run", type=str, required=True, help="e.g., results/neurips_sbi/snpe_hybrid_t25_f15")
    ap.add_argument("--splits", type=str, default="data_out/splits.npz")
    ap.add_argument("--outdir", type=str, default="plots/ppc_snpe")
    ap.add_argument("--indices", type=str, default="2623,6528,5086,2944,417,8532")
    ap.add_argument("--n-ppc-sims", type=int, default=50)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    snpe_run = Path(args.snpe_run)
    cache = np.load(snpe_run / "cache_data.npz", allow_pickle=True)
    meta = json.loads(str(cache["meta_json"]))
    x_test = cache["x_test"].astype(np.float32)
    eeg_test = cache["eeg_test"].astype(np.float32)

    splits = np.load(args.splits, allow_pickle=True)
    test_idx = splits["test_idx"].astype(int)

    train_meta = json.loads((snpe_run / "train_meta.json").read_text())
    mu = np.array(train_meta["x_zscore_mu"], dtype=np.float32)
    sd = np.array(train_meta["x_zscore_sd"], dtype=np.float32)

    import torch
    try:
        posterior = torch.load(snpe_run / "posterior_snpe.pt", map_location=args.device, weights_only=False)
    except TypeError:
        posterior = torch.load(snpe_run / "posterior_snpe.pt", map_location=args.device)

    sim_fn, sim_spec = resolve_simulator()
    print("[ppc] using simulator:", sim_spec)

    cfg = TokenConfig(
        n_time_patches=int(meta["n_time_patches"]),
        n_freq_patches=int(meta["n_freq_patches"]),
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wanted = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    def flat(z): return z.reshape(z.shape[0], -1).astype(np.float32)
    def zscore(x): return ((x - mu[None, :]) / sd[None, :]).astype(np.float32)

    for g in wanted:
        pos = np.where(test_idx == g)[0]
        if len(pos) == 0:
            print(f"[WARN] global idx {g} not in test split; skipping")
            continue
        j = int(pos[0])

        x_obs = x_test[j]       # (tokens,16)
        eeg_obs = eeg_test[j]   # (16,500)

        x_flat = zscore(flat(x_obs[None, ...]))
        x_t = torch.tensor(x_flat, dtype=torch.float32, device=args.device)

        theta_samps = posterior.sample((args.n_ppc_sims,), x=x_t).detach().cpu().numpy().astype(np.float32)
        if theta_samps.ndim == 3:
            theta_samps = theta_samps[:, 0, :]

        erp_sims = []
        tfr_sims = []
        for k in range(args.n_ppc_sims):
            out = call_sim(sim_fn, theta_samps[k], seed=int(rng.integers(0, 2**31 - 1)))
            eeg_sim = normalize_sim_output(out)

            erp = compute_erp_tokens(eeg_sim, cfg)  # (T,16)
            tfr_tok, _ = compute_tfr_tokens(eeg_sim, cfg)  # (T*F,16)
            tfr = tfr_tok.reshape(cfg.n_time_patches, cfg.n_freq_patches, cfg.n_channels)

            erp_sims.append(erp)
            tfr_sims.append(tfr)

        erp_sims = np.stack(erp_sims, axis=0)  # (S,T,16)
        tfr_sims = np.stack(tfr_sims, axis=0)  # (S,T,F,16)

        erp_obs = compute_erp_tokens(eeg_obs, cfg)
        tfr_obs_tok, _ = compute_tfr_tokens(eeg_obs, cfg)
        tfr_obs = tfr_obs_tok.reshape(cfg.n_time_patches, cfg.n_freq_patches, cfg.n_channels)

        # channel-averaged summaries for visualization
        erp_obs_m = erp_obs.mean(axis=1)                # (T,)
        erp_sim_chmean = erp_sims.mean(axis=2)          # (S,T)
        erp_mean = erp_sim_chmean.mean(axis=0)          # (T,)
        erp_lo = np.quantile(erp_sim_chmean, 0.05, axis=0)
        erp_hi = np.quantile(erp_sim_chmean, 0.95, axis=0)

        tfr_obs_m = tfr_obs.mean(axis=2)                # (T,F)
        tfr_mean = tfr_sims.mean(axis=0).mean(axis=2)   # (T,F)

        T = cfg.n_time_patches
        t_centers = (np.arange(T) + 0.5) * (cfg.duration / T)

        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])

        ax0 = fig.add_subplot(gs[0, 0])
        im0 = ax0.imshow(tfr_obs_m.T, origin="lower", aspect="auto")
        ax0.set_title("Observed TFR (ch-avg, dB)")
        ax0.set_xlabel("time patch")
        ax0.set_ylabel("freq patch")
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(tfr_mean.T, origin="lower", aspect="auto")
        ax1.set_title("SNPE PPC mean TFR (ch-avg, dB)")
        ax1.set_xlabel("time patch")
        ax1.set_ylabel("freq patch")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(t_centers, erp_obs_m, label="Observed ERP (ch-avg)")
        ax2.plot(t_centers, erp_mean, label="SNPE PPC mean ERP")
        ax2.fill_between(t_centers, erp_lo, erp_hi, alpha=0.25, label="SNPE PPC 90% band")
        ax2.axvline(cfg.stim_onset, linestyle="--")
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("ERP token (a.u.)")
        ax2.set_title(f"SNPE posterior predictive check (global idx {g})")
        ax2.legend(loc="best", fontsize=9)

        fig.tight_layout()
        out = outdir / f"ppc_snpe_idx{g}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("[ppc] wrote", out)

if __name__ == "__main__":
    main()
