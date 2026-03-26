from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import asdict

import h5py
import numpy as np

from data.feature_tokens import TokenConfig
from data.priors import build_prior_spec, sample_theta
from sim import simulate_eeg
from sim.leadfield_mne import DEFAULT_16_CH_NAMES, make_leadfield
from sim.regime_filter import RegimeFilterConfig, regime_reject


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"ERROR: {msg}")


def _theta_from_params_dict(params: dict[str, float], names: list[str]) -> np.ndarray:
    theta = np.asarray([float(params[n]) for n in names], dtype=np.float32)
    if theta.ndim != 1:
        raise SystemExit(f"ERROR: theta must be rank-1, got shape={theta.shape}")
    if not np.all(np.isfinite(theta)):
        raise SystemExit("ERROR: theta contains non-finite values")
    return theta


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

    print(f"[generate_dataset] {name} shape={arr.shape} min={mn:.6g} max={mx:.6g}")
    print(f"[generate_dataset] {name} std per param={std}")

    if abs(mx - mn) < 1e-12 or np.all(std < 1e-12):
        raise SystemExit(
            f"ERROR: degenerate {name}: values are constant or zero-variance. "
            f"min={mn} max={mx} std={std}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--out", type=str, default="data/synthetic_cmc_dataset.h5")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--leadfield-seed", type=int, default=0)
    ap.add_argument("--n-sources", type=int, default=3)
    ap.add_argument("--n-trials", type=int, default=10)
    ap.add_argument("--internal-fs", type=int, default=1000)

    ap.add_argument("--input-noise-std", type=float, default=0.2)
    ap.add_argument("--sensor-noise-std", type=float, default=2.0)
    ap.add_argument("--stim-sigma", type=float, default=0.05)
    ap.add_argument("--warmup-sec", type=float, default=3.0)
    ap.add_argument("--uV-scale", type=float, default=100.0)

    ap.add_argument("--bandpass-lo", type=float, default=0.5)
    ap.add_argument("--bandpass-hi", type=float, default=40.0)
    ap.add_argument("--downsample-method", choices=["slice", "poly"], default="slice")

    ap.add_argument("--baseline-correct", dest="baseline_correct", action="store_true")
    ap.add_argument("--no-baseline-correct", dest="baseline_correct", action="store_false")
    ap.set_defaults(baseline_correct=True)

    ap.add_argument("--stim-causal", dest="stim_causal", action="store_true")
    ap.add_argument("--stim-noncausal", dest="stim_causal", action="store_false")
    ap.set_defaults(stim_causal=True)

    ap.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="0 means auto=50*n accepted-target attempts cap.",
    )

    args = ap.parse_args()

    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"ERROR: {args.out} exists. Use --overwrite.")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Use the same default config that the rest of the repo uses.
    cfg = TokenConfig()
    fs = int(cfg.fs)
    duration = float(cfg.duration)
    n_channels = int(cfg.n_channels)
    stim_onset = float(cfg.stim_onset)
    n_time = int(round(fs * duration))

    rf_cfg = RegimeFilterConfig(fs=fs, duration=duration, stim_onset=stim_onset)

    L, _info, meta, _ch_pos, _src_pos, _src_ori = make_leadfield(
        fs=fs,
        n_sources=int(args.n_sources),
        ch_names=list(DEFAULT_16_CH_NAMES[:n_channels]),
        seed=int(args.leadfield_seed),
    )

    names, low, high, dist, prior_params, prior_json = build_prior_spec()
    names = [str(x) for x in names]
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    prior_params = np.asarray(prior_params, dtype=np.float32)
    prior_dist = np.asarray(dist, dtype="S")
    P = len(names)

    _require(low.shape == (P,), f"prior_low shape mismatch: {low.shape} vs {(P,)}")
    _require(high.shape == (P,), f"prior_high shape mismatch: {high.shape} vs {(P,)}")

    max_attempts = int(args.max_attempts) if int(args.max_attempts) > 0 else int(50 * args.n)
    rng = np.random.default_rng(args.seed)
    counts = Counter()

    t0 = time.perf_counter()

    with h5py.File(args.out, "w") as f:
        d_eeg = f.create_dataset("eeg", shape=(int(args.n), n_channels, n_time), dtype="float32")
        d_theta = f.create_dataset("theta", shape=(int(args.n), P), dtype="float32")
        d_seed = f.create_dataset("sim_seed", shape=(int(args.n),), dtype="int64")

        f.create_dataset("ch_names", data=np.asarray(DEFAULT_16_CH_NAMES[:n_channels], dtype="S"))
        f.create_dataset("leadfield", data=np.asarray(L, dtype=np.float32))
        f.create_dataset("leadfield_meta_json", data=np.bytes_(json.dumps(asdict(meta))))

        f.create_dataset("param_names", data=np.asarray(names, dtype="S"))
        f.create_dataset("prior_low", data=low)
        f.create_dataset("prior_high", data=high)
        f.create_dataset("prior_dist", data=prior_dist)
        f.create_dataset("prior_params", data=prior_params)
        f.create_dataset("prior_spec_json", data=np.bytes_(str(prior_json)))

        f.attrs["fs"] = int(fs)
        f.attrs["duration_sec"] = float(duration)
        f.attrs["n_channels"] = int(n_channels)
        f.attrs["stim_onset_sec"] = float(stim_onset)
        f.attrs["stim_sigma_sec"] = float(args.stim_sigma)
        f.attrs["warmup_sec"] = float(args.warmup_sec)
        f.attrs["bandpass_lo_hz"] = float(args.bandpass_lo)
        f.attrs["bandpass_hi_hz"] = float(args.bandpass_hi)
        f.attrs["n_sources"] = int(args.n_sources)
        f.attrs["n_trials"] = int(args.n_trials)
        f.attrs["input_noise_std"] = float(args.input_noise_std)
        f.attrs["sensor_noise_std"] = float(args.sensor_noise_std)
        f.attrs["internal_fs"] = int(args.internal_fs)
        f.attrs["baseline_correct"] = int(bool(args.baseline_correct))
        f.attrs["downsample_method"] = str(args.downsample_method)
        f.attrs["uV_scale"] = float(args.uV_scale)
        f.attrs["stim_causal"] = int(bool(args.stim_causal))
        f.attrs["generator_seed"] = int(args.seed)
        f.attrs["leadfield_seed"] = int(args.leadfield_seed)

        n_written = 0
        n_attempts = 0

        while n_written < int(args.n):
            if n_attempts >= max_attempts:
                raise SystemExit(
                    f"ERROR: reached max_attempts={max_attempts} with only {n_written}/{args.n} accepted samples. "
                    f"Reject counts so far: {dict(counts)}"
                )

            n_attempts += 1

            # sample_theta may return a broken first output in your repo.
            # We therefore rebuild theta from the simulator params dict in canonical order.
            _theta_unused, params = sample_theta(rng, names, low, high, dist, prior_params)
            theta_vec = _theta_from_params_dict(params, names)

            sim_seed = int(rng.integers(0, np.iinfo(np.int32).max))

            eeg = simulate_eeg(
                params=params,
                fs=fs,
                duration=duration,
                n_channels=n_channels,
                seed=sim_seed,
                bandpass=(float(args.bandpass_lo), float(args.bandpass_hi)),
                stim_onset=stim_onset,
                stim_sigma=float(args.stim_sigma),
                stim_causal=bool(args.stim_causal),
                n_sources=int(args.n_sources),
                leadfield=L,
                sensor_noise_std=float(args.sensor_noise_std),
                n_trials=int(args.n_trials),
                input_noise_std=float(args.input_noise_std),
                internal_fs=int(args.internal_fs),
                baseline_correct=bool(args.baseline_correct),
                baseline_window=None,
                warmup_sec=float(args.warmup_sec),
                downsample_method=str(args.downsample_method),
                uV_scale=float(args.uV_scale),
            )

            ok, reason = regime_reject(eeg, rf_cfg)
            counts[str(reason)] += 1
            if not ok:
                continue

            d_eeg[n_written] = np.asarray(eeg, dtype=np.float32)
            d_theta[n_written] = theta_vec
            d_seed[n_written] = sim_seed
            n_written += 1

            if n_written % 500 == 0 or n_written == int(args.n):
                probe = np.asarray(d_theta[: min(n_written, 2048)], dtype=np.float32)
                print(
                    f"[generate_dataset] accepted {n_written}/{args.n} "
                    f"after {n_attempts} attempts | accept_rate={n_written / max(1, n_attempts):.3f}"
                )
                print(f"[generate_dataset] theta probe std={probe.std(axis=0)}")
                f.flush()

        f.attrs["n_attempts_total"] = int(n_attempts)
        f.attrs["accept_rate"] = float(n_written / max(1, n_attempts))
        f.attrs["reject_counts_json"] = np.bytes_(json.dumps(dict(counts)))
        f.flush()

    with h5py.File(args.out, "r") as f:
        th = np.asarray(f["theta"][: min(int(args.n), 2048)], dtype=np.float32)
        _check_theta_probe("final H5 theta", th)

    t1 = time.perf_counter()
    print(f"[generate_dataset] wrote: {args.out}")
    print(f"[generate_dataset] n={args.n} attempts={n_attempts} accept_rate={args.n / max(1, n_attempts):.3f}")
    print(f"[generate_dataset] total_sec={t1 - t0:.2f} sec_per_accept={(t1 - t0) / max(1, args.n):.4f}")


if __name__ == "__main__":
    main()
