from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from data.feature_tokens import TokenConfig
from sim.leadfield_mne import DEFAULT_16_CH_NAMES

from sensitivity.common import (
    array_sha1,
    check_cache_manifest,
    decode_bytes_array,
    ensure_parent,
    load_config,
    long_table_to_csv,
    save_manifest,
    save_theta_csv,
    sensitivity_root,
    validate_project_parameter_spec,
)
from sensitivity.forward_wrapper import ForwardSettings, forward_settings_from_config, evaluate_theta_matrix
from sensitivity.plotting import (
    plot_acceptance_morris,
    plot_erp_heatmaps,
    plot_morris_family_summary,
    plot_tfr_heatmaps,
)
from sensitivity.sampling import MorrisDesign, MorrisStats, compute_morris_statistics, generate_morris_design



def _token_cfg_from_settings(settings: ForwardSettings) -> TokenConfig:
    return settings.token_config()



def _np_savez(path: Path, **arrays: Any) -> None:
    ensure_parent(path)
    np.savez_compressed(path, **arrays)



def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value



def _load_or_run_forward(
    design: MorrisDesign,
    settings: ForwardSettings,
    out_dir: Path,
    cache_manifest_payload: Dict[str, Any],
    use_cache: bool,
    force: bool,
    n_jobs: int,
    verbose: int,
) -> Dict[str, np.ndarray]:
    cache_npz = out_dir / "morris_forward_outputs.npz"
    cache_manifest = out_dir / "morris_manifest.json"
    if use_cache and (not force) and cache_npz.exists() and check_cache_manifest(cache_manifest, cache_manifest_payload):
        return dict(np.load(cache_npz, allow_pickle=False))

    stacked = evaluate_theta_matrix(design.theta, settings=settings, n_jobs=n_jobs, verbose=verbose)
    _np_savez(cache_npz, **stacked)
    save_manifest(cache_manifest, _json_ready(cache_manifest_payload))
    return stacked



def _write_rejection_summary(stacked: Mapping[str, np.ndarray], out_dir: Path) -> Dict[str, Any]:
    accepted = np.asarray(stacked["accepted"], dtype=bool)
    reasons = decode_bytes_array(stacked["rejection_reason"])
    counts: Dict[str, int] = {}
    reject_counts: Dict[str, int] = {}
    for acc, reason in zip(accepted, reasons):
        counts[reason] = counts.get(reason, 0) + 1
        if not acc:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
    payload = {
        "n_total": int(accepted.size),
        "n_accepted": int(np.sum(accepted)),
        "n_rejected": int(np.sum(~accepted)),
        "acceptance_rate": float(np.mean(accepted.astype(np.float64))),
        "rejection_rate": float(np.mean((~accepted).astype(np.float64))),
        "all_reason_counts": counts,
        "rejected_reason_counts": reject_counts,
    }
    save_manifest(out_dir / "morris_rejection_summary.json", payload)
    rows = []
    for reason, count in sorted(counts.items()):
        rows.append(
            {
                "reason": reason,
                "count_all": int(count),
                "count_rejected": int(reject_counts.get(reason, 0)),
            }
        )
    long_table_to_csv(out_dir / "morris_rejection_summary.csv", rows, ["reason", "count_all", "count_rejected"])
    return payload



def _write_scalar_stats(
    scalar_names: Sequence[str],
    stats: MorrisStats,
    param_names: Sequence[str],
    design: MorrisDesign,
    out_dir: Path,
    stem: str,
) -> None:
    rows = []
    for p, param in enumerate(param_names):
        for o, output in enumerate(scalar_names):
            rows.append(
                {
                    "param": param,
                    "output": output,
                    "mu_star": float(stats.mu_star[p, o]),
                    "mu": float(stats.mu[p, o]),
                    "sigma": float(stats.sigma[p, o]),
                    "n_effects": int(stats.n_effects[p, o]),
                    "effect_fraction": float(stats.n_effects[p, o] / max(1, design.num_trajectories)),
                    "output_mean": float(stats.output_mean[o]),
                    "output_std": float(stats.output_std[o]),
                }
            )
    long_table_to_csv(
        out_dir / f"{stem}.csv",
        rows,
        ["param", "output", "mu_star", "mu", "sigma", "n_effects", "effect_fraction", "output_mean", "output_std"],
    )



def _family_aggregate_rows(
    family: str,
    stats: MorrisStats,
    param_names: Sequence[str],
    num_trajectories: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p, param in enumerate(param_names):
        mean_mu_star = float(np.nanmean(stats.mu_star[p]))
        mean_mu = float(np.nanmean(stats.mu[p]))
        mean_sigma = float(np.nanmean(stats.sigma[p]))
        mean_effect_fraction = float(np.nanmean(stats.n_effects[p] / max(1, num_trajectories)))
        rows.append(
            {
                "family": family,
                "param": param,
                "mean_mu_star": mean_mu_star,
                "mean_abs_mu": float(np.nanmean(np.abs(stats.mu[p]))),
                "mean_mu": mean_mu,
                "mean_sigma": mean_sigma,
                "mean_effect_fraction": mean_effect_fraction,
                "sigma_to_mu_star": float(mean_sigma / (mean_mu_star + 1e-12)) if np.isfinite(mean_sigma) else float("nan"),
                "n_outputs": int(stats.mu_star.shape[1]),
            }
        )
    return rows



def _reshape_erp_stats(stats: MorrisStats, cfg: TokenConfig, n_params: int) -> Dict[str, np.ndarray]:
    shape = (n_params, cfg.n_time_patches, cfg.n_channels)
    return {
        "mu_star": stats.mu_star.reshape(shape),
        "mu": stats.mu.reshape(shape),
        "sigma": stats.sigma.reshape(shape),
        "n_effects": stats.n_effects.reshape(shape),
    }



def _reshape_tfr_stats(stats: MorrisStats, cfg: TokenConfig, n_params: int) -> Dict[str, np.ndarray]:
    shape = (n_params, cfg.n_time_patches, cfg.n_freq_patches, cfg.n_channels)
    return {
        "mu_star": stats.mu_star.reshape(shape),
        "mu": stats.mu.reshape(shape),
        "sigma": stats.sigma.reshape(shape),
        "n_effects": stats.n_effects.reshape(shape),
    }



def _write_erp_long(reshaped: Mapping[str, np.ndarray], param_names: Sequence[str], out_dir: Path) -> None:
    rows = []
    mu_star = reshaped["mu_star"]
    mu = reshaped["mu"]
    sigma = reshaped["sigma"]
    n_effects = reshaped["n_effects"]
    for p, param in enumerate(param_names):
        for ti in range(mu_star.shape[1]):
            for ci, ch in enumerate(DEFAULT_16_CH_NAMES):
                rows.append(
                    {
                        "param": param,
                        "time_patch": int(ti),
                        "channel": ch,
                        "mu_star": float(mu_star[p, ti, ci]),
                        "mu": float(mu[p, ti, ci]),
                        "sigma": float(sigma[p, ti, ci]),
                        "n_effects": int(n_effects[p, ti, ci]),
                    }
                )
    long_table_to_csv(
        out_dir / "morris_erp_heatmap_long.csv",
        rows,
        ["param", "time_patch", "channel", "mu_star", "mu", "sigma", "n_effects"],
    )



def _write_tfr_long(reshaped: Mapping[str, np.ndarray], param_names: Sequence[str], out_dir: Path) -> None:
    rows = []
    mu_star = reshaped["mu_star"]
    mu = reshaped["mu"]
    sigma = reshaped["sigma"]
    n_effects = reshaped["n_effects"]
    for p, param in enumerate(param_names):
        for ti in range(mu_star.shape[1]):
            for fi in range(mu_star.shape[2]):
                rows.append(
                    {
                        "param": param,
                        "time_patch": int(ti),
                        "freq_patch": int(fi),
                        "mu_star": float(np.nanmean(mu_star[p, ti, fi, :])),
                        "mu": float(np.nanmean(mu[p, ti, fi, :])),
                        "sigma": float(np.nanmean(sigma[p, ti, fi, :])),
                        "n_effects": float(np.nanmean(n_effects[p, ti, fi, :])),
                    }
                )
    long_table_to_csv(
        out_dir / "morris_tfr_heatmap_long.csv",
        rows,
        ["param", "time_patch", "freq_patch", "mu_star", "mu", "sigma", "n_effects"],
    )

    ch_rows = []
    for p, param in enumerate(param_names):
        for ci, ch in enumerate(DEFAULT_16_CH_NAMES):
            ch_rows.append(
                {
                    "param": param,
                    "channel": ch,
                    "mu_star": float(np.nanmean(mu_star[p, :, :, ci])),
                    "mu": float(np.nanmean(mu[p, :, :, ci])),
                    "sigma": float(np.nanmean(sigma[p, :, :, ci])),
                    "n_effects": float(np.nanmean(n_effects[p, :, :, ci])),
                }
            )
    long_table_to_csv(
        out_dir / "morris_tfr_channel_summary.csv",
        ch_rows,
        ["param", "channel", "mu_star", "mu", "sigma", "n_effects"],
    )



def _write_markdown_report(
    out_path: Path,
    rejection: Mapping[str, Any],
    family_aggregates: Sequence[Mapping[str, Any]],
    acceptance_stats: MorrisStats,
    param_names: Sequence[str],
    runtime_seconds: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_family: Dict[str, List[Mapping[str, Any]]] = {}
    for row in family_aggregates:
        by_family.setdefault(str(row["family"]), []).append(row)

    lines: List[str] = []
    lines.append("# Morris sensitivity summary")
    lines.append("")
    lines.append(f"- Runtime: {runtime_seconds:.1f} s")
    lines.append(f"- Total simulations: {rejection['n_total']}")
    lines.append(f"- Acceptance rate: {100.0 * rejection['acceptance_rate']:.2f}%")
    lines.append(f"- Rejection rate: {100.0 * rejection['rejection_rate']:.2f}%")
    if rejection["rejected_reason_counts"]:
        lines.append("- Rejection reasons:")
        for reason, count in sorted(rejection["rejected_reason_counts"].items()):
            lines.append(f"  - {reason}: {count}")
    else:
        lines.append("- No rejected simulations were observed in this Morris run.")
    lines.append("")
    for family, rows in sorted(by_family.items()):
        top = sorted(rows, key=lambda r: float(r["mean_mu_star"]), reverse=True)[:3]
        lines.append(f"## {family}")
        lines.append("")
        if top:
            top_str = ", ".join(f"{row['param']} ({float(row['mean_mu_star']):.3g})" for row in top)
            lines.append(f"Top mean mu* parameters: {top_str}.")
        else:
            lines.append("No valid Morris effects were available.")
        lines.append("")
    acc_mu = acceptance_stats.mu_star[:, 0]
    if np.isfinite(acc_mu).any() and np.nanmax(acc_mu) > 0:
        order = np.argsort(-np.nan_to_num(acc_mu, nan=-np.inf))
        acc_str = ", ".join(f"{param_names[i]} ({acc_mu[i]:.3g})" for i in order[:3])
        lines.append(f"Acceptance-indicator sensitivity is strongest for: {acc_str}.")
    else:
        lines.append("Acceptance-indicator sensitivity is negligible in this Morris run.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(description="Run Morris global sensitivity analysis on the CMC EEG forward model.")
    parser.add_argument("--config", type=str, default=None, help="Path to config_sensitivity.yaml")
    parser.add_argument("--out-root", type=str, default=None, help="Root output directory (defaults to config paths.results_root)")
    parser.add_argument("--trajectories", type=int, default=None, help="Override morris.num_trajectories")
    parser.add_argument("--levels", type=int, default=None, help="Override morris.num_levels")
    parser.add_argument("--grid-jump", type=int, default=None, help="Override morris.grid_jump")
    parser.add_argument("--candidate-pool-size", type=int, default=None, help="Override morris.candidate_pool_size")
    parser.add_argument("--seed", type=int, default=None, help="Override morris.seed")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for forward evaluation")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache matches")
    parser.add_argument("--no-cache", action="store_true", help="Disable loading/saving cached forward outputs")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny sample budget for a fast smoke run")
    parser.add_argument("--verbose", type=int, default=0, help="joblib verbosity for forward simulation")
    args = parser.parse_args()

    t_start = time.time()
    config = load_config(args.config)
    spec = validate_project_parameter_spec()
    settings = forward_settings_from_config(config)

    morris_cfg = dict(config.get("morris", {}))
    n_jobs = int(args.n_jobs if args.n_jobs is not None else morris_cfg.get("n_jobs", 1))
    num_trajectories = int(args.trajectories if args.trajectories is not None else morris_cfg.get("num_trajectories", 24))
    num_levels = int(args.levels if args.levels is not None else morris_cfg.get("num_levels", 6))
    grid_jump = int(args.grid_jump if args.grid_jump is not None else morris_cfg.get("grid_jump", max(1, num_levels // 2)))
    candidate_pool_size = int(args.candidate_pool_size if args.candidate_pool_size is not None else morris_cfg.get("candidate_pool_size", num_trajectories))
    seed = int(args.seed if args.seed is not None else morris_cfg.get("seed", 2026))
    standardize = bool(morris_cfg.get("standardize_outputs", True))
    mask_rejected_outputs = bool(morris_cfg.get("mask_rejected_outputs", True))
    if args.smoke:
        num_trajectories = min(num_trajectories, 4)
        num_levels = min(num_levels, 4)
        grid_jump = min(grid_jump, max(1, num_levels // 2))
        candidate_pool_size = min(candidate_pool_size, max(4, num_trajectories))
        n_jobs = 1

    results_root = sensitivity_root(args.out_root or config.get("paths", {}).get("results_root", "results_sensitivity"))
    out_dir = results_root / "morris"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = results_root / "figures"
    table_dir = results_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(config.get("figures", {}).get("formats", ["png", "pdf"]))

    design = generate_morris_design(
        spec=spec,
        num_trajectories=num_trajectories,
        num_levels=num_levels,
        seed=seed,
        grid_jump=grid_jump,
        candidate_pool_size=candidate_pool_size,
    )
    save_theta_csv(out_dir / "morris_design_theta.csv", design.theta, spec.names)
    _np_savez(
        out_dir / "morris_design.npz",
        theta=design.theta.astype(np.float32),
        unit=design.unit.astype(np.float32),
        trajectory_ids=design.trajectory_ids.astype(np.int32),
        point_in_trajectory=design.point_in_trajectory.astype(np.int32),
        step_from=design.step_from.astype(np.int32),
        step_to=design.step_to.astype(np.int32),
        step_param=design.step_param.astype(np.int32),
    )

    cache_manifest_payload = {
        "analysis": "morris",
        "theta_sha1": array_sha1(design.theta.astype(np.float32)),
        "unit_sha1": array_sha1(design.unit.astype(np.float32)),
        "settings": {
            "deterministic_seed": settings.deterministic_seed,
            "fs": settings.fs,
            "duration": settings.duration,
            "n_trials": settings.n_trials,
            "bandpass": list(settings.bandpass),
            "tfr_method": settings.tfr_method,
        },
        "morris": {
            "num_levels": design.num_levels,
            "grid_jump": design.grid_jump,
            "num_trajectories": design.num_trajectories,
            "candidate_pool_size": design.candidate_pool_size,
            "seed": design.seed,
            "standardize_outputs": standardize,
        },
    }
    stacked = _load_or_run_forward(
        design=design,
        settings=settings,
        out_dir=out_dir,
        cache_manifest_payload=cache_manifest_payload,
        use_cache=bool(config.get("caching", {}).get("use_cache", True)) and (not args.no_cache),
        force=args.force,
        n_jobs=n_jobs,
        verbose=args.verbose,
    )

    rejection = _write_rejection_summary(stacked, out_dir)

    scalar_names = decode_bytes_array(stacked["scalar_names"])
    scalar_values = np.asarray(stacked["scalar_values"], dtype=np.float64)
    accepted_indicator = np.asarray(stacked["accepted"], dtype=np.float64)[:, None]
    erp_tokens = np.asarray(stacked["erp_tokens"], dtype=np.float64)
    tfr_tokens = np.asarray(stacked["tfr_tokens"], dtype=np.float64)
    hybrid_tokens = np.asarray(stacked["hybrid_tokens"], dtype=np.float64)

    if mask_rejected_outputs:
        rejected = ~np.asarray(stacked["accepted"], dtype=bool)
        if np.any(rejected):
            scalar_values = scalar_values.copy()
            erp_tokens = erp_tokens.copy()
            tfr_tokens = tfr_tokens.copy()
            hybrid_tokens = hybrid_tokens.copy()
            scalar_values[rejected, :] = np.nan
            erp_tokens[rejected, ...] = np.nan
            tfr_tokens[rejected, ...] = np.nan
            hybrid_tokens[rejected, ...] = np.nan

    scalar_stats = compute_morris_statistics(scalar_values, design=design, standardize=standardize)
    acceptance_stats = compute_morris_statistics(accepted_indicator, design=design, standardize=False)
    erp_stats = compute_morris_statistics(erp_tokens.reshape(erp_tokens.shape[0], -1), design=design, standardize=standardize)
    tfr_stats = compute_morris_statistics(tfr_tokens.reshape(tfr_tokens.shape[0], -1), design=design, standardize=standardize)
    hybrid_stats = compute_morris_statistics(hybrid_tokens.reshape(hybrid_tokens.shape[0], -1), design=design, standardize=standardize)

    _write_scalar_stats(scalar_names, scalar_stats, spec.names, design, out_dir, "morris_scalar_stats")
    _write_scalar_stats(["accepted_indicator"], acceptance_stats, spec.names, design, out_dir, "morris_acceptance_stats")

    family_rows: List[Dict[str, Any]] = []
    family_rows.extend(_family_aggregate_rows("scalar", scalar_stats, spec.names, design.num_trajectories))
    family_rows.extend(_family_aggregate_rows("erp", erp_stats, spec.names, design.num_trajectories))
    family_rows.extend(_family_aggregate_rows("tfr", tfr_stats, spec.names, design.num_trajectories))
    family_rows.extend(_family_aggregate_rows("hybrid", hybrid_stats, spec.names, design.num_trajectories))
    long_table_to_csv(
        out_dir / "morris_family_aggregates.csv",
        family_rows,
        ["family", "param", "mean_mu_star", "mean_abs_mu", "mean_mu", "mean_sigma", "mean_effect_fraction", "sigma_to_mu_star", "n_outputs"],
    )

    token_cfg = _token_cfg_from_settings(settings)
    time_edges = np.linspace(0.0, settings.duration, token_cfg.n_time_patches + 1, dtype=np.float64)
    freq_edges = np.geomspace(token_cfg.f_min, token_cfg.f_max, token_cfg.n_freq_patches + 1).astype(np.float64)

    erp_reshaped = _reshape_erp_stats(erp_stats, token_cfg, spec.dim)
    tfr_reshaped = _reshape_tfr_stats(tfr_stats, token_cfg, spec.dim)
    _np_savez(
        out_dir / "morris_erp_stats.npz",
        mu_star=erp_reshaped["mu_star"].astype(np.float32),
        mu=erp_reshaped["mu"].astype(np.float32),
        sigma=erp_reshaped["sigma"].astype(np.float32),
        n_effects=erp_reshaped["n_effects"].astype(np.int32),
        time_edges=time_edges.astype(np.float32),
        channel_names=np.asarray([c.encode("utf-8") for c in DEFAULT_16_CH_NAMES], dtype="S"),
    )
    _np_savez(
        out_dir / "morris_tfr_stats.npz",
        mu_star=tfr_reshaped["mu_star"].astype(np.float32),
        mu=tfr_reshaped["mu"].astype(np.float32),
        sigma=tfr_reshaped["sigma"].astype(np.float32),
        n_effects=tfr_reshaped["n_effects"].astype(np.int32),
        time_edges=time_edges.astype(np.float32),
        freq_edges=freq_edges.astype(np.float32),
        channel_names=np.asarray([c.encode("utf-8") for c in DEFAULT_16_CH_NAMES], dtype="S"),
    )
    _np_savez(
        out_dir / "morris_hybrid_stats.npz",
        mu_star=hybrid_stats.mu_star.astype(np.float32),
        mu=hybrid_stats.mu.astype(np.float32),
        sigma=hybrid_stats.sigma.astype(np.float32),
        n_effects=hybrid_stats.n_effects.astype(np.int32),
    )

    _write_erp_long(erp_reshaped, spec.names, out_dir)
    _write_tfr_long(tfr_reshaped, spec.names, out_dir)

    mu_star_by_family = {
        "scalar": np.asarray([row["mean_mu_star"] for row in family_rows if row["family"] == "scalar"], dtype=np.float64),
        "erp": np.asarray([row["mean_mu_star"] for row in family_rows if row["family"] == "erp"], dtype=np.float64),
        "tfr": np.asarray([row["mean_mu_star"] for row in family_rows if row["family"] == "tfr"], dtype=np.float64),
        "hybrid": np.asarray([row["mean_mu_star"] for row in family_rows if row["family"] == "hybrid"], dtype=np.float64),
    }
    sigma_by_family = {
        "scalar": np.asarray([row["mean_sigma"] for row in family_rows if row["family"] == "scalar"], dtype=np.float64),
        "erp": np.asarray([row["mean_sigma"] for row in family_rows if row["family"] == "erp"], dtype=np.float64),
        "tfr": np.asarray([row["mean_sigma"] for row in family_rows if row["family"] == "tfr"], dtype=np.float64),
        "hybrid": np.asarray([row["mean_sigma"] for row in family_rows if row["family"] == "hybrid"], dtype=np.float64),
    }

    plot_morris_family_summary(
        mu_star_by_family=mu_star_by_family,
        sigma_by_family=sigma_by_family,
        param_names=spec.names,
        out_base=fig_dir / "morris_per_parameter_summary",
        formats=formats,
        title="Morris summary by parameter and output family",
    )
    plot_erp_heatmaps(
        mu_star_time_channel=erp_reshaped["mu_star"],
        param_names=spec.names,
        channel_names=DEFAULT_16_CH_NAMES,
        time_edges=time_edges,
        out_base=fig_dir / "morris_erp_heatmaps",
        formats=formats,
    )
    plot_tfr_heatmaps(
        mu_star_time_freq=np.nanmean(tfr_reshaped["mu_star"], axis=3),
        param_names=spec.names,
        time_edges=time_edges,
        freq_edges=freq_edges,
        out_base=fig_dir / "morris_tfr_heatmaps",
        formats=formats,
    )
    plot_acceptance_morris(
        mu_star=acceptance_stats.mu_star[:, 0],
        sigma=acceptance_stats.sigma[:, 0],
        param_names=spec.names,
        out_base=fig_dir / "morris_acceptance_indicator",
        formats=formats,
    )

    runtime_seconds = float(time.time() - t_start)
    save_manifest(
        out_dir / "morris_runtime.json",
        {
            "runtime_seconds": runtime_seconds,
            "num_trajectories": design.num_trajectories,
            "num_levels": design.num_levels,
            "grid_jump": design.grid_jump,
            "standardize_outputs": standardize,
            "n_jobs": n_jobs,
        },
    )
    _write_markdown_report(
        table_dir / "morris_summary.md",
        rejection=rejection,
        family_aggregates=family_rows,
        acceptance_stats=acceptance_stats,
        param_names=spec.names,
        runtime_seconds=runtime_seconds,
    )

    print(
        json.dumps(
            {
                "analysis": "morris",
                "output_dir": str(out_dir),
                "rejection_rate": rejection["rejection_rate"],
                "runtime_seconds": runtime_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
