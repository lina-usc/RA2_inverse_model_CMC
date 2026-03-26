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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

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
from sensitivity.plotting import plot_sobol_bars
from sensitivity.sampling import (
    PCABasis,
    SobolDesign,
    SobolStats,
    apply_pca_basis,
    compute_sobol_statistics,
    fit_pca_basis,
    generate_sobol_design,
)



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
    all_theta: np.ndarray,
    settings: ForwardSettings,
    out_dir: Path,
    cache_manifest_payload: Dict[str, Any],
    use_cache: bool,
    force: bool,
    n_jobs: int,
    verbose: int,
) -> Dict[str, np.ndarray]:
    cache_npz = out_dir / "sobol_forward_outputs.npz"
    cache_manifest = out_dir / "sobol_manifest.json"
    if use_cache and (not force) and cache_npz.exists() and check_cache_manifest(cache_manifest, cache_manifest_payload):
        return dict(np.load(cache_npz, allow_pickle=False))
    stacked = evaluate_theta_matrix(all_theta, settings=settings, n_jobs=n_jobs, verbose=verbose)
    _np_savez(cache_npz, **stacked)
    save_manifest(cache_manifest, _json_ready(cache_manifest_payload))
    return stacked



def _split_blocks(stacked: Mapping[str, np.ndarray], design: SobolDesign) -> Dict[str, Any]:
    N = design.base_samples
    D = design.dim
    out: Dict[str, Any] = {}

    def _slice_any(arr: np.ndarray, sl: slice) -> np.ndarray:
        return np.asarray(arr[sl])

    def _extract_block(start: int, stop: int) -> Dict[str, np.ndarray]:
        sl = slice(start, stop)
        return {
            "theta": _slice_any(stacked["theta"], sl),
            "accepted": _slice_any(stacked["accepted"], sl),
            "rejection_reason": _slice_any(stacked["rejection_reason"], sl),
            "eeg": _slice_any(stacked["eeg"], sl),
            "erp_tokens": _slice_any(stacked["erp_tokens"], sl),
            "tfr_tokens": _slice_any(stacked["tfr_tokens"], sl),
            "hybrid_tokens": _slice_any(stacked["hybrid_tokens"], sl),
            "scalar_values": _slice_any(stacked["scalar_values"], sl),
            "error_message": _slice_any(stacked["error_message"], sl),
        }

    out["A"] = _extract_block(0, N)
    out["B"] = _extract_block(N, 2 * N)
    out["AB"] = []
    for i in range(D):
        start = (2 + i) * N
        stop = (3 + i) * N
        out["AB"].append(_extract_block(start, stop))
    out["scalar_names"] = decode_bytes_array(stacked["scalar_names"])
    return out



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
    save_manifest(out_dir / "sobol_rejection_summary.json", payload)
    rows = []
    for reason, count in sorted(counts.items()):
        rows.append({"reason": reason, "count_all": int(count), "count_rejected": int(reject_counts.get(reason, 0))})
    long_table_to_csv(out_dir / "sobol_rejection_summary.csv", rows, ["reason", "count_all", "count_rejected"])
    return payload



def _select_scalar_outputs(
    scalar_names: Sequence[str],
    scalar_values: np.ndarray,
    selected_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    lookup = {name: i for i, name in enumerate(scalar_names)}
    keep_idx: List[int] = []
    keep_names: List[str] = []
    for name in selected_names:
        if name in lookup:
            keep_idx.append(lookup[name])
            keep_names.append(name)
    if not keep_idx:
        raise ValueError("None of the requested scalar Sobol outputs were found in the forward wrapper outputs")
    return np.asarray(scalar_values[:, keep_idx], dtype=np.float64), keep_names



def _fit_basis_or_none(X_fit: np.ndarray, n_components: int, source: str) -> Optional[PCABasis]:
    if int(n_components) <= 0:
        return None
    finite_rows = np.all(np.isfinite(X_fit), axis=1)
    X_ok = np.asarray(X_fit[finite_rows], dtype=np.float64)
    if X_ok.shape[0] < 4 or X_ok.shape[1] == 0:
        return None
    n_comp = min(int(n_components), max(1, min(X_ok.shape[0] - 1, X_ok.shape[1])))
    if X_ok.shape[0] < n_comp + 1:
        return None
    return fit_pca_basis(X_ok, n_components=n_comp, source=source)



def _apply_family_pca(
    basis: Optional[PCABasis],
    A: np.ndarray,
    B: np.ndarray,
    AB_list: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if basis is None:
        raise ValueError("PCA basis is not available")
    YA = apply_pca_basis(A, basis)
    YB = apply_pca_basis(B, basis)
    YAB = np.stack([apply_pca_basis(arr, basis) for arr in AB_list], axis=2)
    names = [f"PC{i + 1}" for i in range(YA.shape[1])]
    return YA, YB, YAB, names



def _write_sobol_long(
    family: str,
    output_names: Sequence[str],
    stats: SobolStats,
    param_names: Sequence[str],
    rows: List[Dict[str, Any]],
) -> None:
    for p, param in enumerate(param_names):
        for o, output in enumerate(output_names):
            rows.append(
                {
                    "family": family,
                    "output": output,
                    "param": param,
                    "S1": float(stats.S1[p, o]),
                    "ST": float(stats.ST[p, o]),
                    "S1_ci_low": float(stats.S1_ci_low[p, o]),
                    "S1_ci_high": float(stats.S1_ci_high[p, o]),
                    "ST_ci_low": float(stats.ST_ci_low[p, o]),
                    "ST_ci_high": float(stats.ST_ci_high[p, o]),
                    "interaction_gap": float(stats.ST[p, o] - stats.S1[p, o]) if np.isfinite(stats.ST[p, o]) and np.isfinite(stats.S1[p, o]) else float("nan"),
                    "n_valid": int(stats.n_valid[p, o]),
                }
            )



def _aggregate_sobol_family(family: str, stats: SobolStats, param_names: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p, param in enumerate(param_names):
        rows.append(
            {
                "family": family,
                "param": param,
                "mean_S1": float(np.nanmean(stats.S1[p])),
                "mean_ST": float(np.nanmean(stats.ST[p])),
                "mean_interaction_gap": float(np.nanmean(stats.ST[p] - stats.S1[p])),
                "mean_n_valid": float(np.nanmean(stats.n_valid[p])),
                "n_outputs": int(stats.S1.shape[1]),
            }
        )
    return rows



def _write_pca_basis(path: Path, basis: Optional[PCABasis]) -> None:
    if basis is None:
        return
    _np_savez(
        path,
        scaler_mean=np.asarray(basis.scaler_mean, dtype=np.float32),
        scaler_scale=np.asarray(basis.scaler_scale, dtype=np.float32),
        pca_mean=np.asarray(basis.pca_mean, dtype=np.float32),
        components=np.asarray(basis.components, dtype=np.float32),
        explained_variance_ratio=np.asarray(basis.explained_variance_ratio, dtype=np.float32),
        source=np.asarray(str(basis.source).encode("utf-8"), dtype="S"),
    )



def _write_markdown_report(
    out_path: Path,
    rejection: Mapping[str, Any],
    family_aggregates: Sequence[Mapping[str, Any]],
    runtime_seconds: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Sobol sensitivity summary")
    lines.append("")
    lines.append(f"- Runtime: {runtime_seconds:.1f} s")
    lines.append(f"- Total simulations: {rejection['n_total']}")
    lines.append(f"- Acceptance rate: {100.0 * rejection['acceptance_rate']:.2f}%")
    lines.append(f"- Rejection rate: {100.0 * rejection['rejection_rate']:.2f}%")
    if rejection["rejected_reason_counts"]:
        lines.append("- Rejection reasons:")
        for reason, count in sorted(rejection["rejected_reason_counts"].items()):
            lines.append(f"  - {reason}: {count}")
    lines.append("")

    by_family: Dict[str, List[Mapping[str, Any]]] = {}
    for row in family_aggregates:
        by_family.setdefault(str(row["family"]), []).append(row)

    for family, rows in sorted(by_family.items()):
        top_main = sorted(rows, key=lambda r: float(r["mean_S1"]), reverse=True)[:3]
        top_total = sorted(rows, key=lambda r: float(r["mean_ST"]), reverse=True)[:3]
        top_inter = sorted(rows, key=lambda r: float(r["mean_interaction_gap"]), reverse=True)[:3]
        lines.append(f"## {family}")
        lines.append("")
        if top_main:
            lines.append("Top mean S1 parameters: " + ", ".join(f"{r['param']} ({float(r['mean_S1']):.3g})" for r in top_main) + ".")
            lines.append("Top mean ST parameters: " + ", ".join(f"{r['param']} ({float(r['mean_ST']):.3g})" for r in top_total) + ".")
            lines.append("Largest mean interaction gaps (ST-S1): " + ", ".join(f"{r['param']} ({float(r['mean_interaction_gap']):.3g})" for r in top_inter) + ".")
        else:
            lines.append("No valid Sobol indices were available.")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sobol global sensitivity analysis on selected CMC EEG outputs.")
    parser.add_argument("--config", type=str, default=None, help="Path to config_sensitivity.yaml")
    parser.add_argument("--out-root", type=str, default=None, help="Root output directory (defaults to config paths.results_root)")
    parser.add_argument("--base-samples", type=int, default=None, help="Override sobol.base_samples (prefer a power of two)")
    parser.add_argument("--seed", type=int, default=None, help="Override sobol.seed")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel jobs for forward evaluation")
    parser.add_argument("--bootstrap", type=int, default=None, help="Override sobol.bootstrap_resamples")
    parser.add_argument("--pca-components", type=int, default=None, help="Override sobol.pca_components")
    parser.add_argument("--force", action="store_true", help="Recompute even if cache matches")
    parser.add_argument("--no-cache", action="store_true", help="Disable loading/saving cached forward outputs")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny sample budget for a fast smoke run")
    parser.add_argument("--verbose", type=int, default=0, help="joblib verbosity for forward simulation")
    args = parser.parse_args()

    t_start = time.time()
    config = load_config(args.config)
    spec = validate_project_parameter_spec()
    settings = forward_settings_from_config(config)

    sobol_cfg = dict(config.get("sobol", {}))
    base_samples = int(args.base_samples if args.base_samples is not None else sobol_cfg.get("base_samples", 128))
    seed = int(args.seed if args.seed is not None else sobol_cfg.get("seed", 2027))
    n_jobs = int(args.n_jobs if args.n_jobs is not None else sobol_cfg.get("n_jobs", 1))
    bootstrap = int(args.bootstrap if args.bootstrap is not None else sobol_cfg.get("bootstrap_resamples", 200))
    pca_components = int(args.pca_components if args.pca_components is not None else sobol_cfg.get("pca_components", 3))
    selected_scalar_names = list(sobol_cfg.get("selected_scalar_outputs", []))
    if args.smoke:
        base_samples = min(base_samples, 8)
        bootstrap = min(bootstrap, 20)
        pca_components = min(pca_components, 2)
        n_jobs = 1

    results_root = sensitivity_root(args.out_root or config.get("paths", {}).get("results_root", "results_sensitivity"))
    out_dir = results_root / "sobol"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = results_root / "figures"
    table_dir = results_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(config.get("figures", {}).get("formats", ["png", "pdf"]))

    design = generate_sobol_design(spec=spec, base_samples=base_samples, seed=seed)
    save_theta_csv(out_dir / "sobol_A_theta.csv", design.A, spec.names)
    save_theta_csv(out_dir / "sobol_B_theta.csv", design.B, spec.names)
    ab_rows = []
    for i, name in enumerate(spec.names):
        for row in design.AB[i]:
            ab_rows.append({**{spec.names[j]: float(row[j]) for j in range(spec.dim)}, "ab_param": name})
    long_table_to_csv(out_dir / "sobol_AB_theta.csv", ab_rows, list(spec.names) + ["ab_param"])
    _np_savez(out_dir / "sobol_design.npz", A=design.A.astype(np.float32), B=design.B.astype(np.float32), AB=design.AB.astype(np.float32))

    all_theta = np.vstack([design.A, design.B] + [design.AB[i] for i in range(spec.dim)])
    cache_manifest_payload = {
        "analysis": "sobol",
        "theta_sha1": array_sha1(all_theta.astype(np.float32)),
        "settings": {
            "deterministic_seed": settings.deterministic_seed,
            "fs": settings.fs,
            "duration": settings.duration,
            "n_trials": settings.n_trials,
            "bandpass": list(settings.bandpass),
            "tfr_method": settings.tfr_method,
        },
        "sobol": {
            "base_samples": design.base_samples,
            "seed": design.seed,
            "bootstrap_resamples": bootstrap,
            "pca_components": pca_components,
            "second_order": False,
        },
    }
    stacked = _load_or_run_forward(
        all_theta=all_theta,
        settings=settings,
        out_dir=out_dir,
        cache_manifest_payload=cache_manifest_payload,
        use_cache=bool(config.get("caching", {}).get("use_cache", True)) and (not args.no_cache),
        force=args.force,
        n_jobs=n_jobs,
        verbose=args.verbose,
    )
    rejection = _write_rejection_summary(stacked, out_dir)
    blocks = _split_blocks(stacked, design)

    scalar_names = list(blocks["scalar_names"])
    A_scalar, scalar_keep_names = _select_scalar_outputs(scalar_names, np.asarray(blocks["A"]["scalar_values"], dtype=np.float64), selected_scalar_names)
    B_scalar, _ = _select_scalar_outputs(scalar_names, np.asarray(blocks["B"]["scalar_values"], dtype=np.float64), scalar_keep_names)
    AB_scalar = np.stack(
        [
            _select_scalar_outputs(scalar_names, np.asarray(blocks["AB"][i]["scalar_values"], dtype=np.float64), scalar_keep_names)[0]
            for i in range(spec.dim)
        ],
        axis=2,
    )
    scalar_stats = compute_sobol_statistics(A_scalar, B_scalar, AB_scalar, bootstrap_resamples=bootstrap, seed=seed)

    acc_A = np.asarray(blocks["A"]["accepted"], dtype=np.float64)[:, None]
    acc_B = np.asarray(blocks["B"]["accepted"], dtype=np.float64)[:, None]
    acc_AB = np.stack([np.asarray(blocks["AB"][i]["accepted"], dtype=np.float64)[:, None] for i in range(spec.dim)], axis=2)
    acceptance_stats = compute_sobol_statistics(acc_A, acc_B, acc_AB, bootstrap_resamples=bootstrap, seed=seed + 1)

    erp_A = np.asarray(blocks["A"]["erp_tokens"], dtype=np.float64).reshape(design.base_samples, -1)
    erp_B = np.asarray(blocks["B"]["erp_tokens"], dtype=np.float64).reshape(design.base_samples, -1)
    erp_AB_list = [np.asarray(blocks["AB"][i]["erp_tokens"], dtype=np.float64).reshape(design.base_samples, -1) for i in range(spec.dim)]
    erp_basis = _fit_basis_or_none(np.vstack([erp_A, erp_B]), n_components=pca_components, source="erp_tokens")
    _write_pca_basis(out_dir / "sobol_erp_pca_basis.npz", erp_basis)
    erp_pc_stats: Optional[SobolStats] = None
    erp_pc_names: List[str] = []
    if erp_basis is not None:
        erp_YA, erp_YB, erp_YAB, erp_pc_names = _apply_family_pca(erp_basis, erp_A, erp_B, erp_AB_list)
        erp_pc_stats = compute_sobol_statistics(erp_YA, erp_YB, erp_YAB, bootstrap_resamples=bootstrap, seed=seed + 2)

    tfr_A = np.asarray(blocks["A"]["tfr_tokens"], dtype=np.float64).reshape(design.base_samples, -1)
    tfr_B = np.asarray(blocks["B"]["tfr_tokens"], dtype=np.float64).reshape(design.base_samples, -1)
    tfr_AB_list = [np.asarray(blocks["AB"][i]["tfr_tokens"], dtype=np.float64).reshape(design.base_samples, -1) for i in range(spec.dim)]
    tfr_basis = _fit_basis_or_none(np.vstack([tfr_A, tfr_B]), n_components=pca_components, source="tfr_tokens")
    _write_pca_basis(out_dir / "sobol_tfr_pca_basis.npz", tfr_basis)
    tfr_pc_stats: Optional[SobolStats] = None
    tfr_pc_names: List[str] = []
    if tfr_basis is not None:
        tfr_YA, tfr_YB, tfr_YAB, tfr_pc_names = _apply_family_pca(tfr_basis, tfr_A, tfr_B, tfr_AB_list)
        tfr_pc_stats = compute_sobol_statistics(tfr_YA, tfr_YB, tfr_YAB, bootstrap_resamples=bootstrap, seed=seed + 3)

    hybrid_A = np.asarray(blocks["A"]["hybrid_tokens"], dtype=np.float64).reshape(design.base_samples, -1)
    hybrid_B = np.asarray(blocks["B"]["hybrid_tokens"], dtype=np.float64).reshape(design.base_samples, -1)
    hybrid_AB_list = [np.asarray(blocks["AB"][i]["hybrid_tokens"], dtype=np.float64).reshape(design.base_samples, -1) for i in range(spec.dim)]
    hybrid_basis = _fit_basis_or_none(np.vstack([hybrid_A, hybrid_B]), n_components=pca_components, source="hybrid_tokens")
    _write_pca_basis(out_dir / "sobol_hybrid_pca_basis.npz", hybrid_basis)
    hybrid_pc_stats: Optional[SobolStats] = None
    hybrid_pc_names: List[str] = []
    if hybrid_basis is not None:
        hybrid_YA, hybrid_YB, hybrid_YAB, hybrid_pc_names = _apply_family_pca(hybrid_basis, hybrid_A, hybrid_B, hybrid_AB_list)
        hybrid_pc_stats = compute_sobol_statistics(hybrid_YA, hybrid_YB, hybrid_YAB, bootstrap_resamples=bootstrap, seed=seed + 4)

    detailed_rows: List[Dict[str, Any]] = []
    family_aggregates: List[Dict[str, Any]] = []
    _write_sobol_long("scalar", scalar_keep_names, scalar_stats, spec.names, detailed_rows)
    family_aggregates.extend(_aggregate_sobol_family("scalar", scalar_stats, spec.names))
    _write_sobol_long("acceptance", ["accepted_indicator"], acceptance_stats, spec.names, detailed_rows)
    family_aggregates.extend(_aggregate_sobol_family("acceptance", acceptance_stats, spec.names))
    if erp_pc_stats is not None:
        _write_sobol_long("erp_pc", erp_pc_names, erp_pc_stats, spec.names, detailed_rows)
        family_aggregates.extend(_aggregate_sobol_family("erp_pc", erp_pc_stats, spec.names))
    if tfr_pc_stats is not None:
        _write_sobol_long("tfr_pc", tfr_pc_names, tfr_pc_stats, spec.names, detailed_rows)
        family_aggregates.extend(_aggregate_sobol_family("tfr_pc", tfr_pc_stats, spec.names))
    if hybrid_pc_stats is not None:
        _write_sobol_long("hybrid_pc", hybrid_pc_names, hybrid_pc_stats, spec.names, detailed_rows)
        family_aggregates.extend(_aggregate_sobol_family("hybrid_pc", hybrid_pc_stats, spec.names))

    long_table_to_csv(
        out_dir / "sobol_detailed_indices.csv",
        detailed_rows,
        ["family", "output", "param", "S1", "ST", "S1_ci_low", "S1_ci_high", "ST_ci_low", "ST_ci_high", "interaction_gap", "n_valid"],
    )
    long_table_to_csv(
        out_dir / "sobol_family_param_aggregates.csv",
        family_aggregates,
        ["family", "param", "mean_S1", "mean_ST", "mean_interaction_gap", "mean_n_valid", "n_outputs"],
    )

    _np_savez(
        out_dir / "sobol_scalar_stats.npz",
        S1=scalar_stats.S1.astype(np.float32),
        ST=scalar_stats.ST.astype(np.float32),
        S1_ci_low=scalar_stats.S1_ci_low.astype(np.float32),
        S1_ci_high=scalar_stats.S1_ci_high.astype(np.float32),
        ST_ci_low=scalar_stats.ST_ci_low.astype(np.float32),
        ST_ci_high=scalar_stats.ST_ci_high.astype(np.float32),
        n_valid=scalar_stats.n_valid.astype(np.int32),
        output_names=np.asarray([n.encode("utf-8") for n in scalar_keep_names], dtype="S"),
    )
    _np_savez(
        out_dir / "sobol_acceptance_stats.npz",
        S1=acceptance_stats.S1.astype(np.float32),
        ST=acceptance_stats.ST.astype(np.float32),
        S1_ci_low=acceptance_stats.S1_ci_low.astype(np.float32),
        S1_ci_high=acceptance_stats.S1_ci_high.astype(np.float32),
        ST_ci_low=acceptance_stats.ST_ci_low.astype(np.float32),
        ST_ci_high=acceptance_stats.ST_ci_high.astype(np.float32),
        n_valid=acceptance_stats.n_valid.astype(np.int32),
    )
    if erp_pc_stats is not None:
        _np_savez(
            out_dir / "sobol_erp_pc_stats.npz",
            S1=erp_pc_stats.S1.astype(np.float32),
            ST=erp_pc_stats.ST.astype(np.float32),
            S1_ci_low=erp_pc_stats.S1_ci_low.astype(np.float32),
            S1_ci_high=erp_pc_stats.S1_ci_high.astype(np.float32),
            ST_ci_low=erp_pc_stats.ST_ci_low.astype(np.float32),
            ST_ci_high=erp_pc_stats.ST_ci_high.astype(np.float32),
            n_valid=erp_pc_stats.n_valid.astype(np.int32),
            output_names=np.asarray([n.encode("utf-8") for n in erp_pc_names], dtype="S"),
        )
    if tfr_pc_stats is not None:
        _np_savez(
            out_dir / "sobol_tfr_pc_stats.npz",
            S1=tfr_pc_stats.S1.astype(np.float32),
            ST=tfr_pc_stats.ST.astype(np.float32),
            S1_ci_low=tfr_pc_stats.S1_ci_low.astype(np.float32),
            S1_ci_high=tfr_pc_stats.S1_ci_high.astype(np.float32),
            ST_ci_low=tfr_pc_stats.ST_ci_low.astype(np.float32),
            ST_ci_high=tfr_pc_stats.ST_ci_high.astype(np.float32),
            n_valid=tfr_pc_stats.n_valid.astype(np.int32),
            output_names=np.asarray([n.encode("utf-8") for n in tfr_pc_names], dtype="S"),
        )
    if hybrid_pc_stats is not None:
        _np_savez(
            out_dir / "sobol_hybrid_pc_stats.npz",
            S1=hybrid_pc_stats.S1.astype(np.float32),
            ST=hybrid_pc_stats.ST.astype(np.float32),
            S1_ci_low=hybrid_pc_stats.S1_ci_low.astype(np.float32),
            S1_ci_high=hybrid_pc_stats.S1_ci_high.astype(np.float32),
            ST_ci_low=hybrid_pc_stats.ST_ci_low.astype(np.float32),
            ST_ci_high=hybrid_pc_stats.ST_ci_high.astype(np.float32),
            n_valid=hybrid_pc_stats.n_valid.astype(np.int32),
            output_names=np.asarray([n.encode("utf-8") for n in hybrid_pc_names], dtype="S"),
        )

    plot_sobol_bars(
        S1=scalar_stats.S1.T,
        ST=scalar_stats.ST.T,
        param_names=spec.names,
        output_names=scalar_keep_names,
        out_base=fig_dir / "sobol_scalar_outputs",
        family_name="scalar",
        formats=formats,
    )
    if erp_pc_stats is not None:
        plot_sobol_bars(
            S1=erp_pc_stats.S1.T,
            ST=erp_pc_stats.ST.T,
            param_names=spec.names,
            output_names=erp_pc_names,
            out_base=fig_dir / "sobol_erp_pc_outputs",
            family_name="ERP PC",
            formats=formats,
        )
    if tfr_pc_stats is not None:
        plot_sobol_bars(
            S1=tfr_pc_stats.S1.T,
            ST=tfr_pc_stats.ST.T,
            param_names=spec.names,
            output_names=tfr_pc_names,
            out_base=fig_dir / "sobol_tfr_pc_outputs",
            family_name="TFR PC",
            formats=formats,
        )
    if hybrid_pc_stats is not None:
        plot_sobol_bars(
            S1=hybrid_pc_stats.S1.T,
            ST=hybrid_pc_stats.ST.T,
            param_names=spec.names,
            output_names=hybrid_pc_names,
            out_base=fig_dir / "sobol_hybrid_pc_outputs",
            family_name="Hybrid PC",
            formats=formats,
        )
    if rejection["rejection_rate"] > float(config.get("acceptance", {}).get("non_negligible_rate", 0.01)):
        plot_sobol_bars(
            S1=acceptance_stats.S1.T,
            ST=acceptance_stats.ST.T,
            param_names=spec.names,
            output_names=["accepted_indicator"],
            out_base=fig_dir / "sobol_acceptance_indicator",
            family_name="acceptance",
            formats=formats,
        )

    runtime_seconds = float(time.time() - t_start)
    save_manifest(
        out_dir / "sobol_runtime.json",
        {
            "runtime_seconds": runtime_seconds,
            "base_samples": design.base_samples,
            "seed": design.seed,
            "bootstrap_resamples": bootstrap,
            "pca_components": pca_components,
            "n_jobs": n_jobs,
            "second_order": False,
        },
    )
    _write_markdown_report(table_dir / "sobol_summary.md", rejection, family_aggregates, runtime_seconds)

    print(
        json.dumps(
            {
                "analysis": "sobol",
                "output_dir": str(out_dir),
                "rejection_rate": rejection["rejection_rate"],
                "runtime_seconds": runtime_seconds,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
