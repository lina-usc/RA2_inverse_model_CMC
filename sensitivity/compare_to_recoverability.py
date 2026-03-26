from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from sensitivity.common import (
    discover_recoverability_csvs,
    fallback_recoverability_csvs,
    load_config,
    long_table_to_csv,
    nanrank,
    save_manifest,
    sensitivity_root,
    validate_project_parameter_spec,
)
from sensitivity.plotting import plot_rank_comparison_grid, plot_value_scatter



def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))



def _guess_family_from_path(path: Path) -> Optional[str]:
    s = "/".join(part.lower() for part in path.parts)
    if "hybrid" in s or "hyb" in s:
        return "hybrid"
    if "tfr" in s:
        return "tfr"
    if "erp" in s:
        return "erp"
    return None



def _choose_recoverability_files(config: Mapping[str, Any]) -> Dict[str, Path]:
    chosen: Dict[str, Path] = {}
    explicit = dict(config.get("recoverability", {}).get("files", {}))
    for family, raw_path in explicit.items():
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            chosen[family.lower()] = path

    for path in discover_recoverability_csvs():
        family = _guess_family_from_path(path)
        if family is None or family in chosen:
            continue
        chosen[family] = path

    if bool(config.get("recoverability", {}).get("fallback_to_reference", True)):
        for path in fallback_recoverability_csvs():
            family = _guess_family_from_path(path)
            if family is None or family in chosen:
                continue
            chosen[family] = path

    return chosen



def _parse_recoverability(path: Path, param_names: Sequence[str], param_ranges: Mapping[str, float]) -> Dict[str, np.ndarray]:
    rows = _read_csv(path)
    by_param = {row["param"]: row for row in rows if "param" in row}
    pearson = np.full((len(param_names),), np.nan, dtype=np.float64)
    nrmse = np.full((len(param_names),), np.nan, dtype=np.float64)
    rmse = np.full((len(param_names),), np.nan, dtype=np.float64)

    for i, param in enumerate(param_names):
        row = by_param.get(param)
        if not row:
            continue
        pearson[i] = float(row.get("pearson_mean", row.get("pearson", "nan")))

        nrmse_val = row.get("nrmse_mean", row.get("rmse_norm_mean", row.get("nrmse", row.get("rmse_norm", ""))))
        if nrmse_val not in (None, ""):
            nrmse[i] = float(nrmse_val)

        rmse_val = row.get("rmse_mean", row.get("rmse", ""))
        if rmse_val not in (None, ""):
            rmse[i] = float(rmse_val)

        if not np.isfinite(rmse[i]) and np.isfinite(nrmse[i]):
            rmse[i] = float(nrmse[i] * param_ranges[param])
        if not np.isfinite(nrmse[i]) and np.isfinite(rmse[i]):
            nrmse[i] = float(rmse[i] / param_ranges[param])

    return {
        "pearson": pearson,
        "nrmse": nrmse,
        "rmse": rmse,
        "path": np.asarray(str(path).encode("utf-8"), dtype="S"),
    }



def _read_family_aggregate_csv(path: Path, value_fields: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    rows = _read_csv(path)
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        family = str(row.get("family", "")).strip().lower()
        param = str(row.get("param", "")).strip()
        if not family or not param:
            continue
        out.setdefault(family, {})
        out[family].setdefault(param, {})
        for field in value_fields:
            value = row.get(field, "")
            out[family][param][field] = float(value) if value not in ("", None) else float("nan")
    return out



def _array_from_family_dict(
    family_dict: Mapping[str, Mapping[str, Mapping[str, float]]],
    family: str,
    param_names: Sequence[str],
    field: str,
) -> np.ndarray:
    arr = np.full((len(param_names),), np.nan, dtype=np.float64)
    fam = family_dict.get(family, {})
    for i, param in enumerate(param_names):
        if param in fam and field in fam[param]:
            arr[i] = float(fam[param][field])
    return arr



def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])



def _classify_explanation(rank_corr: float, top_overlap: int, bottom_overlap: int) -> str:
    if np.isfinite(rank_corr) and rank_corr >= 0.60 and top_overlap >= 2 and bottom_overlap >= 2:
        return "supports"
    if (np.isfinite(rank_corr) and rank_corr >= 0.25) or top_overlap >= 1 or bottom_overlap >= 1:
        return "partially supports"
    return "fails to explain"



def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sensitivity summaries to existing recoverability outputs.")
    parser.add_argument("--config", type=str, default=None, help="Path to config_sensitivity.yaml")
    parser.add_argument("--out-root", type=str, default=None, help="Root output directory (defaults to config paths.results_root)")
    args = parser.parse_args()

    config = load_config(args.config)
    spec = validate_project_parameter_spec()
    param_ranges = {name: float(spec.bounds[i, 1] - spec.bounds[i, 0]) for i, name in enumerate(spec.names)}
    results_root = sensitivity_root(args.out_root or config.get("paths", {}).get("results_root", "results_sensitivity"))
    cmp_dir = results_root / "comparisons"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = results_root / "figures"
    table_dir = results_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(config.get("figures", {}).get("formats", ["png", "pdf"]))

    morris_csv = results_root / "morris" / "morris_family_aggregates.csv"
    sobol_csv = results_root / "sobol" / "sobol_family_param_aggregates.csv"
    if not morris_csv.exists():
        raise FileNotFoundError(f"Expected Morris aggregates at {morris_csv}")
    if not sobol_csv.exists():
        raise FileNotFoundError(f"Expected Sobol aggregates at {sobol_csv}")

    morris = _read_family_aggregate_csv(morris_csv, ["mean_mu_star", "mean_sigma"])
    sobol = _read_family_aggregate_csv(sobol_csv, ["mean_S1", "mean_ST", "mean_interaction_gap"])

    recoverability_files = _choose_recoverability_files(config)
    if not recoverability_files:
        raise FileNotFoundError(
            "No recoverability CSVs were found. Point config.recoverability.files to your existing metrics_test.csv files."
        )

    recoverability: Dict[str, Dict[str, np.ndarray]] = {}
    for family, path in recoverability_files.items():
        recoverability[family] = _parse_recoverability(path, spec.names, param_ranges)

    combined_rows: List[Dict[str, Any]] = []
    for i, param in enumerate(spec.names):
        row: Dict[str, Any] = {"param": param}
        for family in ["scalar", "erp", "tfr", "hybrid"]:
            row[f"morris_{family}_mean_mu_star"] = float(_array_from_family_dict(morris, family, spec.names, "mean_mu_star")[i])
            row[f"morris_{family}_mean_sigma"] = float(_array_from_family_dict(morris, family, spec.names, "mean_sigma")[i])
        for family in ["scalar", "acceptance", "erp_pc", "tfr_pc", "hybrid_pc"]:
            row[f"sobol_{family}_mean_S1"] = float(_array_from_family_dict(sobol, family, spec.names, "mean_S1")[i])
            row[f"sobol_{family}_mean_ST"] = float(_array_from_family_dict(sobol, family, spec.names, "mean_ST")[i])
            row[f"sobol_{family}_mean_interaction_gap"] = float(_array_from_family_dict(sobol, family, spec.names, "mean_interaction_gap")[i])
        for family in ["erp", "tfr", "hybrid"]:
            rec = recoverability.get(family)
            row[f"recoverability_{family}_pearson"] = float(rec["pearson"][i]) if rec else float("nan")
            row[f"recoverability_{family}_nrmse"] = float(rec["nrmse"][i]) if rec else float("nan")
            row[f"recoverability_{family}_rmse"] = float(rec["rmse"][i]) if rec else float("nan")
        combined_rows.append(row)

    fieldnames = list(combined_rows[0].keys())
    long_table_to_csv(cmp_dir / "sensitivity_recoverability_table.csv", combined_rows, fieldnames)

    rank_grid: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    summary_rows: List[Dict[str, Any]] = []
    md_lines: List[str] = []
    md_lines.append("# Sensitivity vs recoverability comparison")
    md_lines.append("")
    md_lines.append("Forward sensitivity is compared to empirical recoverability; it is not treated as a substitute for recoverability, calibration, or identifiability.")
    md_lines.append("")

    for family in ["erp", "tfr", "hybrid"]:
        if family not in recoverability:
            continue
        sens_morris = _array_from_family_dict(morris, family, spec.names, "mean_mu_star")
        rec_pearson = recoverability[family]["pearson"]
        rec_nrmse = recoverability[family]["nrmse"]
        sens_rank = nanrank(sens_morris, higher_is_better=True)
        rec_rank = nanrank(rec_pearson, higher_is_better=True)
        rank_grid[family.upper()] = (sens_rank.astype(np.float64), rec_rank.astype(np.float64))
        rho_pearson = _safe_corr(sens_rank.astype(np.float64), rec_rank.astype(np.float64))
        rho_nrmse = _safe_corr(sens_rank.astype(np.float64), nanrank(rec_nrmse, higher_is_better=False).astype(np.float64))
        top_rec = set(np.asarray(spec.names)[np.argsort(-np.nan_to_num(rec_pearson, nan=-np.inf))[:3]])
        top_sens = set(np.asarray(spec.names)[np.argsort(-np.nan_to_num(sens_morris, nan=-np.inf))[:3]])
        bottom_rec = set(np.asarray(spec.names)[np.argsort(np.nan_to_num(rec_pearson, nan=np.inf))[:3]])
        bottom_sens = set(np.asarray(spec.names)[np.argsort(np.nan_to_num(sens_morris, nan=np.inf))[:3]])
        top_overlap = len(top_rec & top_sens)
        bottom_overlap = len(bottom_rec & bottom_sens)
        verdict = _classify_explanation(rho_pearson, top_overlap, bottom_overlap)
        sobol_family = f"{family}_pc"
        sobol_st = _array_from_family_dict(sobol, sobol_family, spec.names, "mean_ST")
        sobol_rho = _safe_corr(nanrank(sobol_st, higher_is_better=True).astype(np.float64), rec_rank.astype(np.float64)) if np.isfinite(sobol_st).any() else float("nan")
        summary_rows.append(
            {
                "family": family,
                "morris_rank_corr_vs_pearson": rho_pearson,
                "morris_rank_corr_vs_nrmse": rho_nrmse,
                "sobol_rank_corr_vs_pearson": sobol_rho,
                "top3_overlap": top_overlap,
                "bottom3_overlap": bottom_overlap,
                "verdict": verdict,
            }
        )
        md_lines.append(f"## {family.upper()}")
        md_lines.append("")
        md_lines.append(f"- Morris rank correlation with Pearson recoverability: {rho_pearson:.3f}" if np.isfinite(rho_pearson) else "- Morris rank correlation with Pearson recoverability: NaN")
        md_lines.append(f"- Morris rank correlation with nRMSE ranking: {rho_nrmse:.3f}" if np.isfinite(rho_nrmse) else "- Morris rank correlation with nRMSE ranking: NaN")
        if np.isfinite(sobol_rho):
            md_lines.append(f"- Sobol PC-based rank correlation with Pearson recoverability: {sobol_rho:.3f}")
        md_lines.append(f"- Top-3 overlap: {top_overlap}; bottom-3 overlap: {bottom_overlap}")
        md_lines.append(f"- Overall interpretation: forward sensitivity **{verdict}** the observed {family.upper()} recoverability hierarchy.")
        md_lines.append(f"- Top recoverability parameters: {', '.join(sorted(top_rec))}")
        md_lines.append(f"- Top sensitivity parameters: {', '.join(sorted(top_sens))}")
        md_lines.append(f"- Lowest recoverability parameters: {', '.join(sorted(bottom_rec))}")
        md_lines.append(f"- Lowest sensitivity parameters: {', '.join(sorted(bottom_sens))}")
        md_lines.append("")

    long_table_to_csv(
        cmp_dir / "sensitivity_recoverability_summary.csv",
        summary_rows,
        ["family", "morris_rank_corr_vs_pearson", "morris_rank_corr_vs_nrmse", "sobol_rank_corr_vs_pearson", "top3_overlap", "bottom3_overlap", "verdict"],
    )
    save_manifest(cmp_dir / "sensitivity_recoverability_sources.json", {fam: str(path) for fam, path in recoverability_files.items()})

    if rank_grid:
        plot_rank_comparison_grid(rank_grid, spec.names, fig_dir / "sensitivity_ranking_vs_recoverability", formats=formats)

    if "hybrid" in recoverability:
        hybrid_sens = _array_from_family_dict(morris, "hybrid", spec.names, "mean_mu_star")
        plot_value_scatter(
            x_values=recoverability["hybrid"]["pearson"],
            y_values=hybrid_sens,
            param_names=spec.names,
            x_label="Hybrid Pearson recoverability",
            y_label="Hybrid Morris mean mu*",
            title="Hybrid sensitivity vs recoverability",
            out_base=fig_dir / "hybrid_sensitivity_vs_recoverability",
            formats=formats,
        )

    (table_dir / "comparison_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "analysis": "comparison",
                "output_dir": str(cmp_dir),
                "recoverability_sources": {k: str(v) for k, v in recoverability_files.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
