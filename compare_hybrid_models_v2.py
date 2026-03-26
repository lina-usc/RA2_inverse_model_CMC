from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _decode_str_array(x) -> List[str]:
    x = np.asarray(x)
    out: List[str] = []
    for v in x:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode('utf-8'))
        else:
            out.append(str(v))
    return out


def _pick(npz, keys: Sequence[str], required: bool = True):
    for k in keys:
        if k in npz.files:
            return npz[k]
    if required:
        raise KeyError(f'None of keys found: {keys}. Available: {list(npz.files)}')
    return None


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum()) + 1e-12
    return float((x * y).sum() / denom)


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2)))


def _set_plot_defaults() -> None:
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })


def _load_eval_dir(eval_dir: str, split: str) -> Dict[str, np.ndarray]:
    path = os.path.join(eval_dir, f'eval_{split}_outputs.npz')
    if not os.path.isfile(path):
        path = os.path.join(eval_dir, 'eval_test_outputs.npz')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Could not find eval outputs in {eval_dir}')

    d = np.load(path, allow_pickle=True)
    theta_true = _pick(d, ['theta_true', 'theta', 'y_true_theta', 'theta_test'])
    theta_mean = _pick(d, ['theta_post_mean', 'theta_mean', 'post_mean_theta', 'theta_mu', 'theta_pred_mean'])
    theta_samples = _pick(d, ['theta_post_samples', 'theta_samples', 'post_samples_theta', 'samples_theta'])
    param_names = _pick(d, ['param_names', 'param_names_bytes'], required=False)
    eval_idx = _pick(d, ['eval_idx'], required=False)

    names = _decode_str_array(param_names) if param_names is not None else [f'param_{i}' for i in range(theta_true.shape[1])]
    out = {
        'path': np.array(path, dtype=object),
        'theta_true': np.asarray(theta_true, dtype=np.float32),
        'theta_mean': np.asarray(theta_mean, dtype=np.float32),
        'theta_samples': np.asarray(theta_samples, dtype=np.float32),
        'param_names': np.array(names, dtype=object),
    }
    if eval_idx is not None:
        out['eval_idx'] = np.asarray(eval_idx, dtype=np.int64)
    return out


def _check_alignment(ref: Dict[str, np.ndarray], other: Dict[str, np.ndarray], label_ref: str, label_other: str) -> None:
    if ref['theta_true'].shape != other['theta_true'].shape:
        raise ValueError(f'{label_ref} and {label_other} have different theta_true shapes')
    if not np.allclose(ref['theta_true'], other['theta_true'], atol=1e-6):
        raise ValueError(f'{label_ref} and {label_other} do not appear aligned; check eval ordering')
    if list(ref['param_names'].tolist()) != list(other['param_names'].tolist()):
        raise ValueError(f'{label_ref} and {label_other} param_names differ')


def _point_metrics(theta_true: np.ndarray, theta_mean: np.ndarray, prange: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = theta_true.shape[1]
    pear = np.zeros(p, dtype=np.float64)
    nrmse = np.zeros(p, dtype=np.float64)
    for j in range(p):
        pear[j] = _safe_pearson(theta_true[:, j], theta_mean[:, j])
        nrmse[j] = _rmse(theta_true[:, j], theta_mean[:, j]) / float(prange[j])
    return pear, nrmse


def _coverage_indicators(theta_true: np.ndarray, theta_samples: np.ndarray, level: float) -> np.ndarray:
    alpha = (1.0 - float(level)) / 2.0
    lo = np.quantile(theta_samples, alpha, axis=1)
    hi = np.quantile(theta_samples, 1.0 - alpha, axis=1)
    return ((theta_true >= lo) & (theta_true <= hi)).astype(np.float32)


def _binned_entropy_bits(samples: np.ndarray, edges: np.ndarray, alpha: float = 0.5) -> float:
    counts, _ = np.histogram(np.asarray(samples, dtype=np.float64), bins=edges)
    p = counts.astype(np.float64) + float(alpha)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p)))


def _posterior_informativeness(
    theta_samples: np.ndarray,
    baseline_theta: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    theta_samples = np.asarray(theta_samples, dtype=np.float64)
    baseline_theta = np.asarray(baseline_theta, dtype=np.float64)
    n_eval, _n_samp, p = theta_samples.shape
    entropy_reduction = np.zeros((n_eval, p), dtype=np.float32)
    width_shrinkage = np.zeros((n_eval, p), dtype=np.float32)

    for j in range(p):
        edges = np.linspace(float(low[j]), float(high[j]), int(n_bins) + 1)
        h_prior = _binned_entropy_bits(baseline_theta[:, j], edges)
        prior_q05 = float(np.quantile(baseline_theta[:, j], 0.05))
        prior_q95 = float(np.quantile(baseline_theta[:, j], 0.95))
        prior_w90 = max(1e-12, prior_q95 - prior_q05)

        post_q05 = np.quantile(theta_samples[:, :, j], 0.05, axis=1)
        post_q95 = np.quantile(theta_samples[:, :, j], 0.95, axis=1)
        width_shrinkage[:, j] = 1.0 - ((post_q95 - post_q05) / prior_w90)

        for n in range(n_eval):
            h_post = _binned_entropy_bits(theta_samples[n, :, j], edges)
            entropy_reduction[n, j] = h_prior - h_post

    return entropy_reduction, width_shrinkage


def _bootstrap_point_metrics(theta_true: np.ndarray, theta_mean: np.ndarray, prange: np.ndarray, boot_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    b, _n = boot_idx.shape
    p = theta_true.shape[1]
    pear_boot = np.zeros(b, dtype=np.float64)
    nrmse_boot = np.zeros(b, dtype=np.float64)

    for j in range(p):
        x = theta_true[:, j].astype(np.float64)
        y = theta_mean[:, j].astype(np.float64)
        xb = x[boot_idx]
        yb = y[boot_idx]
        xm = xb - xb.mean(axis=1, keepdims=True)
        ym = yb - yb.mean(axis=1, keepdims=True)
        denom = np.sqrt(np.sum(xm * xm, axis=1) * np.sum(ym * ym, axis=1)) + 1e-12
        pear_boot += np.sum(xm * ym, axis=1) / denom
        nrmse_boot += np.sqrt(np.mean((xb - yb) ** 2, axis=1)) / float(prange[j])

    pear_boot /= p
    nrmse_boot /= p
    return pear_boot, nrmse_boot


def _ci_from_boot(x: np.ndarray) -> Tuple[float, float]:
    lo, hi = np.quantile(np.asarray(x, dtype=np.float64), [0.025, 0.975])
    return float(lo), float(hi)


def _metric_row(model: str, point: Dict[str, float], ci: Dict[str, Tuple[float, float]]) -> Dict[str, object]:
    row: Dict[str, object] = {'model': model}
    for key in point.keys():
        row[key] = float(point[key])
        row[f'{key}_lo'] = float(ci[key][0])
        row[f'{key}_hi'] = float(ci[key][1])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description='Hybrid model comparison with bootstrap uncertainty and accepted-effective-prior informativeness.')
    ap.add_argument('--data-out', type=str, required=True)
    ap.add_argument('--effective-prior-npz', type=str, required=True)
    ap.add_argument('--fullcov-param-dir', type=str, required=True)
    ap.add_argument('--diag-param-dir', type=str, required=True)
    ap.add_argument('--fullcov-noparam-dir', type=str, required=True)
    ap.add_argument('--bilstm-fullcov-dir', type=str, required=True)
    ap.add_argument('--split', type=str, default='test')
    ap.add_argument('--out-dir', type=str, default='plots/hybrid_model_comparison_v2')
    ap.add_argument('--n-bootstrap', type=int, default=2000)
    ap.add_argument('--entropy-bins', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    _set_plot_defaults()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = np.load(os.path.join(args.data_out, 'param_meta.npz'), allow_pickle=True)
    if 'prior_low' in meta.files:
        low = np.asarray(meta['prior_low'], dtype=np.float32)
        high = np.asarray(meta['prior_high'], dtype=np.float32)
        param_names = _decode_str_array(meta['param_names'])
    else:
        low = np.asarray(meta['low'], dtype=np.float32)
        high = np.asarray(meta['high'], dtype=np.float32)
        param_names = _decode_str_array(meta['param_names'])
    prange = np.maximum(high - low, 1e-12)

    ep = np.load(args.effective_prior_npz, allow_pickle=True)
    base_parts = []
    if 'accepted_theta_mc' in ep.files:
        base_parts.append(np.asarray(ep['accepted_theta_mc'], dtype=np.float32))
    if 'dataset_theta' in ep.files:
        base_parts.append(np.asarray(ep['dataset_theta'], dtype=np.float32))
    if not base_parts:
        raise ValueError('effective_prior_npz does not contain accepted_theta_mc or dataset_theta')
    baseline_theta = np.vstack(base_parts).astype(np.float32)

    model_specs = [
        ('transformer_fullcov_paramtoken', 'Transformer, full covariance, parameter tokens', args.fullcov_param_dir),
        ('transformer_diag_paramtoken', 'Transformer, diagonal, parameter tokens', args.diag_param_dir),
        ('transformer_fullcov_noparamtoken', 'Transformer, full covariance, no parameter tokens', args.fullcov_noparam_dir),
        ('bilstm_fullcov', 'BiLSTM, full covariance', args.bilstm_fullcov_dir),
    ]

    loaded: Dict[str, Dict[str, np.ndarray]] = {}
    for key, _label, path in model_specs:
        loaded[key] = _load_eval_dir(path, args.split)

    ref_key = model_specs[0][0]
    for key, _label, _path in model_specs[1:]:
        _check_alignment(loaded[ref_key], loaded[key], ref_key, key)

    theta_true = loaded[ref_key]['theta_true']
    n_eval, p = theta_true.shape
    rng = np.random.default_rng(args.seed)
    boot_idx = rng.integers(0, n_eval, size=(int(args.n_bootstrap), n_eval), dtype=np.int64)

    summary_rows: List[Dict[str, object]] = []
    per_param_rows: List[Dict[str, object]] = []
    model_boot: Dict[str, Dict[str, np.ndarray]] = {}
    model_point: Dict[str, Dict[str, float]] = {}
    label_lookup = {key: label for key, label, _path in model_specs}

    for key, label, _path in model_specs:
        theta_mean = loaded[key]['theta_mean']
        theta_samples = loaded[key]['theta_samples']

        pear, nrmse = _point_metrics(theta_true, theta_mean, prange)
        cov50_ind = _coverage_indicators(theta_true, theta_samples, level=0.50)
        cov90_ind = _coverage_indicators(theta_true, theta_samples, level=0.90)
        entropy_red, width_shrink = _posterior_informativeness(
            theta_samples=theta_samples,
            baseline_theta=baseline_theta,
            low=low,
            high=high,
            n_bins=int(args.entropy_bins),
        )

        point = {
            'mean_pearson': float(np.mean(pear)),
            'mean_nrmse': float(np.mean(nrmse)),
            'cov50': float(np.mean(cov50_ind)),
            'cov90': float(np.mean(cov90_ind)),
            'mean_entropy_reduction_bits': float(np.mean(entropy_red)),
            'mean_width90_shrinkage': float(np.mean(width_shrink)),
        }
        model_point[key] = point

        pear_boot, nrmse_boot = _bootstrap_point_metrics(theta_true, theta_mean, prange, boot_idx)
        cov50_boot = cov50_ind[boot_idx].mean(axis=(1, 2))
        cov90_boot = cov90_ind[boot_idx].mean(axis=(1, 2))
        ent_boot = entropy_red[boot_idx].mean(axis=(1, 2))
        shrink_boot = width_shrink[boot_idx].mean(axis=(1, 2))

        boot = {
            'mean_pearson': pear_boot,
            'mean_nrmse': nrmse_boot,
            'cov50': cov50_boot,
            'cov90': cov90_boot,
            'mean_entropy_reduction_bits': ent_boot,
            'mean_width90_shrinkage': shrink_boot,
        }
        model_boot[key] = boot
        ci = {name: _ci_from_boot(boot[name]) for name in boot.keys()}
        summary_rows.append(_metric_row(label, point, ci))

        for j, pname in enumerate(param_names):
            per_param_rows.append({
                'model': label,
                'param': pname,
                'pearson': float(pear[j]),
                'nrmse': float(nrmse[j]),
                'cov50': float(np.mean(cov50_ind[:, j])),
                'cov90': float(np.mean(cov90_ind[:, j])),
                'entropy_reduction_bits': float(np.mean(entropy_red[:, j])),
                'width90_shrinkage': float(np.mean(width_shrink[:, j])),
            })

    with (out_dir / 'hybrid_ablation_summary_with_ci_v2.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    with (out_dir / 'hybrid_ablation_per_parameter_metrics_v2.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_param_rows[0].keys()))
        w.writeheader()
        w.writerows(per_param_rows)

    delta_specs = [
        ('transformer_diag_paramtoken', 'transformer_fullcov_paramtoken'),
        ('transformer_diag_paramtoken', 'transformer_fullcov_noparamtoken'),
        ('transformer_diag_paramtoken', 'bilstm_fullcov'),
        ('transformer_fullcov_noparamtoken', 'transformer_fullcov_paramtoken'),
    ]
    delta_rows: List[Dict[str, object]] = []
    for a, b in delta_specs:
        row: Dict[str, object] = {'comparison': f"{label_lookup[a]} minus {label_lookup[b]}"}
        for metric in list(model_point[a].keys()):
            delta_point = float(model_point[a][metric] - model_point[b][metric])
            delta_boot = model_boot[a][metric] - model_boot[b][metric]
            lo, hi = _ci_from_boot(delta_boot)
            row[metric] = delta_point
            row[f'{metric}_lo'] = lo
            row[f'{metric}_hi'] = hi
        delta_rows.append(row)

    with (out_dir / 'hybrid_ablation_key_deltas_v2.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
        w.writeheader()
        w.writerows(delta_rows)

    metric_titles = {
        'mean_pearson': 'Mean Pearson',
        'mean_nrmse': 'Mean nRMSE',
        'cov50': 'cov50',
        'cov90': 'cov90',
        'mean_entropy_reduction_bits': 'Mean entropy reduction (bits)',
        'mean_width90_shrinkage': 'Mean width-90 shrinkage',
    }

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.2), squeeze=False)
    axes = axes.reshape(-1)
    for ax, metric in zip(axes, metric_titles.keys()):
        vals = [float(model_point[key][metric]) for key, _label, _path in model_specs]
        los = [float(model_point[key][metric] - _ci_from_boot(model_boot[key][metric])[0]) for key, _label, _path in model_specs]
        his = [float(_ci_from_boot(model_boot[key][metric])[1] - model_point[key][metric]) for key, _label, _path in model_specs]
        x = np.arange(len(model_specs))
        ax.bar(x, vals, yerr=np.vstack([los, his]), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _key, label, _path in model_specs], rotation=20, ha='right')
        ax.set_title(metric_titles[metric])
        ax.grid(True, axis='y', alpha=0.25)
    fig.suptitle('Hybrid model comparison with 95% bootstrap intervals (v2)', y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    fig.savefig(out_dir / 'hybrid_model_summary_v2.png', dpi=int(args.dpi))
    fig.savefig(out_dir / 'hybrid_model_summary_v2.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), squeeze=False)
    axes = axes.reshape(-1)
    x = np.arange(p)
    width = 0.2
    for m_idx, (_key, label, _path) in enumerate(model_specs):
        sub = [row for row in per_param_rows if row['model'] == label]
        ent_vals = [float(r['entropy_reduction_bits']) for r in sub]
        shrink_vals = [float(r['width90_shrinkage']) for r in sub]
        axes[0].bar(x + (m_idx - 1.5) * width, ent_vals, width=width, label=label)
        axes[1].bar(x + (m_idx - 1.5) * width, shrink_vals, width=width, label=label)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(param_names, rotation=25, ha='right')
    axes[0].set_ylabel('Entropy reduction (bits)')
    axes[0].set_title('Per-parameter posterior entropy reduction')
    axes[0].grid(True, axis='y', alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(param_names, rotation=25, ha='right')
    axes[1].set_ylabel('Width-90 shrinkage')
    axes[1].set_title('Per-parameter posterior contraction')
    axes[1].grid(True, axis='y', alpha=0.25)

    axes[0].legend(frameon=False, ncol=1)
    fig.suptitle('Per-parameter posterior informativeness across Hybrid model variants (v2)', y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_dir / 'hybrid_posterior_informativeness_v2.png', dpi=int(args.dpi))
    fig.savefig(out_dir / 'hybrid_posterior_informativeness_v2.pdf')
    plt.close(fig)

    forest_metrics = list(metric_titles.keys())
    fig, axes = plt.subplots(1, len(delta_specs), figsize=(16.0, 5.2), squeeze=False)
    axes = axes.reshape(-1)
    for ax, row in zip(axes, delta_rows):
        y = np.arange(len(forest_metrics))
        vals = [float(row[m]) for m in forest_metrics]
        lo = [float(row[m] - row[f'{m}_lo']) for m in forest_metrics]
        hi = [float(row[f'{m}_hi'] - row[m]) for m in forest_metrics]
        ax.errorbar(vals, y, xerr=np.vstack([lo, hi]), fmt='o', capsize=3)
        ax.axvline(0.0, linestyle='--', linewidth=1.0, color='black')
        ax.set_yticks(y)
        ax.set_yticklabels([metric_titles[m] for m in forest_metrics])
        ax.set_title(str(row['comparison']).replace(' minus ', '\nminus\n'))
        ax.grid(True, axis='x', alpha=0.25)
    fig.suptitle('Key paired bootstrap deltas between Hybrid model variants (v2)', y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_dir / 'hybrid_model_key_deltas_v2.png', dpi=int(args.dpi))
    fig.savefig(out_dir / 'hybrid_model_key_deltas_v2.pdf')
    plt.close(fig)

    md: List[str] = []
    md.append('# Hybrid model comparison summary (v2)\n')
    md.append('## Headline metrics (95% bootstrap intervals)\n')
    for row in summary_rows:
        md.append(
            f"- **{row['model']}**: Mean Pearson = {row['mean_pearson']:.3f} [{row['mean_pearson_lo']:.3f}, {row['mean_pearson_hi']:.3f}], "
            f"Mean nRMSE = {row['mean_nrmse']:.3f} [{row['mean_nrmse_lo']:.3f}, {row['mean_nrmse_hi']:.3f}], "
            f"cov50 = {row['cov50']:.3f} [{row['cov50_lo']:.3f}, {row['cov50_hi']:.3f}], "
            f"cov90 = {row['cov90']:.3f} [{row['cov90_lo']:.3f}, {row['cov90_hi']:.3f}], "
            f"Mean entropy reduction = {row['mean_entropy_reduction_bits']:.3f} [{row['mean_entropy_reduction_bits_lo']:.3f}, {row['mean_entropy_reduction_bits_hi']:.3f}] bits, "
            f"Mean width-90 shrinkage = {row['mean_width90_shrinkage']:.3f} [{row['mean_width90_shrinkage_lo']:.3f}, {row['mean_width90_shrinkage_hi']:.3f}]."
        )

    md.append('\n## Key paired deltas\n')
    for row in delta_rows:
        md.append(
            f"- **{row['comparison']}**: ΔMean Pearson = {row['mean_pearson']:.3f} [{row['mean_pearson_lo']:.3f}, {row['mean_pearson_hi']:.3f}], "
            f"ΔMean nRMSE = {row['mean_nrmse']:.3f} [{row['mean_nrmse_lo']:.3f}, {row['mean_nrmse_hi']:.3f}], "
            f"Δcov50 = {row['cov50']:.3f} [{row['cov50_lo']:.3f}, {row['cov50_hi']:.3f}], "
            f"Δcov90 = {row['cov90']:.3f} [{row['cov90_lo']:.3f}, {row['cov90_hi']:.3f}], "
            f"ΔEntropy reduction = {row['mean_entropy_reduction_bits']:.3f} [{row['mean_entropy_reduction_bits_lo']:.3f}, {row['mean_entropy_reduction_bits_hi']:.3f}] bits, "
            f"ΔWidth-90 shrinkage = {row['mean_width90_shrinkage']:.3f} [{row['mean_width90_shrinkage_lo']:.3f}, {row['mean_width90_shrinkage_hi']:.3f}]."
        )

    (out_dir / 'hybrid_model_comparison_summary_v2.md').write_text('\n'.join(md))

    manifest = {
        'data_out': args.data_out,
        'effective_prior_npz': args.effective_prior_npz,
        'split': args.split,
        'n_eval': int(n_eval),
        'n_bootstrap': int(args.n_bootstrap),
        'entropy_bins': int(args.entropy_bins),
        'outputs': [
            'hybrid_ablation_summary_with_ci_v2.csv',
            'hybrid_ablation_key_deltas_v2.csv',
            'hybrid_ablation_per_parameter_metrics_v2.csv',
            'hybrid_model_summary_v2.png',
            'hybrid_posterior_informativeness_v2.png',
            'hybrid_model_key_deltas_v2.png',
            'hybrid_model_comparison_summary_v2.md',
        ],
    }
    (out_dir / 'hybrid_model_comparison_manifest_v2.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'analysis': 'hybrid_model_comparison_v2', 'out_dir': str(out_dir)}, indent=2))


if __name__ == '__main__':
    main()
