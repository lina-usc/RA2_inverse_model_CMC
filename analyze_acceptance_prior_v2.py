from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None
    delayed = None

from data.priors import build_prior_spec, sample_theta, theta_to_params
from sim.leadfield_mne import make_leadfield
from sim.regime_filter import RegimeFilterConfig, regime_reject
from sim.simulate_eeg import simulate_eeg


@dataclass(frozen=True)
class ReplaySettings:
    fs: int
    duration: float
    n_channels: int
    stim_onset: float
    stim_sigma: float
    stim_causal: bool
    n_sources: int
    n_trials: int
    input_noise_std: float
    sensor_noise_std: float
    internal_fs: int
    bandpass: Tuple[float, float]
    baseline_correct: bool
    baseline_window: Optional[Tuple[float, float]]
    warmup_sec: float
    downsample_method: str
    uV_scale: float
    leadfield_seed: int = 0

    def regime_filter_config(self) -> RegimeFilterConfig:
        return RegimeFilterConfig(fs=int(self.fs), duration=float(self.duration), stim_onset=float(self.stim_onset))


def _decode_scalar(x: Any) -> Any:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode('utf-8')
    if isinstance(x, np.generic):
        return x.item()
    return x


def _decode_list(x: Sequence[Any]) -> List[str]:
    return [str(_decode_scalar(v)) for v in x]


def _bool_from_any(x: Any) -> bool:
    x = _decode_scalar(x)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, float):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {'1', 'true', 't', 'yes', 'y'}:
        return True
    if s in {'0', 'false', 'f', 'no', 'n'}:
        return False
    raise ValueError(f'Cannot parse boolean from {x!r}')


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for p in paths:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as fh:
        out = json.load(fh)
    if not isinstance(out, dict):
        raise ValueError(f'Expected JSON object in {path}')
    return out


def _resolve_relative_to_data_out(data_out: Path, rel: str) -> Optional[Path]:
    p = Path(rel)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((Path.cwd() / p).resolve())
        for parent in [data_out.parent, *data_out.parents]:
            candidates.append((parent / p).resolve())
    name = p.name
    for parent in [data_out.parent, *data_out.parents]:
        candidates.append((parent / name).resolve())
        candidates.append((parent / 'data' / name).resolve())
    for cand in _unique_paths(candidates):
        if cand.exists():
            return cand
    return None


def _infer_stim_causal(attrs: Mapping[str, Any], source_h5: Path) -> Tuple[bool, str]:
    if 'stim_causal' in attrs:
        return _bool_from_any(attrs['stim_causal']), 'h5_attr:stim_causal'
    if 'stim_shape' in attrs:
        s = str(_decode_scalar(attrs['stim_shape'])).strip().lower()
        if any(tok in s for tok in ('noncausal', 'acausal', 'full_gaussian')):
            return False, 'h5_attr:stim_shape'
        if 'causal' in s:
            return True, 'h5_attr:stim_shape'
    name = source_h5.name.lower()
    if 'noncausal' in name or 'acausal' in name:
        return False, 'h5_filename'
    if 'causal' in name:
        return True, 'h5_filename'
    return True, 'default:true'


def _settings_from_h5(source_h5: Path) -> Tuple[ReplaySettings, np.ndarray, Dict[str, Any]]:
    with h5py.File(source_h5, 'r') as f:
        attrs = {str(k): _decode_scalar(v) for k, v in f.attrs.items()}
        stim_causal, stim_causal_from = _infer_stim_causal(attrs, source_h5)
        fs = int(attrs.get('fs', 250))
        duration = float(attrs.get('duration_sec', attrs.get('duration', 2.0)))
        n_channels = int(attrs.get('n_channels', 16))
        stim_onset = float(attrs.get('stim_onset_sec', attrs.get('stim_onset', 0.5)))
        stim_sigma = float(attrs.get('stim_sigma_sec', attrs.get('stim_sigma', 0.05)))
        n_sources = int(attrs.get('n_sources', 3))
        n_trials = int(attrs.get('n_trials', 10))
        input_noise_std = float(attrs.get('input_noise_std', 0.2))
        sensor_noise_std = float(attrs.get('sensor_noise_std', 2.0))
        internal_fs = int(attrs.get('internal_fs', 1000))
        band_lo = float(attrs.get('bandpass_lo_hz', 0.5))
        band_hi = float(attrs.get('bandpass_hi_hz', 40.0))
        baseline_correct = _bool_from_any(attrs.get('baseline_correct', 1))
        warmup_sec = float(attrs.get('warmup_sec', 3.0))
        downsample_method = str(attrs.get('downsample_method', 'slice'))
        uV_scale = float(attrs.get('uV_scale', 100.0))
        leadfield_seed = int(attrs.get('leadfield_seed', 0)) if 'leadfield_seed' in attrs else 0
        if 'leadfield' in f:
            leadfield = np.asarray(f['leadfield'], dtype=np.float32)
        else:
            leadfield, *_ = make_leadfield(fs=fs, n_sources=n_sources, seed=leadfield_seed)
            leadfield = np.asarray(leadfield, dtype=np.float32)
    settings = ReplaySettings(fs=fs, duration=duration, n_channels=n_channels, stim_onset=stim_onset, stim_sigma=stim_sigma, stim_causal=stim_causal, n_sources=n_sources, n_trials=n_trials, input_noise_std=input_noise_std, sensor_noise_std=sensor_noise_std, internal_fs=internal_fs, bandpass=(band_lo, band_hi), baseline_correct=baseline_correct, baseline_window=None, warmup_sec=warmup_sec, downsample_method=downsample_method, uV_scale=uV_scale, leadfield_seed=leadfield_seed)
    meta = dict(attrs)
    meta['stim_causal_effective'] = bool(stim_causal)
    meta['stim_causal_source'] = stim_causal_from
    meta['source_h5'] = str(source_h5)
    return settings, leadfield, meta


def _load_prior_spec(data_out: Path, source_h5: Path) -> Tuple[List[str], np.ndarray, np.ndarray, List[str], np.ndarray, str]:
    """Load the Monte Carlo proposal prior from the prepared dataset when possible.

    Why this order matters:
      - data_out/param_meta.npz is written by prepare_training_data.py from the H5 that
        actually produced the current params.npy used by training/evaluation.
      - source_h5 may be stale or overwritten even when data_out is current.
      - falling back to the H5 is still useful for older folders without param_meta.npz.
    """
    meta_path = data_out / 'param_meta.npz'
    if meta_path.exists():
        z = np.load(meta_path, allow_pickle=True)
        needed = {'param_names', 'prior_low', 'prior_high', 'prior_dist', 'prior_params'}
        if needed.issubset(set(z.files)):
            names = _decode_list(np.asarray(z['param_names']))
            low = np.asarray(z['prior_low'], dtype=np.float32).reshape(-1)
            high = np.asarray(z['prior_high'], dtype=np.float32).reshape(-1)
            dist = _decode_list(np.asarray(z['prior_dist']))
            prior_params = np.asarray(z['prior_params'], dtype=np.float32)
            return names, low, high, dist, prior_params, 'data_out:param_meta.npz'

    with h5py.File(source_h5, 'r') as f:
        if all(k in f for k in ('param_names', 'prior_low', 'prior_high', 'prior_dist', 'prior_params')):
            names = _decode_list(np.asarray(f['param_names']))
            low = np.asarray(f['prior_low'], dtype=np.float32).reshape(-1)
            high = np.asarray(f['prior_high'], dtype=np.float32).reshape(-1)
            dist = _decode_list(np.asarray(f['prior_dist']))
            prior_params = np.asarray(f['prior_params'], dtype=np.float32)
            return names, low, high, dist, prior_params, 'source_h5'

    names, low, high, dist, prior_params, _ = build_prior_spec()
    return list(names), np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32), list(dist), np.asarray(prior_params, dtype=np.float32), 'code_default'


def _sample_prior_theta_and_seeds(n: int, seed: int, param_names: Sequence[str], low: np.ndarray, high: np.ndarray, dist: Sequence[str], prior_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = np.zeros((int(n), len(param_names)), dtype=np.float32)
    sim_seed = np.zeros((int(n),), dtype=np.int64)
    for i in range(int(n)):
        _th_unused, params = sample_theta(rng, list(param_names), low, high, list(dist), prior_params)
        theta[i] = np.asarray([float(params[nm]) for nm in param_names], dtype=np.float32)
        sim_seed[i] = int(rng.integers(0, np.iinfo(np.int32).max))
    return theta, sim_seed


def _load_dataset_theta(data_out: Path) -> np.ndarray:
    theta = np.asarray(np.load(data_out / 'params.npy'), dtype=np.float32)
    if theta.ndim != 2:
        raise ValueError(theta.shape)
    return theta


def _load_source_h5_from_data_out(data_out: Path, explicit_source_h5: Optional[str]) -> Path:
    if explicit_source_h5:
        p = Path(explicit_source_h5).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    log = _load_json(data_out / 'prepare_training_data_log.json')
    rel = log.get('in_h5')
    if not rel:
        raise KeyError('in_h5 missing')
    resolved = _resolve_relative_to_data_out(data_out, str(rel))
    if resolved is None:
        raise FileNotFoundError(rel)
    return resolved


def _set_plot_defaults() -> None:
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9})


def _js_divergence_from_counts(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(a, dtype=np.float64).reshape(-1)
    q = np.asarray(b, dtype=np.float64).reshape(-1)
    if np.sum(p) <= 0 and np.sum(q) <= 0:
        return 0.0
    p = p + eps
    q = q + eps
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))


def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1)); y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if x.size == 0 or y.size == 0:
        return float('nan')
    grid = np.unique(np.concatenate([x, y]))
    if grid.size < 2:
        return 0.0
    cdf_x = np.searchsorted(x, grid, side='right') / float(x.size)
    cdf_y = np.searchsorted(y, grid, side='right') / float(y.size)
    return float(np.sum(np.abs(cdf_x[:-1] - cdf_y[:-1]) * np.diff(grid)))


def _ks_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1)); y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if x.size == 0 or y.size == 0:
        return float('nan')
    grid = np.unique(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side='right') / float(x.size)
    cdf_y = np.searchsorted(y, grid, side='right') / float(y.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _accept_one(theta: np.ndarray, sim_seed: int, settings: ReplaySettings, param_names: Sequence[str], leadfield: np.ndarray, rf_cfg: RegimeFilterConfig) -> Tuple[bool, str]:
    params = theta_to_params(np.asarray(theta, dtype=np.float32), list(param_names))
    eeg = simulate_eeg(params=params, fs=int(settings.fs), duration=float(settings.duration), n_channels=int(settings.n_channels), seed=int(sim_seed), bandpass=tuple(float(v) for v in settings.bandpass), stim_onset=float(settings.stim_onset), stim_sigma=float(settings.stim_sigma), stim_causal=bool(settings.stim_causal), n_sources=int(settings.n_sources), leadfield=leadfield, sensor_noise_std=float(settings.sensor_noise_std), n_trials=int(settings.n_trials), input_noise_std=float(settings.input_noise_std), internal_fs=int(settings.internal_fs), baseline_correct=bool(settings.baseline_correct), baseline_window=settings.baseline_window, warmup_sec=float(settings.warmup_sec), downsample_method=str(settings.downsample_method), uV_scale=float(settings.uV_scale), return_trials=False, return_sources=False)
    ok, reason = regime_reject(np.asarray(eeg, dtype=np.float32), rf_cfg)
    return bool(ok), str(reason)


def _evaluate_acceptance(theta: np.ndarray, sim_seeds: np.ndarray, settings: ReplaySettings, param_names: Sequence[str], leadfield: np.ndarray, rf_cfg: RegimeFilterConfig, n_jobs: int) -> Tuple[np.ndarray, List[str]]:
    theta = np.asarray(theta, dtype=np.float32); sim_seeds = np.asarray(sim_seeds, dtype=np.int64).reshape(-1)
    def _run(i: int):
        return _accept_one(theta[i], int(sim_seeds[i]), settings, param_names, leadfield, rf_cfg)
    if int(n_jobs) > 1 and Parallel is not None and delayed is not None:
        out = Parallel(n_jobs=int(n_jobs), verbose=5)(delayed(_run)(i) for i in range(theta.shape[0]))
    else:
        out = [_run(i) for i in range(theta.shape[0])]
    return np.asarray([x[0] for x in out], dtype=bool), [x[1] for x in out]


def _maybe_load_h5_theta(source_h5: Path) -> Optional[np.ndarray]:
    with h5py.File(source_h5, 'r') as f:
        if 'theta' in f:
            return np.asarray(f['theta'], dtype=np.float32)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='sensitivity/config_sensitivity.yaml')
    ap.add_argument('--data-out', type=str, required=True)
    ap.add_argument('--source-h5', type=str, default=None)
    ap.add_argument('--out-dir', type=str, default='plots/acceptance_prior_analysis_v2')
    ap.add_argument('--n-proposals', type=int, default=12000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--n-bins', type=int, default=24)
    ap.add_argument('--pair-bins', type=int, default=18)
    ap.add_argument('--top-pairs', type=int, default=4)
    ap.add_argument('--min-proposals-per-bin', type=int, default=25)
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()
    _set_plot_defaults()
    data_out = (Path.cwd() / args.data_out).resolve() if not Path(args.data_out).is_absolute() else Path(args.data_out)
    out_dir = (Path.cwd() / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_h5 = _load_source_h5_from_data_out(data_out, args.source_h5)
    settings, leadfield, source_meta = _settings_from_h5(source_h5)
    rf_cfg = settings.regime_filter_config()
    param_names, low, high, dist, prior_params, prior_source = _load_prior_spec(data_out, source_h5)
    theta_dataset = _load_dataset_theta(data_out)
    h5_theta = _maybe_load_h5_theta(source_h5)
    dataset_matches_h5 = bool(h5_theta is not None and h5_theta.shape == theta_dataset.shape and np.allclose(h5_theta, theta_dataset))
    theta_prop, sim_seed_prop = _sample_prior_theta_and_seeds(int(args.n_proposals), int(args.seed), param_names, low, high, dist, prior_params)
    accepted_mask, reasons = _evaluate_acceptance(theta_prop, sim_seed_prop, settings, param_names, leadfield, rf_cfg, int(args.n_jobs))
    theta_acc_mc = theta_prop[accepted_mask]
    sim_seed_acc_mc = sim_seed_prop[accepted_mask]
    if theta_acc_mc.shape[0] == 0:
        raise SystemExit('zero accepted MC proposals')
    reason_counts = {}
    for r in reasons:
        reason_counts[r] = reason_counts.get(r, 0) + 1
    acc_rate = float(np.mean(accepted_mask))
    np.savez_compressed(out_dir / 'acceptance_effective_prior_samples.npz', proposal_theta=theta_prop.astype(np.float32), proposal_sim_seed=sim_seed_prop.astype(np.int64), accepted_mask=accepted_mask.astype(bool), accepted_theta_mc=theta_acc_mc.astype(np.float32), accepted_sim_seed_mc=sim_seed_acc_mc.astype(np.int64), dataset_theta=theta_dataset.astype(np.float32), rejection_reason=np.asarray([r.encode('utf-8') for r in reasons], dtype='S'), param_names=np.asarray([n.encode('utf-8') for n in param_names], dtype='S'), low=np.asarray(low, dtype=np.float32), high=np.asarray(high, dtype=np.float32), source_h5=np.asarray(str(source_h5).encode('utf-8'), dtype='S'))
    prange = np.maximum(np.asarray(high, dtype=np.float64) - np.asarray(low, dtype=np.float64), 1e-12)
    marginal_rows=[]; validation_rows=[]
    for j,name in enumerate(param_names):
        bins = np.linspace(float(low[j]), float(high[j]), int(args.n_bins)+1)
        c_prop,_=np.histogram(theta_prop[:,j], bins=bins); c_acc,_=np.histogram(theta_acc_mc[:,j], bins=bins); c_data,_=np.histogram(theta_dataset[:,j], bins=bins)
        marginal_rows.append({'param':name,'proposed_mean':float(np.mean(theta_prop[:,j])),'accepted_mean':float(np.mean(theta_acc_mc[:,j])),'mean_shift':float(np.mean(theta_acc_mc[:,j])-np.mean(theta_prop[:,j])),'mean_shift_over_range':float((np.mean(theta_acc_mc[:,j])-np.mean(theta_prop[:,j]))/prange[j]),'proposed_q05':float(np.quantile(theta_prop[:,j],0.05)),'accepted_q05':float(np.quantile(theta_acc_mc[:,j],0.05)),'proposed_q50':float(np.quantile(theta_prop[:,j],0.50)),'accepted_q50':float(np.quantile(theta_acc_mc[:,j],0.50)),'proposed_q95':float(np.quantile(theta_prop[:,j],0.95)),'accepted_q95':float(np.quantile(theta_acc_mc[:,j],0.95)),'wasserstein_over_range':float(_wasserstein_1d(theta_prop[:,j], theta_acc_mc[:,j])/prange[j]),'ks_distance':float(_ks_distance(theta_prop[:,j], theta_acc_mc[:,j])),'js_divergence_bits':float(_js_divergence_from_counts(c_prop, c_acc))})
        validation_rows.append({'param':name,'dataset_vs_mc_wasserstein_over_range':float(_wasserstein_1d(theta_dataset[:,j], theta_acc_mc[:,j])/prange[j]),'dataset_vs_mc_ks_distance':float(_ks_distance(theta_dataset[:,j], theta_acc_mc[:,j])),'dataset_vs_mc_js_divergence_bits':float(_js_divergence_from_counts(c_data, c_acc))})
    with (out_dir/'acceptance_prior_marginal_distortion.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(marginal_rows[0].keys())); w.writeheader(); w.writerows(marginal_rows)
    with (out_dir/'acceptance_prior_dataset_validation.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(validation_rows[0].keys())); w.writeheader(); w.writerows(validation_rows)
    pair_rows=[]
    for a in range(len(param_names)):
        for b in range(a+1, len(param_names)):
            bins_x = np.linspace(float(low[a]), float(high[a]), int(args.pair_bins)+1)
            bins_y = np.linspace(float(low[b]), float(high[b]), int(args.pair_bins)+1)
            prop_counts, xedges, yedges = np.histogram2d(theta_prop[:,a], theta_prop[:,b], bins=[bins_x,bins_y])
            acc_counts, _xe, _ye = np.histogram2d(theta_acc_mc[:,a], theta_acc_mc[:,b], bins=[bins_x,bins_y])
            local_rate = np.divide(acc_counts, prop_counts, out=np.full_like(acc_counts, np.nan, dtype=np.float64), where=prop_counts>=float(args.min_proposals_per_bin))
            prop_mass = prop_counts / max(1.0, np.sum(prop_counts)); score = float(np.nansum(prop_mass * (local_rate - acc_rate)**2)); finite=np.isfinite(local_rate)
            pair_rows.append({'param_x':param_names[a],'param_y':param_names[b],'score_weighted_rate_deviation':score,'js_divergence_bits':float(_js_divergence_from_counts(prop_counts, acc_counts)),'n_valid_bins':int(np.sum(finite)),'local_acceptance_min':float(np.nanmin(local_rate)) if np.any(finite) else float('nan'),'local_acceptance_median':float(np.nanmedian(local_rate)) if np.any(finite) else float('nan'),'local_acceptance_max':float(np.nanmax(local_rate)) if np.any(finite) else float('nan')})
    pair_rows.sort(key=lambda r: float(r['score_weighted_rate_deviation']), reverse=True)
    with (out_dir/'acceptance_prior_pairwise_distortion.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(pair_rows[0].keys())); w.writeheader(); w.writerows(pair_rows)
    ncols=3; nrows=int(math.ceil(len(param_names)/ncols)); fig, axes = plt.subplots(nrows,ncols,figsize=(13.5,3.8*nrows), squeeze=False); axes=axes.reshape(-1)
    for j,name in enumerate(param_names):
        ax=axes[j]; bins=np.linspace(float(low[j]), float(high[j]), int(args.n_bins)+1); ax.hist(theta_prop[:,j], bins=bins, density=True, histtype='step', linewidth=1.8, label='proposed prior'); ax.hist(theta_acc_mc[:,j], bins=bins, density=True, histtype='step', linewidth=1.8, label='estimated accepted effective prior'); ax.set_title(name); ax.set_xlim(float(low[j]), float(high[j])); ax.grid(True, axis='y', alpha=0.25)
    for ax in axes[len(param_names):]: ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False); fig.suptitle('Proposed prior vs accepted effective prior', y=0.995); fig.tight_layout(rect=[0.0,0.0,1.0,0.97]); fig.savefig(out_dir/'acceptance_prior_marginals_v2.png', dpi=int(args.dpi)); fig.savefig(out_dir/'acceptance_prior_marginals_v2.pdf'); plt.close(fig)
    top_pairs = pair_rows[:min(int(args.top_pairs), len(pair_rows))]
    fig, axes = plt.subplots(1, len(top_pairs), figsize=(5.0*max(1,len(top_pairs)), 4.2), squeeze=False); axes=axes.reshape(-1)
    for ax,row in zip(axes, top_pairs):
        a=param_names.index(str(row['param_x'])); b=param_names.index(str(row['param_y'])); bins_x=np.linspace(float(low[a]), float(high[a]), int(args.pair_bins)+1); bins_y=np.linspace(float(low[b]), float(high[b]), int(args.pair_bins)+1); prop_counts,xedges,yedges=np.histogram2d(theta_prop[:,a], theta_prop[:,b], bins=[bins_x,bins_y]); acc_counts,_xe,_ye=np.histogram2d(theta_acc_mc[:,a], theta_acc_mc[:,b], bins=[bins_x,bins_y]); local_rate=np.divide(acc_counts,prop_counts,out=np.full_like(acc_counts,np.nan,dtype=np.float64),where=prop_counts>=float(args.min_proposals_per_bin)); cmap=plt.get_cmap('viridis').copy(); cmap.set_bad(color='#f0f0f0'); im=ax.imshow(local_rate.T, origin='lower', aspect='auto', extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]], vmin=0.0, vmax=1.0, cmap=cmap); ax.set_xlabel(str(row['param_x'])); ax.set_ylabel(str(row['param_y'])); ax.set_title(f"{row['param_x']} vs {row['param_y']}\nscore={row['score_weighted_rate_deviation']:.4f}"); cbar=fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('local acceptance rate')
    fig.suptitle('Top pairwise regions most distorted by acceptance filter', y=1.02); fig.tight_layout(); fig.savefig(out_dir/'acceptance_prior_pairwise_top_v2.png', dpi=int(args.dpi), bbox_inches='tight'); fig.savefig(out_dir/'acceptance_prior_pairwise_top_v2.pdf', bbox_inches='tight'); plt.close(fig)
    top_marginals = sorted(marginal_rows, key=lambda r: float(r['wasserstein_over_range']), reverse=True)[:5]; top_valid = sorted(validation_rows, key=lambda r: float(r['dataset_vs_mc_wasserstein_over_range']), reverse=True)[:5]
    md=[]; md.append('# Acceptance regime diagnostics (v2)\n'); md.append(f'- Prepared dataset: **{data_out}**.'); md.append(f'- Source H5 inferred from provenance: **{source_h5}**.'); md.append(f'- Prior specification source used for Monte Carlo: **{prior_source}**.'); md.append(f'- Dataset params match source H5 theta: **{dataset_matches_h5}**.'); md.append(f'- Stimulus causal flag used in replay: **{settings.stim_causal}** ({source_meta.get("stim_causal_source","unknown")}).'); md.append(f'- Proposed prior draws: **{theta_prop.shape[0]}**.'); md.append(f'- Accepted effective-prior draws (Monte Carlo estimate): **{theta_acc_mc.shape[0]}**.'); md.append(f'- Empirical acceptance rate: **{100.0*acc_rate:.2f}%**.'); md.append(f'- Rejection rate: **{100.0*(1.0-acc_rate):.2f}%**.'); md.append(f'- Rejection reason counts: `{json.dumps(reason_counts, sort_keys=True)}`.');
    if 'accept_rate' in source_meta: md.append(f'- Source H5 stored acceptance-rate attr: **{source_meta["accept_rate"]}**.');
    if 'n_attempts_total' in source_meta: md.append(f'- Source H5 stored attempts attr: **{source_meta["n_attempts_total"]}**.');
    if 'tries' in source_meta: md.append(f'- Source H5 stored tries attr: **{source_meta["tries"]}**.');
    md.append('\n## Largest marginal distortions (proposed prior -> accepted effective prior)\n')
    for row in top_marginals: md.append(f"- **{row['param']}**: mean shift/range = {row['mean_shift_over_range']:.3f}, Wasserstein/range = {row['wasserstein_over_range']:.3f}, KS = {row['ks_distance']:.3f}, JS = {row['js_divergence_bits']:.3f} bits.")
    md.append('\n## Most distorted parameter pairs\n')
    for row in top_pairs: md.append(f"- **({row['param_x']}, {row['param_y']})**: weighted acceptance-rate deviation score = {row['score_weighted_rate_deviation']:.4f}; local acceptance spans {row['local_acceptance_min']:.2f} to {row['local_acceptance_max']:.2f} across adequately sampled bins.")
    md.append('\n## Dataset vs Monte Carlo effective-prior validation\n')
    for row in top_valid: md.append(f"- **{row['param']}**: dataset-vs-MC Wasserstein/range = {row['dataset_vs_mc_wasserstein_over_range']:.3f}, KS = {row['dataset_vs_mc_ks_distance']:.3f}, JS = {row['dataset_vs_mc_js_divergence_bits']:.3f} bits.")
    (out_dir/'acceptance_prior_summary_v2.md').write_text('\n'.join(md), encoding='utf-8')
    manifest={'config':args.config,'data_out':str(data_out),'source_h5':str(source_h5),'prior_source':prior_source,'n_proposals':int(args.n_proposals),'seed':int(args.seed),'acceptance_rate':acc_rate,'rejection_reason_counts':reason_counts,'source_meta':source_meta,'outputs':['acceptance_effective_prior_samples.npz','acceptance_prior_marginal_distortion.csv','acceptance_prior_pairwise_distortion.csv','acceptance_prior_dataset_validation.csv','acceptance_prior_marginals_v2.png','acceptance_prior_marginals_v2.pdf','acceptance_prior_pairwise_top_v2.png','acceptance_prior_pairwise_top_v2.pdf','acceptance_prior_summary_v2.md']}
    (out_dir/'acceptance_prior_manifest_v2.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'analysis':'acceptance_prior_v2','data_out':str(data_out),'source_h5':str(source_h5),'out_dir':str(out_dir),'acceptance_rate':acc_rate,'n_accepted_mc':int(theta_acc_mc.shape[0])}, indent=2))

if __name__ == '__main__':
    main()
