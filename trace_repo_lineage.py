#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py  # type: ignore
except Exception:
    h5py = None


def _safe_load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _decode_bytes(x: Any) -> Any:
    if isinstance(x, (bytes, np.bytes_)):
        try:
            return x.decode('utf-8')
        except Exception:
            return repr(x)
    return x


def _npz_scalar(npz: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in npz.files:
        return None
    arr = npz[key]
    if getattr(arr, 'ndim', 0) == 0:
        try:
            return _decode_bytes(arr.item())
        except Exception:
            return _decode_bytes(arr)
    return arr


def _sha1_of_int_array(x: np.ndarray) -> str:
    arr = np.asarray(x)
    if arr.dtype.kind not in ('i', 'u'):
        arr = arr.astype(np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def _file_mtime(path: Path) -> str:
    try:
        return str(path.stat().st_mtime_ns)
    except Exception:
        return 'NA'


def _load_h5_summary(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'path': str(path),
        'mtime_ns': _file_mtime(path),
        'size_mb': round(path.stat().st_size / (1024 * 1024), 2),
    }
    if h5py is None:
        out['error'] = 'h5py_not_available'
        return out
    try:
        with h5py.File(path, 'r') as f:
            out['datasets'] = sorted(list(f.keys()))
            out['shape_eeg'] = tuple(int(v) for v in f['eeg'].shape) if 'eeg' in f else None
            out['shape_theta'] = tuple(int(v) for v in f['theta'].shape) if 'theta' in f else None
            attrs = {}
            for k in [
                'fs', 'duration_sec', 'n_channels', 'stim_onset_sec', 'stim_sigma_sec',
                'warmup_sec', 'bandpass_lo_hz', 'bandpass_hi_hz', 'n_sources', 'n_trials',
                'input_noise_std', 'sensor_noise_std', 'internal_fs', 'baseline_correct',
                'downsample_method', 'uV_scale', 'stim_causal', 'generator_seed', 'leadfield_seed',
                'n_attempts_total', 'accept_rate'
            ]:
                if k in f.attrs:
                    v = f.attrs[k]
                    try:
                        attrs[k] = _decode_bytes(v.item() if hasattr(v, 'item') else v)
                    except Exception:
                        attrs[k] = _decode_bytes(v)
            out['attrs'] = attrs
            if 'param_names' in f:
                names = [_decode_bytes(x) for x in f['param_names'][()]]
                out['param_names'] = names
    except Exception as e:
        out['error'] = repr(e)
    return out


def _score_h5_against_manuscript(h5: Dict[str, Any]) -> int:
    attrs = h5.get('attrs', {}) or {}
    score = 0
    want = {
        'fs': 250,
        'duration_sec': 2.0,
        'stim_onset_sec': 0.5,
        'stim_sigma_sec': 0.05,
        'warmup_sec': 3.0,
        'bandpass_lo_hz': 0.5,
        'bandpass_hi_hz': 40.0,
        'n_sources': 3,
        'n_trials': 10,
        'input_noise_std': 0.2,
        'sensor_noise_std': 2.0,
        'internal_fs': 1000,
        'baseline_correct': 1,
        'uV_scale': 100.0,
        'stim_causal': 1,
    }
    for k, v in want.items():
        if k in attrs:
            try:
                if abs(float(attrs[k]) - float(v)) < 1e-9:
                    score += 1
            except Exception:
                if attrs[k] == v:
                    score += 1
    return score


def _load_data_out_summary(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {'path': str(path), 'mtime_ns': _file_mtime(path)}
    params = path / 'params.npy'
    feat = path / 'features.npy'
    erp = path / 'features_erp.npy'
    tfr = path / 'features_tfr.npy'
    out['has_params'] = params.exists()
    out['has_features'] = feat.exists()
    out['has_features_erp'] = erp.exists()
    out['has_features_tfr'] = tfr.exists()
    out['has_prepare_log'] = (path / 'prepare_training_data_log.json').exists()
    out['has_generation_log'] = (path / 'generation_log.json').exists()
    out['has_param_meta'] = (path / 'param_meta.npz').exists()
    out['has_tfr_meta'] = (path / 'tfr_meta.npz').exists()
    out['has_splits'] = (path / 'splits.npz').exists()

    if params.exists():
        try:
            arr = np.load(params, mmap_mode='r')
            out['params_shape'] = tuple(int(v) for v in arr.shape)
        except Exception as e:
            out['params_shape_error'] = repr(e)
    if feat.exists():
        try:
            arr = np.load(feat, mmap_mode='r')
            out['features_shape'] = tuple(int(v) for v in arr.shape)
        except Exception as e:
            out['features_shape_error'] = repr(e)

    plog = path / 'prepare_training_data_log.json'
    if plog.exists():
        j = _safe_load_json(plog)
        out['prepare_log'] = j

    glog = path / 'generation_log.json'
    if glog.exists():
        j = _safe_load_json(glog)
        out['generation_log'] = j

    pmeta = path / 'param_meta.npz'
    if pmeta.exists():
        try:
            z = np.load(pmeta, allow_pickle=True)
            out['param_names'] = [_decode_bytes(x) for x in z['param_names']]
            out['prior_low'] = np.asarray(z['prior_low']).tolist()
            out['prior_high'] = np.asarray(z['prior_high']).tolist()
        except Exception as e:
            out['param_meta_error'] = repr(e)

    tmeta = path / 'tfr_meta.npz'
    if tmeta.exists():
        try:
            z = np.load(tmeta, allow_pickle=True)
            keys = [
                'fs', 'duration', 'n_channels', 'stim_onset', 'stim_sigma', 'warmup_sec',
                'n_time_patches', 'n_freq_patches', 'n_tokens_erp', 'n_tokens_tfr', 'n_tokens_total',
                'f_min', 'f_max', 'tfr_backend', 'tfr_method', 'morlet_n_freqs',
                'morlet_cycles_low', 'morlet_cycles_high', 'morlet_decim', 'morlet_n_jobs'
            ]
            meta = {}
            for k in keys:
                if k in z.files:
                    v = _npz_scalar(z, k)
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    meta[k] = v
            out['tfr_meta'] = meta
        except Exception as e:
            out['tfr_meta_error'] = repr(e)

    splits = path / 'splits.npz'
    if splits.exists():
        try:
            z = np.load(splits, allow_pickle=True)
            meta = {}
            for k in ['seed', 'train_frac', 'val_frac', 'test_frac', 'n_samples']:
                if k in z.files:
                    v = _npz_scalar(z, k)
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    meta[k] = v
            if 'train_idx' in z.files:
                meta['train_n'] = int(len(z['train_idx']))
                meta['train_hash'] = _sha1_of_int_array(z['train_idx'])
            if 'val_idx' in z.files:
                meta['val_n'] = int(len(z['val_idx']))
                meta['val_hash'] = _sha1_of_int_array(z['val_idx'])
            if 'test_idx' in z.files:
                meta['test_n'] = int(len(z['test_idx']))
                meta['test_hash'] = _sha1_of_int_array(z['test_idx'])
            out['splits'] = meta
        except Exception as e:
            out['splits_error'] = repr(e)
    return out


def _score_data_out_against_manuscript(d: Dict[str, Any]) -> int:
    score = 0
    t = d.get('tfr_meta', {}) or {}
    want = {
        'fs': 250,
        'duration': 2.0,
        'stim_onset': 0.5,
        'stim_sigma': 0.05,
        'warmup_sec': 3.0,
        'n_time_patches': 25,
        'n_freq_patches': 15,
        'f_min': 4.0,
        'f_max': 40.0,
        'tfr_backend': 'morlet',
        'morlet_n_freqs': 48,
        'morlet_cycles_low': 4.0,
        'morlet_cycles_high': 8.0,
        'morlet_decim': 1,
        'morlet_n_jobs': 1,
    }
    for k, v in want.items():
        if k not in t:
            continue
        x = t[k]
        try:
            if isinstance(v, str):
                if str(x).strip("b'") == v:
                    score += 1
            elif abs(float(x) - float(v)) < 1e-9:
                score += 1
        except Exception:
            pass
    s = d.get('splits', {}) or {}
    if s.get('seed') == 42:
        score += 1
    if s.get('train_n') == 7000 and s.get('val_n') == 1500 and s.get('test_n') == 1500:
        score += 1
    if d.get('params_shape') == (10000, 9):
        score += 1
    if d.get('features_shape') == (10000, 400, 16):
        score += 1
    return score


def _load_acceptance_manifest(path: Path) -> Dict[str, Any]:
    out = {'path': str(path), 'mtime_ns': _file_mtime(path)}
    j = _safe_load_json(path)
    if j is None:
        out['error'] = 'json_parse_failed'
        return out
    out.update(j)
    return out


def _load_model_dir_summary(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {'path': str(path), 'mtime_ns': _file_mtime(path)}
    keras_files = sorted([p.name for p in path.glob('*.keras')])
    if not keras_files and not (path / 'scaler_stats.npz').exists() and not (path / 'split_indices_used.npz').exists():
        return out
    out['keras_files'] = keras_files
    out['has_scaler_stats'] = (path / 'scaler_stats.npz').exists()
    out['has_scaler_pkl'] = (path / 'scaler.pkl').exists()
    out['has_split_indices_used'] = (path / 'split_indices_used.npz').exists()
    if (path / 'split_indices_used.npz').exists():
        try:
            z = np.load(path / 'split_indices_used.npz', allow_pickle=True)
            meta = {}
            for k in ['train_idx', 'val_idx', 'test_idx']:
                if k in z.files:
                    meta[f'{k}_n'] = int(len(z[k]))
                    meta[f'{k}_hash'] = _sha1_of_int_array(z[k])
            out['split_indices_used'] = meta
        except Exception as e:
            out['split_indices_error'] = repr(e)
    if (path / 'scaler_stats.npz').exists():
        try:
            z = np.load(path / 'scaler_stats.npz', allow_pickle=True)
            if 'mu' in z.files and 'sd' in z.files:
                out['scaler_shape'] = (tuple(int(v) for v in z['mu'].shape), tuple(int(v) for v in z['sd'].shape))
            elif 'mean' in z.files and 'std' in z.files:
                out['scaler_shape'] = (tuple(int(v) for v in z['mean'].shape), tuple(int(v) for v in z['std'].shape))
        except Exception as e:
            out['scaler_error'] = repr(e)
    return out


def _find_candidate_data_out_dirs(root: Path) -> List[Path]:
    out = []
    for p in root.rglob('params.npy'):
        parent = p.parent
        if parent not in out:
            out.append(parent)
    return sorted(out)


def _find_candidate_model_dirs(root: Path) -> List[Path]:
    seen = set()
    out = []
    for p in root.rglob('*.keras'):
        parent = p.parent
        if parent not in seen:
            seen.add(parent)
            out.append(parent)
    for p in root.rglob('split_indices_used.npz'):
        parent = p.parent
        if parent not in seen:
            seen.add(parent)
            out.append(parent)
    for p in root.rglob('scaler_stats.npz'):
        parent = p.parent
        if parent not in seen:
            seen.add(parent)
            out.append(parent)
    return sorted(out)


def _match_models_to_data_out(models: List[Dict[str, Any]], data_outs: List[Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
    # Match by split hash if possible.
    d_hash_to_path: Dict[Tuple[str, str, str], str] = {}
    for d in data_outs:
        s = d.get('splits', {}) or {}
        key = (str(s.get('train_hash')), str(s.get('val_hash')), str(s.get('test_hash')))
        if all(k != 'None' for k in key):
            d_hash_to_path[key] = d['path']
    out = []
    for m in models:
        matches: List[str] = []
        s = m.get('split_indices_used', {}) or {}
        key = (str(s.get('train_idx_hash')), str(s.get('val_idx_hash')), str(s.get('test_idx_hash')))
        # fall back to custom names saved above
        key = (str(s.get('train_idx_hash', s.get('train_hash'))), str(s.get('val_idx_hash', s.get('val_hash'))), str(s.get('test_idx_hash', s.get('test_hash'))))
        if all(k != 'None' for k in key) and key in d_hash_to_path:
            matches.append(d_hash_to_path[key])
        out.append((m['path'], matches))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Trace likely active dataset/model lineage in the repo.')
    ap.add_argument('root', nargs='?', default='.', help='Repo root to scan')
    ap.add_argument('--json-out', default='', help='Optional path to write machine-readable summary JSON')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    h5s = sorted(root.rglob('*.h5'))
    data_out_dirs = _find_candidate_data_out_dirs(root)
    manifests = sorted([*root.rglob('acceptance_prior_manifest*.json'), *root.rglob('prepare_training_data_log.json')])
    model_dirs = _find_candidate_model_dirs(root)

    h5_summ = [_load_h5_summary(p) for p in h5s]
    data_summ = [_load_data_out_summary(p) for p in data_out_dirs]
    man_summ = [_load_acceptance_manifest(p) for p in manifests if 'acceptance_prior_manifest' in p.name]
    prep_logs = [_safe_load_json(p) for p in manifests if p.name == 'prepare_training_data_log.json']
    model_summ = [_load_model_dir_summary(p) for p in model_dirs]
    model_summ = [m for m in model_summ if m.get('keras_files') or m.get('has_split_indices_used') or m.get('has_scaler_stats') or m.get('has_scaler_pkl')]

    for x in h5_summ:
        x['manuscript_score'] = _score_h5_against_manuscript(x)
    for x in data_summ:
        x['manuscript_score'] = _score_data_out_against_manuscript(x)

    matches = _match_models_to_data_out(model_summ, data_summ)

    print('\n=== CANDIDATE H5 DATASETS ===')
    if not h5_summ:
        print('(none found)')
    for h in sorted(h5_summ, key=lambda d: (d.get('manuscript_score', 0), d.get('mtime_ns', '0')), reverse=True):
        print(f"- {h['path']}")
        print(f"  size_mb={h.get('size_mb')} manuscript_score={h.get('manuscript_score')} mtime_ns={h.get('mtime_ns')}")
        print(f"  shape_theta={h.get('shape_theta')} shape_eeg={h.get('shape_eeg')}")
        print(f"  attrs={json.dumps(h.get('attrs', {}), sort_keys=True)}")
        if h.get('param_names'):
            print(f"  param_names={h['param_names']}")
        if h.get('error'):
            print(f"  ERROR={h['error']}")

    print('\n=== CANDIDATE data_out DIRS ===')
    if not data_summ:
        print('(none found)')
    for d in sorted(data_summ, key=lambda x: (x.get('manuscript_score', 0), x.get('mtime_ns', '0')), reverse=True):
        print(f"- {d['path']}")
        print(f"  manuscript_score={d.get('manuscript_score')} mtime_ns={d.get('mtime_ns')}")
        print(f"  params_shape={d.get('params_shape')} features_shape={d.get('features_shape')}")
        print(f"  has_prepare_log={d.get('has_prepare_log')} has_generation_log={d.get('has_generation_log')} has_tfr_meta={d.get('has_tfr_meta')} has_splits={d.get('has_splits')}")
        if d.get('prepare_log'):
            pl = d['prepare_log']
            print(f"  prepare_log.in_h5={pl.get('in_h5')} out_dir={pl.get('out_dir')}")
            cli = pl.get('cli', {}) or {}
            print(f"  prepare_log.cli={json.dumps(cli, sort_keys=True)}")
        if d.get('generation_log'):
            print(f"  generation_log.keys={sorted((d['generation_log'] or {}).keys())}")
        if d.get('tfr_meta'):
            print(f"  tfr_meta={json.dumps(d['tfr_meta'], sort_keys=True)}")
        if d.get('splits'):
            print(f"  splits={json.dumps(d['splits'], sort_keys=True)}")
        if d.get('param_names'):
            print(f"  param_names={d['param_names']}")

    print('\n=== ACCEPTANCE MANIFESTS ===')
    if not man_summ:
        print('(none found)')
    for m in sorted(man_summ, key=lambda x: x.get('mtime_ns', '0'), reverse=True):
        print(f"- {m['path']}")
        print(f"  data_out={m.get('data_out')} out_dir={m.get('out_dir')} seed={m.get('seed')} n_prior={m.get('n_prior', m.get('n_proposals'))}")

    print('\n=== PREPROCESS LOGS ===')
    if not prep_logs:
        print('(none found)')
    else:
        for j in prep_logs:
            if not j:
                continue
            print(f"- out_dir={j.get('out_dir')} in_h5={j.get('in_h5')} N={j.get('N')}")
            print(f"  cli={json.dumps(j.get('cli', {}), sort_keys=True)}")

    print('\n=== CANDIDATE MODEL DIRS ===')
    if not model_summ:
        print('(none found)')
    for m in sorted(model_summ, key=lambda x: x.get('mtime_ns', '0'), reverse=True):
        print(f"- {m['path']}")
        print(f"  keras_files={m.get('keras_files', [])}")
        print(f"  has_scaler_stats={m.get('has_scaler_stats')} has_scaler_pkl={m.get('has_scaler_pkl')} has_split_indices_used={m.get('has_split_indices_used')}")
        if m.get('scaler_shape'):
            print(f"  scaler_shape={m['scaler_shape']}")
        if m.get('split_indices_used'):
            print(f"  split_indices_used={json.dumps(m['split_indices_used'], sort_keys=True)}")

    print('\n=== MODEL -> data_out MATCHES (by split hash, if available) ===')
    any_match = False
    for model_path, ds in matches:
        print(f"- {model_path}")
        if ds:
            any_match = True
            for d in ds:
                print(f"  MATCH {d}")
        else:
            print('  (no split-hash match found)')
    if not any_match:
        print('(none found; this usually means the model dir does not contain split_indices_used.npz)')

    summary = {
        'root': str(root),
        'h5': h5_summ,
        'data_out': data_summ,
        'acceptance_manifests': man_summ,
        'model_dirs': model_summ,
        'model_to_data_out_matches': [
            {'model_dir': mp, 'matches': ds} for mp, ds in matches
        ],
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f"\nWrote JSON summary to {args.json_out}")


if __name__ == '__main__':
    main()
