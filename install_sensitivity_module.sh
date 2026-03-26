#!/usr/bin/env bash
set -euo pipefail
mkdir -p sensitivity
cat > sensitivity/common.py <<'EOF'
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REQUIRED_PARAM_NAMES: List[str] = [
    "tau_e",
    "tau_i",
    "g",
    "p0",
    "stim_amp",
    "w_ei",
    "w_ie",
    "w_ff",
    "w_fb",
]
REQUIRED_BOUNDS = np.asarray(
    [
        [0.005, 0.05],
        [0.003, 0.03],
        [0.5, 2.0],
        [0.05, 2.0],
        [0.1, 4.0],
        [0.2, 3.0],
        [-3.0, -0.2],
        [0.1, 2.5],
        [0.1, 2.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ParameterSpec:
    names: List[str]
    bounds: np.ndarray  # (P, 2)
    dist: List[str]
    prior_params: np.ndarray  # (P, 2)
    spec_json: str

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def lows(self) -> np.ndarray:
        return self.bounds[:, 0]

    @property
    def highs(self) -> np.ndarray:
        return self.bounds[:, 1]

    @property
    def ranges(self) -> np.ndarray:
        return self.highs - self.lows


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sensitivity_root(path: Optional[str | Path] = None) -> Path:
    if path is None:
        return repo_root() / "results_sensitivity"
    p = Path(path)
    if p.is_absolute():
        return p
    return repo_root() / p


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def array_sha1(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha1(arr.tobytes()).hexdigest()


def save_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def check_cache_manifest(path: Path, expected: Mapping[str, Any]) -> bool:
    existing = load_manifest(path)
    if existing is None:
        return False
    return existing == dict(expected)


def decode_bytes_array(x: np.ndarray) -> List[str]:
    arr = np.asarray(x)
    out: List[str] = []
    for item in arr.reshape(-1):
        if isinstance(item, (bytes, np.bytes_)):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def _default_config_path() -> Path:
    return Path(__file__).resolve().with_name("config_sensitivity.yaml")


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path is not None else _default_config_path()
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Sensitivity config not found: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    if yaml is None:
        raise ImportError("PyYAML is required to load config_sensitivity.yaml")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Sensitivity config must parse to a mapping")
    return data


def long_table_to_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            clean: Dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, np.generic):
                    value = value.item()
                clean[key] = value
            writer.writerow(clean)


def save_theta_csv(path: Path, theta: np.ndarray, names: Sequence[str]) -> None:
    rows = []
    theta = np.asarray(theta, dtype=np.float64)
    for i in range(theta.shape[0]):
        row = {str(name): float(theta[i, j]) for j, name in enumerate(names)}
        row["row"] = i
        rows.append(row)
    long_table_to_csv(path, rows, ["row", *list(names)])


def validate_project_parameter_spec() -> ParameterSpec:
    try:
        from data.priors import build_prior_spec
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not import data.priors.build_prior_spec. "
            "Create the sensitivity package inside the original repo so imports resolve."
        ) from exc

    names, low, high, dist, prior_params, spec_json = build_prior_spec()
    low = np.asarray(low, dtype=np.float64).reshape(-1)
    high = np.asarray(high, dtype=np.float64).reshape(-1)
    bounds = np.stack([low, high], axis=1)

    if list(names) != REQUIRED_PARAM_NAMES:
        raise ValueError(
            "Project parameter order does not match required order. "
            f"Expected {REQUIRED_PARAM_NAMES}, got {list(names)}"
        )
    if bounds.shape != REQUIRED_BOUNDS.shape or not np.allclose(bounds, REQUIRED_BOUNDS, atol=1e-12):
        raise ValueError(
            "Project parameter bounds do not match the manuscript bounds required for sensitivity analysis. "
            f"Expected {REQUIRED_BOUNDS.tolist()}, got {bounds.tolist()}"
        )

    return ParameterSpec(
        names=list(names),
        bounds=bounds.astype(np.float64),
        dist=list(dist),
        prior_params=np.asarray(prior_params, dtype=np.float64),
        spec_json=str(spec_json),
    )


def nanrank(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """
    Rank finite values from 1..K with 1=best. NaNs stay NaN.
    Ties are broken by stable order after sorting.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    ranks = np.full(x.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return ranks
    idx = np.where(finite)[0]
    vals = x[idx]
    order = np.argsort(-vals if higher_is_better else vals, kind="mergesort")
    ranks[idx[order]] = np.arange(1, order.size + 1, dtype=np.float64)
    return ranks


def discover_recoverability_csvs(search_root: Optional[Path] = None) -> List[Path]:
    root = repo_root() if search_root is None else Path(search_root)
    patterns = [
        "**/metrics_test.csv",
        "**/metrics_val.csv",
        "**/metrics_train.csv",
        "**/recoverability_table_test.csv",
        "**/recoverability_table_val.csv",
        "**/recoverability_table_train.csv",
    ]
    found: List[Path] = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    # Exclude sensitivity outputs themselves.
    found = [p for p in found if "results_sensitivity" not in p.parts and "reference" not in p.parts]
    # Prefer deterministic ordering.
    uniq = sorted({p.resolve() for p in found})
    return uniq


def fallback_recoverability_csvs() -> List[Path]:
    ref_dir = Path(__file__).resolve().with_name("reference")
    if not ref_dir.exists():
        return []
    return sorted(ref_dir.glob("*.csv"))
EOF
cat > sensitivity/forward_wrapper.py <<'EOF'
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import welch

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None

from data.feature_tokens import TokenConfig, compute_erp_tokens, compute_tfr_tokens, regime_reject
from data.priors import theta_to_params
from sim.leadfield_mne import DEFAULT_16_CH_NAMES, make_leadfield
from sim.simulate_eeg import simulate_eeg

from sensitivity.common import ParameterSpec, validate_project_parameter_spec


@dataclass(frozen=True)
class ForwardSettings:
    fs: int = 250
    duration: float = 2.0
    n_channels: int = 16
    stim_onset: float = 0.5
    stim_sigma: float = 0.05
    stim_causal: bool = False
    n_sources: int = 3
    n_trials: int = 10
    input_noise_std: float = 0.2
    sensor_noise_std: float = 2.0
    internal_fs: int = 1000
    bandpass: Tuple[float, float] = (0.5, 40.0)
    baseline_correct: bool = True
    baseline_window: Tuple[float, float] = (0.0, 0.5)
    warmup_sec: float = 3.0
    downsample_method: str = "slice"
    uV_scale: float = 100.0
    deterministic_seed: int = 314159
    post_window_sec: float = 0.60
    late_window_sec: float = 0.50
    tfr_method: str = "stft"
    tfr_f_min: float = 2.0
    tfr_f_max: float = 40.0
    n_time_patches: int = 25
    n_freq_patches: int = 15
    nperseg: int = 128
    noverlap: int = 112
    nfft: int = 256
    morlet_n_freqs: int = 48
    morlet_n_cycles_low: float = 3.0
    morlet_n_cycles_high: float = 10.0
    morlet_decim: int = 1
    morlet_n_jobs: int = 1
    bands: Mapping[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 12.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 40.0),
        }
    )

    def token_config(self) -> TokenConfig:
        return TokenConfig(
            fs=int(self.fs),
            duration=float(self.duration),
            n_channels=int(self.n_channels),
            n_time_patches=int(self.n_time_patches),
            n_freq_patches=int(self.n_freq_patches),
            stim_onset=float(self.stim_onset),
            f_min=float(self.tfr_f_min),
            f_max=float(self.tfr_f_max),
            nperseg=int(self.nperseg),
            noverlap=int(self.noverlap),
            nfft=int(self.nfft),
            tfr_method=str(self.tfr_method),
            morlet_n_freqs=int(self.morlet_n_freqs),
            morlet_n_cycles_low=float(self.morlet_n_cycles_low),
            morlet_n_cycles_high=float(self.morlet_n_cycles_high),
            morlet_decim=int(self.morlet_decim),
            morlet_n_jobs=int(self.morlet_n_jobs),
        )


@dataclass(frozen=True)
class ForwardResult:
    theta: np.ndarray
    accepted: bool
    rejection_reason: str
    error_message: str
    eeg: np.ndarray
    erp_tokens: np.ndarray
    tfr_tokens: np.ndarray
    hybrid_tokens: np.ndarray
    scalar_names: Tuple[str, ...]
    scalar_values: np.ndarray


def forward_settings_from_config(config: Mapping[str, Any]) -> ForwardSettings:
    sim = dict(config.get("simulator", {}))
    det = dict(config.get("deterministic_wrapper", {}))
    acc = dict(config.get("acceptance", {}))
    tokenizer = dict(config.get("tokenizer", {}))
    bands_raw = dict(config.get("bands", {}))
    bands: Dict[str, Tuple[float, float]] = {}
    for key, value in bands_raw.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            bands[str(key)] = (float(value[0]), float(value[1]))

    baseline_window_raw = sim.get("baseline_window", [0.0, 0.5])
    if baseline_window_raw is None:
        baseline_window = (0.0, float(sim.get("stim_onset", 0.5)))
    else:
        baseline_window = (float(baseline_window_raw[0]), float(baseline_window_raw[1]))

    return ForwardSettings(
        fs=int(sim.get("fs", 250)),
        duration=float(sim.get("duration", 2.0)),
        n_channels=int(sim.get("n_channels", 16)),
        stim_onset=float(sim.get("stim_onset", 0.5)),
        stim_sigma=float(sim.get("stim_sigma", 0.05)),
        stim_causal=bool(sim.get("stim_causal", False)),
        n_sources=int(sim.get("n_sources", 3)),
        n_trials=int(sim.get("n_trials", 10)),
        input_noise_std=float(sim.get("input_noise_std", 0.2)),
        sensor_noise_std=float(sim.get("sensor_noise_std", 2.0)),
        internal_fs=int(sim.get("internal_fs", 1000)),
        bandpass=(float(sim.get("bandpass", [0.5, 40.0])[0]), float(sim.get("bandpass", [0.5, 40.0])[1])),
        baseline_correct=bool(sim.get("baseline_correct", True)),
        baseline_window=baseline_window,
        warmup_sec=float(sim.get("warmup_sec", 3.0)),
        downsample_method=str(sim.get("downsample_method", "slice")),
        uV_scale=float(sim.get("uV_scale", 100.0)),
        deterministic_seed=int(det.get("seed", 314159)),
        post_window_sec=float(acc.get("post_window_sec", 0.60)),
        late_window_sec=float(acc.get("late_window_sec", 0.50)),
        tfr_method=str(tokenizer.get("tfr_method", TokenConfig.tfr_method)),
        tfr_f_min=float(tokenizer.get("f_min", TokenConfig.f_min)),
        tfr_f_max=float(tokenizer.get("f_max", TokenConfig.f_max)),
        n_time_patches=int(tokenizer.get("n_time_patches", TokenConfig.n_time_patches)),
        n_freq_patches=int(tokenizer.get("n_freq_patches", TokenConfig.n_freq_patches)),
        nperseg=int(tokenizer.get("nperseg", TokenConfig.nperseg)),
        noverlap=int(tokenizer.get("noverlap", TokenConfig.noverlap)),
        nfft=int(tokenizer.get("nfft", TokenConfig.nfft)),
        morlet_n_freqs=int(tokenizer.get("morlet_n_freqs", TokenConfig.morlet_n_freqs)),
        morlet_n_cycles_low=float(tokenizer.get("morlet_n_cycles_low", TokenConfig.morlet_n_cycles_low)),
        morlet_n_cycles_high=float(tokenizer.get("morlet_n_cycles_high", TokenConfig.morlet_n_cycles_high)),
        morlet_decim=int(tokenizer.get("morlet_decim", TokenConfig.morlet_decim)),
        morlet_n_jobs=int(tokenizer.get("morlet_n_jobs", TokenConfig.morlet_n_jobs)),
        bands=bands or ForwardSettings().bands,
    )


def _empty_result(theta: np.ndarray, token_cfg: TokenConfig, scalar_names: Sequence[str], reason: str, error_message: str = "") -> ForwardResult:
    n_t = int(round(token_cfg.duration * token_cfg.fs))
    eeg = np.full((token_cfg.n_channels, n_t), np.nan, dtype=np.float32)
    erp = np.full((token_cfg.n_time_patches, token_cfg.n_channels), np.nan, dtype=np.float32)
    tfr = np.full((token_cfg.n_time_patches * token_cfg.n_freq_patches, token_cfg.n_channels), np.nan, dtype=np.float32)
    hybrid = np.full((erp.shape[0] + tfr.shape[0], token_cfg.n_channels), np.nan, dtype=np.float32)
    scalars = np.full((len(scalar_names),), np.nan, dtype=np.float32)
    return ForwardResult(
        theta=np.asarray(theta, dtype=np.float32),
        accepted=False,
        rejection_reason=str(reason),
        error_message=str(error_message),
        eeg=eeg,
        erp_tokens=erp,
        tfr_tokens=tfr,
        hybrid_tokens=hybrid,
        scalar_names=tuple(str(s) for s in scalar_names),
        scalar_values=scalars,
    )


def _channel_names(settings: ForwardSettings) -> Tuple[str, ...]:
    if len(DEFAULT_16_CH_NAMES) == settings.n_channels:
        return tuple(DEFAULT_16_CH_NAMES)
    return tuple(f"Ch{i + 1}" for i in range(settings.n_channels))


def _build_time_vector(settings: ForwardSettings) -> np.ndarray:
    n_t = int(round(settings.duration * settings.fs))
    return np.arange(n_t, dtype=np.float64) / float(settings.fs)


def _window_slices(settings: ForwardSettings, n_t: int) -> Tuple[slice, slice, slice]:
    stim_idx = int(round(settings.stim_onset * settings.fs))
    stim_idx = max(0, min(stim_idx, n_t - 1))
    post_end = min(n_t, stim_idx + int(round(settings.post_window_sec * settings.fs)))
    late_start = max(0, n_t - int(round(settings.late_window_sec * settings.fs)))
    return slice(0, max(1, stim_idx)), slice(stim_idx, max(stim_idx + 1, post_end)), slice(late_start, n_t)


def _safe_peak_abs(signal: np.ndarray) -> Tuple[float, int, float]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.isfinite(x).any():
        return float("nan"), -1, float("nan")
    idx = int(np.nanargmax(np.abs(x)))
    return float(np.abs(x[idx])), idx, float(x[idx])


def _bandpower_db(eeg: np.ndarray, fs: int, baseline_slice: slice, post_slice: slice, bands: Mapping[str, Tuple[float, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2:
        for name in bands:
            out[f"band_{name}_db"] = float("nan")
        return out

    base = eeg[:, baseline_slice]
    post = eeg[:, post_slice]
    if base.shape[1] < 4 or post.shape[1] < 4:
        for name in bands:
            out[f"band_{name}_db"] = float("nan")
        return out

    nperseg_base = min(max(8, base.shape[1]), 128)
    nperseg_post = min(max(8, post.shape[1]), 128)
    f_base, p_base = welch(base, fs=float(fs), nperseg=nperseg_base, axis=1)
    f_post, p_post = welch(post, fs=float(fs), nperseg=nperseg_post, axis=1)

    for name, (lo, hi) in bands.items():
        m_base = (f_base >= float(lo)) & (f_base < float(hi))
        m_post = (f_post >= float(lo)) & (f_post < float(hi))
        if not np.any(m_base) or not np.any(m_post):
            out[f"band_{name}_db"] = float("nan")
            continue
        base_pow = float(np.nanmean(p_base[:, m_base]))
        post_pow = float(np.nanmean(p_post[:, m_post]))
        out[f"band_{name}_db"] = float(10.0 * np.log10((post_pow + 1e-12) / (base_pow + 1e-12)))
    return out


def _compute_scalar_summaries(eeg: np.ndarray, settings: ForwardSettings) -> Dict[str, float]:
    eeg = np.asarray(eeg, dtype=np.float64)
    n_t = eeg.shape[1]
    t = _build_time_vector(settings)
    baseline_slice, post_slice, late_slice = _window_slices(settings, n_t)

    gfp = np.std(eeg, axis=0)
    baseline_gfp = gfp[baseline_slice]
    post_gfp = gfp[post_slice]
    late_gfp = gfp[late_slice]
    post_t = t[post_slice]

    gfp_peak_amp = float(np.nanmax(post_gfp)) if post_gfp.size else float("nan")
    if post_gfp.size and np.isfinite(post_gfp).any():
        gfp_peak_idx = int(np.nanargmax(post_gfp))
        gfp_peak_latency = float(post_t[gfp_peak_idx] - settings.stim_onset)
    else:
        gfp_peak_idx = -1
        gfp_peak_latency = float("nan")
    gfp_auc_post = float(np.trapz(post_gfp, post_t)) if post_gfp.size else float("nan")

    mean_abs_post = np.mean(np.abs(eeg[:, post_slice]), axis=0) if eeg[:, post_slice].size else np.asarray([], dtype=np.float64)
    erp_auc_post_absmean = float(np.trapz(mean_abs_post, post_t)) if mean_abs_post.size else float("nan")

    ch_names = _channel_names(settings)
    cz_peak_abs = float("nan")
    cz_peak_latency = float("nan")
    cz_peak_signed = float("nan")
    if "Cz" in ch_names:
        cz_idx = ch_names.index("Cz")
        cz_sig = eeg[cz_idx, post_slice]
        cz_peak_abs, cz_peak_i, cz_peak_signed = _safe_peak_abs(cz_sig)
        if cz_peak_i >= 0 and post_t.size > cz_peak_i:
            cz_peak_latency = float(post_t[cz_peak_i] - settings.stim_onset)

    abs99 = float(np.nanpercentile(np.abs(eeg), 99.0)) if np.isfinite(eeg).any() else float("nan")
    baseline_mean = float(np.nanmean(baseline_gfp)) if baseline_gfp.size else float("nan")
    baseline_sd = float(np.nanstd(baseline_gfp)) if baseline_gfp.size else float("nan")
    baseline_rms = float(np.sqrt(np.nanmean(baseline_gfp ** 2)) + 1e-12) if baseline_gfp.size else float("nan")
    late_rms = float(np.sqrt(np.nanmean(late_gfp ** 2)) + 1e-12) if late_gfp.size else float("nan")
    gfp_peak_z = float((gfp_peak_amp - baseline_mean) / (baseline_sd + 1e-6)) if np.isfinite(gfp_peak_amp) else float("nan")
    late_rms_over_baseline = float(late_rms / (baseline_rms + 1e-12)) if np.isfinite(late_rms) and np.isfinite(baseline_rms) else float("nan")
    late_peak_over_post_peak = float((np.nanmax(late_gfp) if late_gfp.size else np.nan) / (gfp_peak_amp + 1e-12)) if np.isfinite(gfp_peak_amp) else float("nan")

    out: Dict[str, float] = {
        "gfp_peak_amp_uV": gfp_peak_amp,
        "gfp_peak_latency_s": gfp_peak_latency,
        "gfp_auc_post_uV_s": gfp_auc_post,
        "erp_auc_post_absmean_uV_s": erp_auc_post_absmean,
        "cz_peak_abs_uV": cz_peak_abs,
        "cz_peak_latency_s": cz_peak_latency,
        "cz_peak_signed_uV": cz_peak_signed,
        "abs99_uV": abs99,
        "gfp_peak_z": gfp_peak_z,
        "late_rms_over_baseline": late_rms_over_baseline,
        "late_peak_over_post_peak": late_peak_over_post_peak,
    }
    out.update(_bandpower_db(eeg, settings.fs, baseline_slice, post_slice, settings.bands))
    return out


@lru_cache(maxsize=8)
def _cached_leadfield(fs: int, n_sources: int) -> np.ndarray:
    leadfield, _info, _meta, _ch_pos, _src_pos, _src_ori = make_leadfield(
        fs=int(fs),
        n_sources=int(n_sources),
        seed=0,
    )
    return np.asarray(leadfield, dtype=np.float32)


def _leadfield(settings: ForwardSettings) -> np.ndarray:
    return _cached_leadfield(int(settings.fs), int(settings.n_sources))


def evaluate_one_theta(theta: np.ndarray, settings: ForwardSettings, spec: Optional[ParameterSpec] = None) -> ForwardResult:
    spec = validate_project_parameter_spec() if spec is None else spec
    theta = np.asarray(theta, dtype=np.float32).reshape(-1)
    token_cfg = settings.token_config()
    scalar_template = [
        "gfp_peak_amp_uV",
        "gfp_peak_latency_s",
        "gfp_auc_post_uV_s",
        "erp_auc_post_absmean_uV_s",
        "cz_peak_abs_uV",
        "cz_peak_latency_s",
        "cz_peak_signed_uV",
        "abs99_uV",
        "gfp_peak_z",
        "late_rms_over_baseline",
        "late_peak_over_post_peak",
        *[f"band_{band}_db" for band in settings.bands.keys()],
    ]

    if theta.shape != (spec.dim,):
        return _empty_result(theta, token_cfg, scalar_template, reason="bad_theta_shape", error_message=str(theta.shape))

    if not np.all(np.isfinite(theta)):
        return _empty_result(theta, token_cfg, scalar_template, reason="nan_theta", error_message="theta contains non-finite values")

    params = theta_to_params(theta, spec.names)
    try:
        eeg = simulate_eeg(
            params=params,
            fs=int(settings.fs),
            duration=float(settings.duration),
            n_channels=int(settings.n_channels),
            seed=int(settings.deterministic_seed),
            bandpass=tuple(float(v) for v in settings.bandpass),
            stim_onset=float(settings.stim_onset),
            stim_sigma=float(settings.stim_sigma),
            stim_causal=bool(settings.stim_causal),
            n_sources=int(settings.n_sources),
            leadfield=_leadfield(settings),
            sensor_noise_std=float(settings.sensor_noise_std),
            n_trials=int(settings.n_trials),
            input_noise_std=float(settings.input_noise_std),
            internal_fs=int(settings.internal_fs),
            baseline_correct=bool(settings.baseline_correct),
            baseline_window=tuple(float(v) for v in settings.baseline_window) if settings.baseline_window is not None else None,
            warmup_sec=float(settings.warmup_sec),
            downsample_method=str(settings.downsample_method),
            uV_scale=float(settings.uV_scale),
            return_trials=False,
            return_sources=False,
        )
        eeg = np.asarray(eeg, dtype=np.float32)
    except Exception as exc:
        return _empty_result(theta, token_cfg, scalar_template, reason="simulation_error", error_message=repr(exc))

    try:
        accepted, reason = regime_reject(eeg, token_cfg)
    except Exception as exc:
        accepted = False
        reason = "regime_filter_error"
        return _empty_result(theta, token_cfg, scalar_template, reason=reason, error_message=repr(exc))

    if not accepted:
        # Preserve the raw EEG for QC, but mask downstream features/scalars so accepted-only
        # sensitivity summaries are not distorted by rejected regimes.
        out = _empty_result(theta, token_cfg, scalar_template, reason=reason, error_message="")
        return ForwardResult(
            theta=out.theta,
            accepted=False,
            rejection_reason=str(reason),
            error_message="",
            eeg=eeg,
            erp_tokens=out.erp_tokens,
            tfr_tokens=out.tfr_tokens,
            hybrid_tokens=out.hybrid_tokens,
            scalar_names=out.scalar_names,
            scalar_values=out.scalar_values,
        )

    try:
        erp_tokens = compute_erp_tokens(eeg, token_cfg).astype(np.float32)
        tfr_tokens, _tfr_meta = compute_tfr_tokens(eeg, token_cfg)
        tfr_tokens = np.asarray(tfr_tokens, dtype=np.float32)
    except Exception as exc:
        out = _empty_result(theta, token_cfg, scalar_template, reason="feature_error", error_message=repr(exc))
        return ForwardResult(
            theta=out.theta,
            accepted=False,
            rejection_reason="feature_error",
            error_message=repr(exc),
            eeg=eeg,
            erp_tokens=out.erp_tokens,
            tfr_tokens=out.tfr_tokens,
            hybrid_tokens=out.hybrid_tokens,
            scalar_names=out.scalar_names,
            scalar_values=out.scalar_values,
        )

    hybrid_tokens = np.concatenate([erp_tokens, tfr_tokens], axis=0).astype(np.float32)
    scalar_dict = _compute_scalar_summaries(eeg, settings)
    scalar_names = tuple(scalar_template)
    scalar_values = np.asarray([scalar_dict.get(name, np.nan) for name in scalar_names], dtype=np.float32)

    return ForwardResult(
        theta=np.asarray(theta, dtype=np.float32),
        accepted=True,
        rejection_reason="ok",
        error_message="",
        eeg=eeg,
        erp_tokens=erp_tokens,
        tfr_tokens=tfr_tokens,
        hybrid_tokens=hybrid_tokens,
        scalar_names=scalar_names,
        scalar_values=scalar_values,
    )


def _stack_results(results: Sequence[ForwardResult]) -> Dict[str, np.ndarray]:
    if not results:
        raise ValueError("No forward results to stack")
    scalar_names = np.asarray([name.encode("utf-8") for name in results[0].scalar_names], dtype="S")
    out: Dict[str, np.ndarray] = {
        "theta": np.stack([r.theta for r in results], axis=0).astype(np.float32),
        "accepted": np.asarray([r.accepted for r in results], dtype=bool),
        "rejection_reason": np.asarray([str(r.rejection_reason).encode("utf-8") for r in results], dtype="S"),
        "error_message": np.asarray([str(r.error_message).encode("utf-8") for r in results], dtype="S"),
        "eeg": np.stack([r.eeg for r in results], axis=0).astype(np.float32),
        "erp_tokens": np.stack([r.erp_tokens for r in results], axis=0).astype(np.float32),
        "tfr_tokens": np.stack([r.tfr_tokens for r in results], axis=0).astype(np.float32),
        "hybrid_tokens": np.stack([r.hybrid_tokens for r in results], axis=0).astype(np.float32),
        "scalar_names": scalar_names,
        "scalar_values": np.stack([r.scalar_values for r in results], axis=0).astype(np.float32),
    }
    return out


def evaluate_theta_matrix(theta_matrix: np.ndarray, settings: ForwardSettings, n_jobs: int = 1, verbose: int = 0) -> Dict[str, np.ndarray]:
    spec = validate_project_parameter_spec()
    theta_matrix = np.asarray(theta_matrix, dtype=np.float32)
    if theta_matrix.ndim != 2 or theta_matrix.shape[1] != spec.dim:
        raise ValueError(f"theta_matrix must have shape (N, {spec.dim}), got {theta_matrix.shape}")

    def _run_one(row: np.ndarray) -> ForwardResult:
        return evaluate_one_theta(row, settings=settings, spec=spec)

    if int(n_jobs) != 1 and Parallel is not None and delayed is not None:
        results = Parallel(n_jobs=int(n_jobs), verbose=int(verbose))(delayed(_run_one)(theta_matrix[i]) for i in range(theta_matrix.shape[0]))
    else:
        results = [_run_one(theta_matrix[i]) for i in range(theta_matrix.shape[0])]
    return _stack_results(results)
EOF
cat > sensitivity/sampling.py <<'EOF'
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import qmc

from sensitivity.common import ParameterSpec


@dataclass(frozen=True)
class MorrisDesign:
    theta: np.ndarray           # (N, D)
    unit: np.ndarray            # (N, D) in [0, 1]
    trajectory_ids: np.ndarray  # (N,)
    point_in_trajectory: np.ndarray  # (N,)
    step_from: np.ndarray       # (T*D,)
    step_to: np.ndarray         # (T*D,)
    step_param: np.ndarray      # (T*D,)
    num_levels: int
    grid_jump: int
    num_trajectories: int
    candidate_pool_size: int
    seed: int

    @property
    def dim(self) -> int:
        return int(self.theta.shape[1])

    @property
    def delta(self) -> float:
        return float(self.grid_jump) / float(self.num_levels - 1)


@dataclass(frozen=True)
class MorrisStats:
    mu_star: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    n_effects: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray


@dataclass(frozen=True)
class SobolDesign:
    A: np.ndarray
    B: np.ndarray
    AB: np.ndarray  # (D, N, D) in theta-space
    A_unit: np.ndarray
    B_unit: np.ndarray
    AB_unit: np.ndarray
    base_samples: int
    seed: int

    @property
    def dim(self) -> int:
        return int(self.A.shape[1])


@dataclass(frozen=True)
class SobolStats:
    S1: np.ndarray
    ST: np.ndarray
    S1_ci_low: np.ndarray
    S1_ci_high: np.ndarray
    ST_ci_low: np.ndarray
    ST_ci_high: np.ndarray
    n_valid: np.ndarray


@dataclass(frozen=True)
class PCABasis:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    source: str


def _map_unit_to_theta(unit: np.ndarray, spec: ParameterSpec) -> np.ndarray:
    unit = np.asarray(unit, dtype=np.float64)
    return spec.lows[None, :] + unit * spec.ranges[None, :]


def _generate_one_morris_trajectory(dim: int, num_levels: int, grid_jump: int, rng: np.random.Generator) -> np.ndarray:
    delta = float(grid_jump) / float(num_levels - 1)
    grid = np.linspace(0.0, 1.0, num_levels, dtype=np.float64)

    x0 = np.zeros((dim,), dtype=np.float64)
    signs = np.ones((dim,), dtype=np.float64)
    for j in range(dim):
        valid_plus = grid[grid <= (1.0 - delta + 1e-12)]
        valid_minus = grid[grid >= (delta - 1e-12)]
        choose_plus = bool(rng.integers(0, 2))
        if choose_plus and valid_plus.size > 0:
            x0[j] = float(valid_plus[int(rng.integers(0, valid_plus.size))])
            signs[j] = +1.0
        elif valid_minus.size > 0:
            x0[j] = float(valid_minus[int(rng.integers(0, valid_minus.size))])
            signs[j] = -1.0
        else:
            # Fallback should be unreachable for standard Morris configs.
            x0[j] = 0.0
            signs[j] = +1.0

    order = rng.permutation(dim)
    pts = np.zeros((dim + 1, dim), dtype=np.float64)
    pts[0] = x0
    cur = x0.copy()
    for k, j in enumerate(order, start=1):
        cur = cur.copy()
        cur[j] = np.clip(cur[j] + signs[j] * delta, 0.0, 1.0)
        pts[k] = cur
    return pts


def _flatten_trajectory(pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, dtype=np.float64).reshape(-1)


def _choose_diverse_trajectories(candidates: Sequence[np.ndarray], num_keep: int, rng: np.random.Generator) -> List[np.ndarray]:
    if len(candidates) <= num_keep:
        return list(candidates)
    flats = np.stack([_flatten_trajectory(c) for c in candidates], axis=0)
    # Start from the trajectory farthest from the global mean.
    mean_flat = flats.mean(axis=0, keepdims=True)
    d0 = np.linalg.norm(flats - mean_flat, axis=1)
    first = int(np.argmax(d0))
    selected = [first]
    remaining = set(range(len(candidates)))
    remaining.remove(first)

    while len(selected) < num_keep and remaining:
        rem = np.asarray(sorted(remaining), dtype=np.int64)
        dists = np.full((rem.size,), np.inf, dtype=np.float64)
        for s in selected:
            cur = np.linalg.norm(flats[rem] - flats[s][None, :], axis=1)
            dists = np.minimum(dists, cur)
        best = int(rem[int(np.argmax(dists))])
        selected.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected]


def generate_morris_design(
    spec: ParameterSpec,
    num_trajectories: int,
    num_levels: int,
    seed: int,
    grid_jump: Optional[int] = None,
    candidate_pool_size: Optional[int] = None,
) -> MorrisDesign:
    dim = spec.dim
    if num_levels < 3:
        raise ValueError("num_levels must be at least 3")
    if grid_jump is None:
        grid_jump = max(1, num_levels // 2)
    if grid_jump >= num_levels:
        raise ValueError("grid_jump must be < num_levels")
    if candidate_pool_size is None:
        candidate_pool_size = num_trajectories
    if candidate_pool_size < num_trajectories:
        candidate_pool_size = num_trajectories

    rng = np.random.default_rng(seed)
    candidates = [_generate_one_morris_trajectory(dim, num_levels, grid_jump, rng) for _ in range(candidate_pool_size)]
    chosen = _choose_diverse_trajectories(candidates, num_trajectories, rng)

    units: List[np.ndarray] = []
    traj_ids: List[int] = []
    point_ids: List[int] = []
    step_from: List[int] = []
    step_to: List[int] = []
    step_param: List[int] = []
    offset = 0
    for t_idx, pts in enumerate(chosen):
        units.append(pts)
        traj_ids.extend([t_idx] * pts.shape[0])
        point_ids.extend(list(range(pts.shape[0])))
        diffs = pts[1:] - pts[:-1]
        for k in range(diffs.shape[0]):
            changed = np.where(np.abs(diffs[k]) > 1e-12)[0]
            if changed.size != 1:
                raise RuntimeError("Each Morris step must change exactly one parameter")
            step_from.append(offset + k)
            step_to.append(offset + k + 1)
            step_param.append(int(changed[0]))
        offset += pts.shape[0]

    unit = np.vstack(units).astype(np.float64)
    theta = _map_unit_to_theta(unit, spec).astype(np.float64)

    return MorrisDesign(
        theta=theta,
        unit=unit,
        trajectory_ids=np.asarray(traj_ids, dtype=np.int32),
        point_in_trajectory=np.asarray(point_ids, dtype=np.int32),
        step_from=np.asarray(step_from, dtype=np.int32),
        step_to=np.asarray(step_to, dtype=np.int32),
        step_param=np.asarray(step_param, dtype=np.int32),
        num_levels=int(num_levels),
        grid_jump=int(grid_jump),
        num_trajectories=int(num_trajectories),
        candidate_pool_size=int(candidate_pool_size),
        seed=int(seed),
    )


def compute_morris_statistics(Y: np.ndarray, design: MorrisDesign, standardize: bool = True) -> MorrisStats:
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2 or Y.shape[0] != design.theta.shape[0]:
        raise ValueError(f"Y must have shape ({design.theta.shape[0]}, O), got {Y.shape}")

    output_mean = np.nanmean(Y, axis=0)
    output_std = np.nanstd(Y, axis=0)
    if standardize:
        Y_work = (Y - output_mean[None, :]) / np.where(output_std[None, :] > 1e-12, output_std[None, :], 1.0)
    else:
        Y_work = Y.copy()

    D = design.dim
    O = Y_work.shape[1]
    effects_per_param: List[List[np.ndarray]] = [[[] for _ in range(O)] for _ in range(D)]

    for i_from, i_to, p in zip(design.step_from, design.step_to, design.step_param):
        dx = float(design.unit[i_to, p] - design.unit[i_from, p])
        if abs(dx) < 1e-12:
            continue
        dy = Y_work[i_to] - Y_work[i_from]
        ee = dy / dx
        for o in range(O):
            if np.isfinite(ee[o]):
                effects_per_param[int(p)][o].append(float(ee[o]))

    mu_star = np.full((D, O), np.nan, dtype=np.float64)
    mu = np.full((D, O), np.nan, dtype=np.float64)
    sigma = np.full((D, O), np.nan, dtype=np.float64)
    n_effects = np.zeros((D, O), dtype=np.int32)

    for p in range(D):
        for o in range(O):
            vals = np.asarray(effects_per_param[p][o], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            n = int(vals.size)
            n_effects[p, o] = n
            if n == 0:
                continue
            mu_star[p, o] = float(np.mean(np.abs(vals)))
            mu[p, o] = float(np.mean(vals))
            sigma[p, o] = float(np.std(vals, ddof=1)) if n > 1 else 0.0

    return MorrisStats(
        mu_star=mu_star,
        mu=mu,
        sigma=sigma,
        n_effects=n_effects,
        output_mean=output_mean,
        output_std=output_std,
    )



def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)



def generate_sobol_design(spec: ParameterSpec, base_samples: int, seed: int) -> SobolDesign:
    if base_samples < 4:
        raise ValueError("base_samples must be >= 4")
    dim = spec.dim
    engine = qmc.Sobol(d=2 * dim, scramble=True, seed=int(seed))
    if _is_power_of_two(base_samples):
        unit_all = engine.random_base2(m=int(np.log2(base_samples)))
    else:
        unit_all = engine.random(n=int(base_samples))

    A_unit = np.asarray(unit_all[:, :dim], dtype=np.float64)
    B_unit = np.asarray(unit_all[:, dim:], dtype=np.float64)
    AB_unit = np.repeat(B_unit[None, :, :], repeats=dim, axis=0)
    for i in range(dim):
        AB_unit[i, :, i] = A_unit[:, i]

    A = _map_unit_to_theta(A_unit, spec)
    B = _map_unit_to_theta(B_unit, spec)
    AB = np.stack([_map_unit_to_theta(AB_unit[i], spec) for i in range(dim)], axis=0)

    return SobolDesign(
        A=A.astype(np.float64),
        B=B.astype(np.float64),
        AB=AB.astype(np.float64),
        A_unit=A_unit.astype(np.float64),
        B_unit=B_unit.astype(np.float64),
        AB_unit=AB_unit.astype(np.float64),
        base_samples=int(base_samples),
        seed=int(seed),
    )



def _sobol_s1(yA: np.ndarray, yB: np.ndarray, yAB: np.ndarray) -> float:
    v = float(np.var(np.concatenate([yA, yB]), ddof=1))
    if not np.isfinite(v) or v <= 1e-12:
        return float("nan")
    return float(np.mean(yB * (yAB - yA)) / v)



def _sobol_st(yA: np.ndarray, yAB: np.ndarray, yB: np.ndarray) -> float:
    v = float(np.var(np.concatenate([yA, yB]), ddof=1))
    if not np.isfinite(v) or v <= 1e-12:
        return float("nan")
    return float(0.5 * np.mean((yA - yAB) ** 2) / v)



def compute_sobol_statistics(
    A: np.ndarray,
    B: np.ndarray,
    AB: np.ndarray,
    bootstrap_resamples: int = 200,
    seed: int = 0,
) -> SobolStats:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    AB = np.asarray(AB, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    if A.shape != B.shape:
        raise ValueError(f"A and B must have same shape, got {A.shape} and {B.shape}")
    if AB.ndim != 3 or AB.shape[0] != A.shape[0] or AB.shape[1] != A.shape[1]:
        raise ValueError(f"AB must have shape (N, O, D), got {AB.shape}")

    N, O = A.shape
    D = AB.shape[2]
    rng = np.random.default_rng(seed)

    S1 = np.full((D, O), np.nan, dtype=np.float64)
    ST = np.full((D, O), np.nan, dtype=np.float64)
    S1_ci_low = np.full((D, O), np.nan, dtype=np.float64)
    S1_ci_high = np.full((D, O), np.nan, dtype=np.float64)
    ST_ci_low = np.full((D, O), np.nan, dtype=np.float64)
    ST_ci_high = np.full((D, O), np.nan, dtype=np.float64)
    n_valid = np.zeros((D, O), dtype=np.int32)

    for p in range(D):
        for o in range(O):
            yA = A[:, o]
            yB = B[:, o]
            yAB = AB[:, o, p]
            mask = np.isfinite(yA) & np.isfinite(yB) & np.isfinite(yAB)
            yA_ok = np.asarray(yA[mask], dtype=np.float64)
            yB_ok = np.asarray(yB[mask], dtype=np.float64)
            yAB_ok = np.asarray(yAB[mask], dtype=np.float64)
            n = int(yA_ok.size)
            n_valid[p, o] = n
            if n < 4:
                continue
            S1[p, o] = _sobol_s1(yA_ok, yB_ok, yAB_ok)
            ST[p, o] = _sobol_st(yA_ok, yAB_ok, yB_ok)
            if bootstrap_resamples <= 0:
                continue
            s1_boot = np.full((bootstrap_resamples,), np.nan, dtype=np.float64)
            st_boot = np.full((bootstrap_resamples,), np.nan, dtype=np.float64)
            for b in range(bootstrap_resamples):
                idx = rng.integers(0, n, size=n, dtype=np.int64)
                s1_boot[b] = _sobol_s1(yA_ok[idx], yB_ok[idx], yAB_ok[idx])
                st_boot[b] = _sobol_st(yA_ok[idx], yAB_ok[idx], yB_ok[idx])
            if np.isfinite(s1_boot).any():
                S1_ci_low[p, o], S1_ci_high[p, o] = np.nanquantile(s1_boot, [0.025, 0.975])
            if np.isfinite(st_boot).any():
                ST_ci_low[p, o], ST_ci_high[p, o] = np.nanquantile(st_boot, [0.025, 0.975])

    return SobolStats(
        S1=S1,
        ST=ST,
        S1_ci_low=S1_ci_low,
        S1_ci_high=S1_ci_high,
        ST_ci_low=ST_ci_low,
        ST_ci_high=ST_ci_high,
        n_valid=n_valid,
    )



def fit_pca_basis(X: np.ndarray, n_components: int, source: str = "") -> PCABasis:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be rank-2")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples for PCA")
    scaler_mean = np.mean(X, axis=0)
    scaler_scale = np.std(X, axis=0)
    scaler_scale = np.where(scaler_scale > 1e-12, scaler_scale, 1.0)
    Xs = (X - scaler_mean[None, :]) / scaler_scale[None, :]
    pca_mean = np.mean(Xs, axis=0)
    Xc = Xs - pca_mean[None, :]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n_comp = int(max(1, min(n_components, Vt.shape[0])))
    components = Vt[:n_comp].astype(np.float64)
    explained = (S ** 2) / max(1, Xc.shape[0] - 1)
    explained_ratio = explained[:n_comp] / max(float(np.sum(explained)), 1e-12)
    return PCABasis(
        scaler_mean=scaler_mean.astype(np.float64),
        scaler_scale=scaler_scale.astype(np.float64),
        pca_mean=pca_mean.astype(np.float64),
        components=components.astype(np.float64),
        explained_variance_ratio=explained_ratio.astype(np.float64),
        source=str(source),
    )



def apply_pca_basis(X: np.ndarray, basis: PCABasis) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be rank-2")
    Y = np.full((X.shape[0], basis.components.shape[0]), np.nan, dtype=np.float64)
    finite_rows = np.all(np.isfinite(X), axis=1)
    if not np.any(finite_rows):
        return Y
    X_ok = X[finite_rows]
    Xs = (X_ok - basis.scaler_mean[None, :]) / basis.scaler_scale[None, :]
    Xc = Xs - basis.pca_mean[None, :]
    Y[finite_rows] = Xc @ basis.components.T
    return Y
EOF
cat > sensitivity/plotting.py <<'EOF'
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _savefig_formats(fig: plt.Figure, out_base: Path, formats: Sequence[str]) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _maybe_rotation(n: int) -> int:
    return 30 if n <= 12 else 60


def plot_morris_family_summary(
    mu_star_by_family: Mapping[str, np.ndarray],
    sigma_by_family: Mapping[str, np.ndarray],
    param_names: Sequence[str],
    out_base: Path,
    formats: Sequence[str] = ("png",),
    title: str = "",
) -> None:
    families = [k for k in ["scalar", "erp", "tfr", "hybrid"] if k in mu_star_by_family]
    x = np.arange(len(param_names), dtype=np.float64)
    width = 0.8 / max(1, len(families))

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 1.1 * len(param_names)), 8), sharex=True)
    for i, fam in enumerate(families):
        dx = (i - 0.5 * (len(families) - 1)) * width
        axes[0].bar(x + dx, np.asarray(mu_star_by_family[fam], dtype=np.float64), width=width, label=fam.upper(), alpha=0.9)
        axes[1].bar(x + dx, np.asarray(sigma_by_family.get(fam, np.full_like(mu_star_by_family[fam], np.nan)), dtype=np.float64), width=width, label=fam.upper(), alpha=0.9)

    axes[0].set_ylabel("Mean Morris $\\mu^*$")
    axes[1].set_ylabel("Mean Morris $\\sigma$")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(param_names, rotation=_maybe_rotation(len(param_names)), ha="right")
    axes[0].legend(ncol=max(1, min(4, len(families))), frameon=False)
    if title:
        axes[0].set_title(title)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    _savefig_formats(fig, out_base, formats)



def plot_erp_heatmaps(
    mu_star_time_channel: np.ndarray,
    param_names: Sequence[str],
    channel_names: Sequence[str],
    time_edges: np.ndarray,
    out_base: Path,
    formats: Sequence[str] = ("png",),
) -> None:
    arr = np.asarray(mu_star_time_channel, dtype=np.float64)
    P, T, C = arr.shape
    ncols = 3
    nrows = int(np.ceil(P / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0

    for p in range(nrows * ncols):
        ax = axes.flat[p]
        if p >= P:
            ax.axis("off")
            continue
        im = ax.imshow(arr[p].T, origin="lower", aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_title(str(param_names[p]))
        ax.set_xlabel("Time patch")
        ax.set_ylabel("Channel")
        ax.set_xticks(np.arange(T))
        ax.set_xticklabels([f"{0.5 * (time_edges[i] + time_edges[i + 1]):.1f}" for i in range(T)], rotation=90, fontsize=7)
        ax.set_yticks(np.arange(C))
        ax.set_yticklabels(channel_names, fontsize=7)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Morris $\\mu^*$")
    fig.suptitle("ERP Morris sensitivity heatmaps", y=1.02)
    _savefig_formats(fig, out_base, formats)



def plot_tfr_heatmaps(
    mu_star_time_freq: np.ndarray,
    param_names: Sequence[str],
    time_edges: np.ndarray,
    freq_edges: np.ndarray,
    out_base: Path,
    formats: Sequence[str] = ("png",),
) -> None:
    arr = np.asarray(mu_star_time_freq, dtype=np.float64)
    P, T, F = arr.shape
    ncols = 3
    nrows = int(np.ceil(P / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0

    for p in range(nrows * ncols):
        ax = axes.flat[p]
        if p >= P:
            ax.axis("off")
            continue
        im = ax.imshow(arr[p].T, origin="lower", aspect="auto", vmin=0.0, vmax=vmax)
        ax.set_title(str(param_names[p]))
        ax.set_xlabel("Time patch")
        ax.set_ylabel("Freq patch")
        ax.set_xticks(np.arange(T))
        ax.set_xticklabels([f"{0.5 * (time_edges[i] + time_edges[i + 1]):.1f}" for i in range(T)], rotation=90, fontsize=7)
        ax.set_yticks(np.arange(F))
        ax.set_yticklabels([f"{np.sqrt(freq_edges[i] * freq_edges[i + 1]):.1f}" for i in range(F)], fontsize=7)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Mean channel-collapsed Morris $\\mu^*$")
    fig.suptitle("TFR Morris sensitivity heatmaps", y=1.02)
    _savefig_formats(fig, out_base, formats)



def plot_acceptance_morris(
    mu_star: np.ndarray,
    sigma: np.ndarray,
    param_names: Sequence[str],
    out_base: Path,
    formats: Sequence[str] = ("png",),
) -> None:
    x = np.arange(len(param_names), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(param_names)), 4.5))
    ax.bar(x - width / 2.0, np.asarray(mu_star, dtype=np.float64), width=width, label="$\\mu^*$", alpha=0.9)
    ax.bar(x + width / 2.0, np.asarray(sigma, dtype=np.float64), width=width, label="$\\sigma$", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=_maybe_rotation(len(param_names)), ha="right")
    ax.set_ylabel("Acceptance-indicator sensitivity")
    ax.set_title("Morris sensitivity of acceptance indicator")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    _savefig_formats(fig, out_base, formats)



def plot_sobol_bars(
    S1: np.ndarray,
    ST: np.ndarray,
    param_names: Sequence[str],
    output_names: Sequence[str],
    out_base: Path,
    family_name: str,
    formats: Sequence[str] = ("png",),
) -> None:
    S1 = np.asarray(S1, dtype=np.float64)
    ST = np.asarray(ST, dtype=np.float64)
    if S1.ndim != 2 or ST.shape != S1.shape:
        raise ValueError("S1 and ST must both have shape (n_outputs, n_params)")

    n_outputs, n_params = S1.shape
    ncols = 2 if n_outputs > 1 else 1
    nrows = int(np.ceil(n_outputs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(10, 5.2 * ncols), max(3.8, 3.2 * nrows)), squeeze=False)
    x = np.arange(n_params, dtype=np.float64)
    width = 0.38

    for i in range(nrows * ncols):
        ax = axes.flat[i]
        if i >= n_outputs:
            ax.axis("off")
            continue
        ax.bar(x - width / 2.0, S1[i], width=width, label="S1", alpha=0.9)
        ax.bar(x + width / 2.0, ST[i], width=width, label="ST", alpha=0.9)
        ax.set_title(str(output_names[i]))
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.set_ylim(bottom=min(-0.05, np.nanmin(S1[i]) - 0.05 if np.isfinite(S1[i]).any() else -0.05), top=max(1.0, np.nanmax(ST[i]) + 0.05 if np.isfinite(ST[i]).any() else 1.0))
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f"Sobol S1/ST for {family_name} outputs", y=1.02)
    _savefig_formats(fig, out_base, formats)



def plot_rank_comparison_grid(
    rank_grid: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    param_names: Sequence[str],
    out_base: Path,
    formats: Sequence[str] = ("png",),
) -> None:
    labels = list(rank_grid.keys())
    ncols = len(labels)
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.4), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        sens_rank, rec_rank = rank_grid[label]
        sens_rank = np.asarray(sens_rank, dtype=np.float64)
        rec_rank = np.asarray(rec_rank, dtype=np.float64)
        mask = np.isfinite(sens_rank) & np.isfinite(rec_rank)
        if np.any(mask):
            ax.scatter(sens_rank[mask], rec_rank[mask], s=40)
            for x, y, nm in zip(sens_rank[mask], rec_rank[mask], np.asarray(param_names)[mask]):
                ax.text(float(x) + 0.05, float(y) + 0.05, str(nm), fontsize=8)
            max_rank = int(max(np.nanmax(sens_rank[mask]), np.nanmax(rec_rank[mask])))
            ax.plot([1, max_rank], [1, max_rank], linestyle="--", linewidth=1.0)
            ax.set_xlim(0.5, max_rank + 0.5)
            ax.set_ylim(max_rank + 0.5, 0.5)
        ax.set_title(label)
        ax.set_xlabel("Sensitivity rank (1 = highest)")
        ax.set_ylabel("Recoverability rank (1 = highest)")
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Sensitivity ranking vs recoverability ranking", y=1.02)
    _savefig_formats(fig, out_base, formats)



def plot_value_scatter(
    x_values: np.ndarray,
    y_values: np.ndarray,
    param_names: Sequence[str],
    x_label: str,
    y_label: str,
    title: str,
    out_base: Path,
    formats: Sequence[str] = ("png",),
) -> None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    if np.any(mask):
        ax.scatter(x[mask], y[mask], s=45)
        for xv, yv, nm in zip(x[mask], y[mask], np.asarray(param_names)[mask]):
            ax.text(float(xv) + 0.01, float(yv) + 0.01, str(nm), fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _savefig_formats(fig, out_base, formats)
EOF
cat > sensitivity/run_morris.py <<'EOF'
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
EOF
cat > sensitivity/run_sobol.py <<'EOF'
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
EOF
cat > sensitivity/compare_to_recoverability.py <<'EOF'
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
EOF
cat > sensitivity/summarize_outputs.py <<'EOF'
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from sensitivity.common import load_config, load_manifest, sensitivity_root



def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()



def _fmt_runtime(payload: Optional[Mapping[str, Any]]) -> str:
    if not payload:
        return "n/a"
    val = payload.get("runtime_seconds", None)
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.1f} s"
    except Exception:
        return str(val)



def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sensitivity outputs into one markdown report.")
    parser.add_argument("--config", type=str, default=None, help="Path to config_sensitivity.yaml")
    parser.add_argument("--out-root", type=str, default=None, help="Root output directory (defaults to config paths.results_root)")
    args = parser.parse_args()

    config = load_config(args.config)
    results_root = sensitivity_root(args.out_root or config.get("paths", {}).get("results_root", "results_sensitivity"))
    results_root.mkdir(parents=True, exist_ok=True)
    table_dir = results_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    morris_runtime = load_manifest(results_root / "morris" / "morris_runtime.json")
    sobol_runtime = load_manifest(results_root / "sobol" / "sobol_runtime.json")
    morris_rej = load_manifest(results_root / "morris" / "morris_rejection_summary.json")
    sobol_rej = load_manifest(results_root / "sobol" / "sobol_rejection_summary.json")

    morris_text = _read_text_if_exists(table_dir / "morris_summary.md")
    sobol_text = _read_text_if_exists(table_dir / "sobol_summary.md")
    cmp_text = _read_text_if_exists(table_dir / "comparison_summary.md")

    lines: List[str] = []
    lines.append("# Sensitivity analysis summary")
    lines.append("")
    lines.append("This report combines the Morris screening run, the scoped Sobol analysis, and the recoverability comparison.")
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- Morris runtime: {_fmt_runtime(morris_runtime)}")
    lines.append(f"- Sobol runtime: {_fmt_runtime(sobol_runtime)}")
    lines.append("")

    if morris_rej:
        lines.append("## Morris rejection summary")
        lines.append("")
        lines.append(f"- Acceptance rate: {100.0 * float(morris_rej.get('acceptance_rate', 0.0)):.2f}%")
        lines.append(f"- Rejection rate: {100.0 * float(morris_rej.get('rejection_rate', 0.0)):.2f}%")
        lines.append("")
    if sobol_rej:
        lines.append("## Sobol rejection summary")
        lines.append("")
        lines.append(f"- Acceptance rate: {100.0 * float(sobol_rej.get('acceptance_rate', 0.0)):.2f}%")
        lines.append(f"- Rejection rate: {100.0 * float(sobol_rej.get('rejection_rate', 0.0)):.2f}%")
        lines.append("")

    if morris_text:
        lines.append(morris_text)
        lines.append("")
    if sobol_text:
        lines.append(sobol_text)
        lines.append("")
    if cmp_text:
        lines.append(cmp_text)
        lines.append("")

    out_path = results_root / "sensitivity_summary.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "analysis": "summary",
                "output_file": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
EOF
cat > sensitivity/config_sensitivity.yaml <<'EOF'
paths:
  results_root: results_sensitivity

figures:
  formats: [png, pdf]

caching:
  use_cache: true

simulator:
  fs: 250
  duration: 2.0
  n_channels: 16
  stim_onset: 0.5
  stim_sigma: 0.05
  warmup_sec: 3.0
  n_sources: 3
  n_trials: 10
  input_noise_std: 0.2
  sensor_noise_std: 2.0
  internal_fs: 1000
  bandpass: [0.5, 40.0]
  baseline_correct: true
  baseline_window: [0.0, 0.5]
  downsample_method: slice
  uV_scale: 100.0

deterministic_wrapper:
  # Fixed common-random-numbers seed used only for sensitivity analysis,
  # so the forward wrapper is deterministic for a given theta.
  seed: 314159

# Optional tokenizer overrides. Leave commented if you want to inherit the
# project's existing TokenConfig defaults. Uncomment tfr_method: morlet if your
# current study/run used Morlet TFR tokens and you want sensitivity to match it.
# tokenizer:
#   tfr_method: morlet
#   morlet_n_freqs: 48
#   morlet_n_cycles_low: 4.0
#   morlet_n_cycles_high: 8.0
#   morlet_decim: 1
#   morlet_n_jobs: 1

acceptance:
  post_window_sec: 0.60
  late_window_sec: 0.50
  non_negligible_rate: 0.01

bands:
  delta: [0.5, 4.0]
  theta: [4.0, 8.0]
  alpha: [8.0, 12.0]
  beta: [13.0, 30.0]
  gamma: [30.0, 40.0]

morris:
  num_levels: 6
  grid_jump: 3
  num_trajectories: 24
  candidate_pool_size: 64
  seed: 2026
  n_jobs: 1
  standardize_outputs: true

sobol:
  base_samples: 128
  seed: 2027
  n_jobs: 1
  bootstrap_resamples: 200
  second_order: false
  pca_components: 3
  selected_scalar_outputs:
    - gfp_peak_amp_uV
    - gfp_peak_latency_s
    - gfp_auc_post_uV_s
    - erp_auc_post_absmean_uV_s
    - cz_peak_abs_uV
    - cz_peak_latency_s
    - band_delta_db
    - band_theta_db
    - band_alpha_db
    - band_beta_db
    - band_gamma_db

recoverability:
  files: {}
  fallback_to_reference: true
EOF
cat > sensitivity/__init__.py <<'EOF'
"""Sensitivity analysis package for the CMC EEG inverse-modeling project."""
EOF
echo 'Sensitivity module files written under ./sensitivity'
