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
        accepted=bool(accepted),
        rejection_reason="ok" if bool(accepted) else str(reason),
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
