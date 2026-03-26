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
    AB_unit = np.repeat(A_unit[None, :, :], repeats=dim, axis=0)
    for i in range(dim):
        AB_unit[i, :, i] = B_unit[:, i]

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
