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
