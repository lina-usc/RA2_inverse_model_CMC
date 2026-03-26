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
