
import argparse
import os
import time
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="Plot/output directories to bundle")
    ap.add_argument("--out", required=True, help="Output zip path, e.g. plots/prof_bundle.zip")
    ap.add_argument("--include-npz", action="store_true", help="Include .npz artifacts too (can be large)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".csv", ".json", ".md", ".txt"}
    if args.include_npz:
        exts.add(".npz")

    # Create README
    readme_lines = []
    readme_lines.append("# Professor bundle")
    readme_lines.append("")
    readme_lines.append(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    readme_lines.append("")
    readme_lines.append("Included directories:")
    for d in args.dirs:
        readme_lines.append(f"- {d}")
    readme_lines.append("")
    readme_lines.append("Notes:")
    readme_lines.append("- This zip contains plots + tables produced by Phase 5 scripts.")
    readme_lines.append("- Re-run evaluation/plots to reproduce if needed.")
    readme = "\n".join(readme_lines) + "\n"

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", readme)

        for d in args.dirs:
            root = Path(d)
            if not root.exists():
                print(f"[make_prof_bundle] WARN missing: {root}")
                continue
            for p in root.rglob("*"):
                if p.is_dir():
                    continue
                if p.suffix.lower() not in exts:
                    continue
                # store with relative path from repo root-ish
                arcname = str(p)
                z.write(p, arcname=arcname)

    print("[make_prof_bundle] wrote:", out_path)


if __name__ == "__main__":
    main()
