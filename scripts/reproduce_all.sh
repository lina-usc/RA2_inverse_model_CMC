#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# 1) Main env: your existing pipeline
# ----------------------------
source .venv/bin/activate

python -m eval.qc_forward --n 50 --seed 0 --out plots/qc_forward

python -m data.generate_dataset \
  --n 10000 --seed 0 \
  --out data/synthetic_cmc_dataset.h5 \
  --overwrite

python -m data.prepare_training_data \
  --in data/synthetic_cmc_dataset.h5 \
  --out-dir data_out \
  --overwrite

python -m data.make_splits \
  --data-out data_out \
  --seed 42 \
  --overwrite

python -m eval.qc_tokens --data-out data_out --out plots/qc_tokens --n 64

mkdir -p models_out

# Transformer ensemble (hybrid/erp/tfr)
for s in 0 1 2; do
  python -m models.train --run-name cmc --features hybrid --train-seed $s --split-seed 42 --posterior fullcov
  python -m models.train --run-name cmc --features erp    --train-seed $s --split-seed 42 --posterior fullcov
  python -m models.train --run-name cmc --features tfr    --train-seed $s --split-seed 42 --posterior fullcov
done

# Evaluate transformer ensemble (recommended: pass all 3 dirs)
python -m eval.evaluate_ensemble \
  --model-dirs models_out/cmc_hybrid_seed0 models_out/cmc_hybrid_seed1 models_out/cmc_hybrid_seed2 \
  --features hybrid \
  --split test \
  --out-dir plots/eval_hybrid_ens \
  --n-post-samples 200 \
  --seed 0

deactivate


# ----------------------------
# 2) SBI env: SNPE baselines + mismatch + patch ablation
# ----------------------------
source .venv_sbi/bin/activate

python scripts/neurips_upgrades.py all \
  --h5 data/synthetic_cmc_dataset.h5 \
  --data-out data_out \
  --split-seed 42 \
  --outdir results/neurips_sbi \
  --device cpu \
  --n-post-samples 200 \
  --do-patch-ablation

deactivate

echo ""
echo "[DONE] All outputs:"
echo "  - Transformer plots: plots/"
echo "  - SNPE baseline outputs: results/neurips_sbi/"
