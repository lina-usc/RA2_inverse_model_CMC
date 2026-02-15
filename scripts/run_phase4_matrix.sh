#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="cmc"
SPLIT_SEED=42
POSTERIOR="fullcov"

SEEDS=(0 1 2)
FEATURES=(erp tfr hybrid)

echo "=== Train main model (paramtoken transformer) ==="
for feat in "${FEATURES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python -m models.train \
      --run-name "${RUN_NAME}" \
      --arch paramtoken \
      --features "${feat}" \
      --seed "${seed}" \
      --split-seed "${SPLIT_SEED}" \
      --posterior "${POSTERIOR}"
  done
done

echo "=== Train baseline (BiLSTM) ==="
for feat in "${FEATURES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    python -m models.train \
      --run-name "${RUN_NAME}" \
      --arch bilstm \
      --features "${feat}" \
      --seed "${seed}" \
      --split-seed "${SPLIT_SEED}" \
      --posterior "${POSTERIOR}"
  done
done
