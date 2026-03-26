#!/usr/bin/env bash
set -euo pipefail

# Reproduce ALL manuscript results using Morlet wavelet TFR tokens.
#
# Usage:
#   bash reproduce_morlet_all.sh
#
# Optional overrides (example):
#   DATA_H5=data/synthetic_cmc_dataset.h5 \
#   DATA_OUT=data_out_morlet \
#   MODELS_OUT=models_out_morlet \
#   PLOTS_OUT=plots_morlet \
#   FIG_OUT=figures_morlet \
#   RUN_NAME=cmc_morlet \
#   SEEDS="0 1 2" \
#   SPLIT_SEED=42 \
#   FORCE=0 \
#   SKIP_TRAIN=0 \
#   SKIP_EVAL=0 \
#   SKIP_PLOTS=0 \
#   bash reproduce_morlet_all.sh

DATA_H5=${DATA_H5:-data/synthetic_cmc_dataset.h5}
DATA_OUT=${DATA_OUT:-data_out_morlet}
MODELS_OUT=${MODELS_OUT:-models_out_morlet}
PLOTS_OUT=${PLOTS_OUT:-plots_morlet}
FIG_OUT=${FIG_OUT:-figures_morlet}
RUN_NAME=${RUN_NAME:-cmc_morlet}

SPLIT_SEED=${SPLIT_SEED:-42}
SEEDS_STR=${SEEDS:-"0 1 2"}
read -r -a SEEDS <<<"$SEEDS_STR"

FORCE=${FORCE:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
SKIP_EVAL=${SKIP_EVAL:-0}
SKIP_PLOTS=${SKIP_PLOTS:-0}

mkdir -p "$DATA_OUT" "$MODELS_OUT" "$PLOTS_OUT" "$FIG_OUT"
export DATA_H5 DATA_OUT MODELS_OUT PLOTS_OUT FIG_OUT RUN_NAME SPLIT_SEED

# -----------------------------------------------------------------------------
# Matplotlib defaults for paper-quality, readable figures.
# This fixes the "figure font appears like size 3" complaint.
# -----------------------------------------------------------------------------

echo "=== [0/7] Matplotlib paper style ==="
if [[ ! -f matplotlibrc || "$FORCE" -eq 1 ]]; then
  cat > matplotlibrc <<'EOF'
font.size: 9
axes.titlesize: 9
axes.labelsize: 9
xtick.labelsize: 8
ytick.labelsize: 8
legend.fontsize: 8
figure.titlesize: 10
savefig.dpi: 300
figure.dpi: 150
EOF
fi
export MATPLOTLIBRC="$PWD/matplotlibrc"
export MPLBACKEND=Agg

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

arch_suffix() {
  case "$1" in
    paramtoken) echo "" ;;
    bilstm) echo "_bilstm" ;;
    noparamtoken) echo "_noparamtoken" ;;
    *) echo "ERROR: unknown arch '$1'" >&2; exit 1 ;;
  esac
}

run_dir() {
  local feat=$1
  local arch=$2
  local posterior=$3
  local seed=$4
  echo "$MODELS_OUT/${RUN_NAME}_${feat}$(arch_suffix "$arch")_${posterior}_seed${seed}"
}

run_dirs_three() {
  local feat=$1
  local arch=$2
  local posterior=$3
  echo "$(run_dir "$feat" "$arch" "$posterior" 0)" \
       "$(run_dir "$feat" "$arch" "$posterior" 1)" \
       "$(run_dir "$feat" "$arch" "$posterior" 2)"
}

have_checkpoint() {
  local d=$1
  compgen -G "$d/*_best.keras" > /dev/null || compgen -G "$d/*_final.keras" > /dev/null
}

train_one() {
  local feat=$1
  local arch=$2
  local posterior=$3
  local seed=$4
  local d
  d=$(run_dir "$feat" "$arch" "$posterior" "$seed")

  if [[ "$FORCE" -eq 1 && -d "$d" ]]; then
    rm -rf "$d"
  fi

  if [[ "$FORCE" -eq 0 && -d "$d" ]] && have_checkpoint "$d"; then
    echo "[train] skip existing: $d"
    return 0
  fi

  echo "[train] $feat / $arch / $posterior / seed=$seed"
  python -m models.train \
    --data-out "$DATA_OUT" \
    --models-out "$MODELS_OUT" \
    --run-name "$RUN_NAME" \
    --features "$feat" \
    --arch "$arch" \
    --posterior "$posterior" \
    --train-seed "$seed" \
    --split-seed "$SPLIT_SEED"
}

eval_one() {
  local feat=$1
  local arch=$2
  local posterior=$3
  local out_dir=$4

  if [[ "$FORCE" -eq 1 && -d "$out_dir" ]]; then
    rm -rf "$out_dir"
  fi

  if [[ "$FORCE" -eq 0 && -f "$out_dir/eval_test_outputs.npz" ]]; then
    echo "[eval] skip existing: $out_dir"
    return 0
  fi

  mkdir -p "$out_dir"

  # Prefer module form if eval/ exists (recommended layout).
  if [[ -f eval/evaluate_ensemble.py ]]; then
    python -m eval.evaluate_ensemble \
      --data-out "$DATA_OUT" \
      --model-dirs $(run_dirs_three "$feat" "$arch" "$posterior") \
      --features "$feat" \
      --arch "$arch" \
      --split test \
      --n-eval 1500 \
      --n-post-samples 200 \
      --seed 0 \
      --out-dir "$out_dir"
  else
    python evaluate_ensemble.py \
      --data-out "$DATA_OUT" \
      --model-dirs $(run_dirs_three "$feat" "$arch" "$posterior") \
      --features "$feat" \
      --arch "$arch" \
      --split test \
      --n-eval 1500 \
      --n-post-samples 200 \
      --seed 0 \
      --out-dir "$out_dir"
  fi
}

# -----------------------------------------------------------------------------
# [1/7] Feature extraction
# -----------------------------------------------------------------------------

echo "=== [1/7] Feature extraction (Morlet) ==="
if [[ "$FORCE" -eq 1 || ! -f "$DATA_OUT/features.npy" ]]; then
  python -m data.prepare_training_data \
    --in "$DATA_H5" \
    --out-dir "$DATA_OUT" \
    --overwrite \
    --tfr-method morlet \
    --tfr-fmin 4 \
    --tfr-fmax 40 \
    --morlet-n-freqs 48 \
    --morlet-cycles-low 4 \
    --morlet-cycles-high 8 \
    --morlet-decim 1 \
    --morlet-n-jobs 1
else
  echo "(skip) $DATA_OUT/features.npy already exists (set FORCE=1 to regenerate)"
fi

# -----------------------------------------------------------------------------
# [2/7] Splits
# -----------------------------------------------------------------------------

echo "=== [2/7] Train/val/test splits ==="
if python -m data.make_splits --help >/dev/null 2>&1; then
  # Try the newer signature first; if it fails, try no args.
  python -m data.make_splits --data-out "$DATA_OUT" --overwrite >/dev/null 2>&1 || \
    python -m data.make_splits --overwrite >/dev/null 2>&1 || \
    echo "(info) data.make_splits unavailable; splits will be created automatically by models.train"
else
  echo "(info) data.make_splits unavailable; splits will be created automatically by models.train"
fi

# -----------------------------------------------------------------------------
# [3/7] Train Transformer (paramtoken, fullcov)
# -----------------------------------------------------------------------------

echo "=== [3/7] Train Transformer (paramtoken, fullcov) for ERP/TFR/Hybrid x 3 seeds ==="
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  for feat in erp tfr hybrid; do
    for seed in "${SEEDS[@]}"; do
      train_one "$feat" paramtoken fullcov "$seed"
    done
  done
else
  echo "(skip) SKIP_TRAIN=1"
fi

# -----------------------------------------------------------------------------
# [4/7] Train BiLSTM baseline (fullcov)
# -----------------------------------------------------------------------------

echo "=== [4/7] Train BiLSTM baseline (fullcov) for ERP/TFR/Hybrid x 3 seeds ==="
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  for feat in erp tfr hybrid; do
    for seed in "${SEEDS[@]}"; do
      train_one "$feat" bilstm fullcov "$seed"
    done
  done
else
  echo "(skip) SKIP_TRAIN=1"
fi

# -----------------------------------------------------------------------------
# [5/7] Train key ablations on Hybrid
# -----------------------------------------------------------------------------

echo "=== [5/7] Train key ablations on Hybrid: diag posterior + no-param-token ==="
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  for seed in "${SEEDS[@]}"; do
    train_one hybrid paramtoken diag "$seed"
    train_one hybrid noparamtoken fullcov "$seed"
  done
else
  echo "(skip) SKIP_TRAIN=1"
fi

# -----------------------------------------------------------------------------
# [6/7] Evaluate ensembles
# -----------------------------------------------------------------------------

echo "=== [6/7] Evaluate ensembles + write raw outputs ==="
if [[ "$SKIP_EVAL" -eq 0 ]]; then
  # Transformer (paramtoken) ensembles
  eval_one erp   paramtoken fullcov "$PLOTS_OUT/eval_erp_ens"
  eval_one tfr   paramtoken fullcov "$PLOTS_OUT/eval_tfr_ens"
  eval_one hybrid paramtoken fullcov "$PLOTS_OUT/eval_hybrid_ens"

  # BiLSTM ensembles
  eval_one erp   bilstm fullcov "$PLOTS_OUT/eval_erp_bilstm_ens"
  eval_one tfr   bilstm fullcov "$PLOTS_OUT/eval_tfr_bilstm_ens"
  eval_one hybrid bilstm fullcov "$PLOTS_OUT/eval_hybrid_bilstm_ens"

  # Hybrid ablations
  eval_one hybrid paramtoken diag "$PLOTS_OUT/eval_hybrid_diag_ens"
  eval_one hybrid noparamtoken fullcov "$PLOTS_OUT/eval_hybrid_noparamtoken_ens"
else
  echo "(skip) SKIP_EVAL=1"
fi

# -----------------------------------------------------------------------------
# [7/7] Generate manuscript figures
# -----------------------------------------------------------------------------

echo "=== [7/7] Generate manuscript figures ==="
if [[ "$SKIP_PLOTS" -ne 0 ]]; then
  echo "(skip) SKIP_PLOTS=1"
  echo "=== DONE (plots skipped). Raw eval outputs in: $PLOTS_OUT ==="
  exit 0
fi

# Feature-set comparison (Transformer)
if [[ -f eval/compare_feature_sets.py ]]; then
  python -m eval.compare_feature_sets \
    --data-out "$DATA_OUT" \
    --erp-dir "$PLOTS_OUT/eval_erp_ens" \
    --tfr-dir "$PLOTS_OUT/eval_tfr_ens" \
    --hybrid-dir "$PLOTS_OUT/eval_hybrid_ens" \
    --split test \
    --out-dir "$FIG_OUT"
else
  python compare_feature_sets.py \
    --data-out "$DATA_OUT" \
    --erp-dir "$PLOTS_OUT/eval_erp_ens" \
    --tfr-dir "$PLOTS_OUT/eval_tfr_ens" \
    --hybrid-dir "$PLOTS_OUT/eval_hybrid_ens" \
    --split test \
    --out-dir "$FIG_OUT"
fi

# Delta metrics (Transformer - BiLSTM)
python - <<'PY'
import os
import pandas as pd

plots_out = os.environ.get('PLOTS_OUT', 'plots_morlet')
fig_out = os.environ.get('FIG_OUT', 'figures_morlet')

os.makedirs(fig_out, exist_ok=True)

def read_metrics(dir_path: str) -> pd.DataFrame:
    csv_path = os.path.join(dir_path, 'metrics_test.csv')
    df = pd.read_csv(csv_path)
    return df[['param','pearson_mean','rmse_norm_mean']].set_index('param')

pairs = {
    'erp':   (f'{plots_out}/eval_erp_ens',        f'{plots_out}/eval_erp_bilstm_ens'),
    'tfr':   (f'{plots_out}/eval_tfr_ens',        f'{plots_out}/eval_tfr_bilstm_ens'),
    'hybrid':(f'{plots_out}/eval_hybrid_ens',     f'{plots_out}/eval_hybrid_bilstm_ens'),
}

out_rows = []
for feat,(tr_dir, bl_dir) in pairs.items():
    tr = read_metrics(tr_dir)
    bl = read_metrics(bl_dir)
    d = tr - bl
    d = d.rename(columns={
        'pearson_mean': f'd_pearson_{feat}',
        'rmse_norm_mean': f'd_rmse_norm_{feat}',
    })
    out_rows.append(d)

out = pd.concat(out_rows, axis=1)
out.reset_index(inplace=True)
out_path = os.path.join(fig_out, 'delta_metrics_test.csv')
out.to_csv(out_path, index=False)
print('Wrote', out_path)
PY

if [[ -f eval/plot_arch_compare.py ]]; then
  python -m eval.plot_arch_compare \
    --delta-csv "$FIG_OUT/delta_metrics_test.csv" \
    --out "$FIG_OUT" \
    --tag test
else
  python plot_arch_compare.py \
    --delta-csv "$FIG_OUT/delta_metrics_test.csv" \
    --out "$FIG_OUT" \
    --tag test
fi

if [[ -f eval/plot_results.py ]]; then
  python -m eval.plot_results \
    --eval-npz "$PLOTS_OUT/eval_hybrid_ens/eval_test_outputs.npz" \
    --out "$FIG_OUT"
else
  python plot_results.py \
    --eval-npz "$PLOTS_OUT/eval_hybrid_ens/eval_test_outputs.npz" \
    --out "$FIG_OUT"
fi

if [[ -f eval/plot_reliability.py ]]; then
  python -m eval.plot_reliability \
    --eval-npz "$PLOTS_OUT/eval_hybrid_ens/eval_test_outputs.npz" \
    --out "$FIG_OUT"
else
  python plot_reliability.py \
    --eval-npz "$PLOTS_OUT/eval_hybrid_ens/eval_test_outputs.npz" \
    --out "$FIG_OUT"
fi

if [[ -f eval/plot_sbc.py ]]; then
  python -m eval.plot_sbc \
    --eval-npz "$PLOTS_OUT/eval_hybrid_ens/eval_test_outputs.npz" \
    --out "$FIG_OUT" \
    --bins 20
else
  python plot_sbc.py \
    --eval-npz "$PLOTS_OUT/eval_hybrid_ens/eval_test_outputs.npz" \
    --out "$FIG_OUT" \
    --bins 20
fi

cp "$PLOTS_OUT/eval_hybrid_ens/nllz_hist_test.png" "$FIG_OUT/nllz_hist_test.png"

# Posterior predictive checks
if [[ -f eval/ppc.py ]]; then
  python -m eval.ppc \
    --eval-npz "$PLOTS_OUT/eval_hybrid_ens/eval_test_outputs.npz" \
    --out "$FIG_OUT" \
    --n-examples 6 \
    --n-ppc-sims 200 \
    --seed 0 \
    --h5 "$DATA_H5"
else
  python ppc.py \
    --eval-npz "$PLOTS_OUT/eval_hybrid_ens/eval_test_outputs.npz" \
    --out "$FIG_OUT" \
    --n-examples 6 \
    --n-ppc-sims 200 \
    --seed 0 \
    --h5 "$DATA_H5"
fi

# Robustness sweeps: this script historically hard-coded ./data_out.
if [[ -f eval/generalization_sweeps.py ]]; then
  if python -m eval.generalization_sweeps --help 2>&1 | grep -q -- "--data-out"; then
    python -m eval.generalization_sweeps \
      --data-out "$DATA_OUT" \
      --model-dirs $(run_dirs_three hybrid paramtoken fullcov) \
      --features hybrid \
      --split test \
      --n-eval 1500 \
      --n-post-samples 200 \
      --seed 0 \
      --out-dir "$FIG_OUT"
  else
    if [[ -e data_out && ! -L data_out ]]; then
      echo "ERROR: ./data_out exists and is not a symlink."
      echo "  Rename it (e.g. mv data_out data_out_stft) or delete it, then rerun."
      exit 1
    fi
    ln -sfn "$DATA_OUT" data_out
    python -m eval.generalization_sweeps \
      --model-dirs $(run_dirs_three hybrid paramtoken fullcov) \
      --features hybrid \
      --split test \
      --n-eval 1500 \
      --n-post-samples 200 \
      --seed 0 \
      --out-dir "$FIG_OUT"
  fi
else
  echo "(warn) eval/generalization_sweeps.py not found; skipping robustness sweeps"
fi

# Information gain (bits) per parameter
python - <<'PY'
import os
import numpy as np
import pandas as pd

plots_out = os.environ.get('PLOTS_OUT', 'plots_morlet')
fig_out = os.environ.get('FIG_OUT', 'figures_morlet')
data_out = os.environ.get('DATA_OUT', 'data_out_morlet')

eval_npz = np.load(os.path.join(plots_out, 'eval_hybrid_ens', 'eval_test_outputs.npz'), allow_pickle=True)
meta = np.load(os.path.join(data_out, 'param_meta.npz'), allow_pickle=True)

param_names = []
for x in meta['param_names']:
    if isinstance(x, (bytes, np.bytes_)):
        param_names.append(x.decode('utf-8'))
    else:
        param_names.append(str(x))

theta_true = eval_npz['theta_true']
post = eval_npz['theta_samples']

low = meta['prior_low'].astype(float)
high = meta['prior_high'].astype(float)

P = theta_true.shape[1]
bins = 60

def hist_prob(x, edges):
    h, _ = np.histogram(x, bins=edges)
    h = h.astype(float) + 1e-12
    return h / h.sum()

ig_bits = []
for i in range(P):
    edges = np.linspace(low[i], high[i], bins + 1)
    p = hist_prob(theta_true[:, i], edges)
    kls = []
    for n in range(theta_true.shape[0]):
        q = hist_prob(post[n, :, i], edges)
        kls.append(np.sum(q * (np.log(q) - np.log(p))))
    ig_bits.append(np.mean(kls) / np.log(2.0))

out = pd.DataFrame({'param': param_names, 'info_gain_bits': ig_bits})
os.makedirs(fig_out, exist_ok=True)
csv_path = os.path.join(fig_out, 'info_gain_bits_test.csv')
out.to_csv(csv_path, index=False)
print('Wrote', csv_path)

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.bar(out['param'], out['info_gain_bits'])
    plt.ylabel('Information gain (bits)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig_path = os.path.join(fig_out, 'info_gain_bits_test.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print('Wrote', fig_path)
except Exception as e:
    print('Skipping info_gain_bits_test.png (matplotlib error):', e)
PY

echo "=== DONE. Figures in: $FIG_OUT ; raw eval outputs in: $PLOTS_OUT ==="