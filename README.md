# RA2_inverse_model_CMC

# RA2 Inverse Model (CMC): End‑to‑End Reproduction

This repository reproduces the full experimental pipeline for the CMC inverse modeling project:

- Forward-model sanity checks (QC)
- Synthetic dataset generation (CMC simulator + regime rejection)
- Feature/token extraction (ERP/TFR/Hybrid) + fixed train/val/test splits
- Training inverse models (Transformer ensembles + BiLSTM baselines)
- Evaluation and diagnostics (calibration/reliability, SBC, PPC, metrics)
- Ablations:
  - Diagonal posterior head vs full covariance
  - Transformer WITHOUT per‑parameter tokens (“noparamtoken”)
- SBI baseline (SNPE) + mismatch robustness + PPC for SNPE

All commands below are designed to be run from the **repo root**.

---

## Repository layout (what’s source vs generated)

### Source (tracked)
- `data/` dataset generation + preprocessing code
- `models/` training code + architectures
- `eval/` evaluation + plotting/diagnostics
- `sim/` CMC forward simulator
- `scripts/` convenience/NeurIPS/SBI scripts
- `config/` config templates

### Generated (do not commit)
- `data/synthetic_cmc_dataset*.h5`
- `data_out/` (token features + splits)
- `models_out/` (trained models + scalers + logs)
- `plots/` (QC + evaluation figures + summary CSV/JSON)
- `results/` (SBI/SNPE baseline outputs)

---

