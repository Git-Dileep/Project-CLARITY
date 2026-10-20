"""Master entry point to execute the complete detection workflow end-to-end.

Steps:
1. Load WorkstreamConfig
2. Dataset Audit (KOU-01)
3. Traditional 3-Model Ensemble (KOU-02)
4. Deep Learning 3-Model Ensemble (KOU-03)
5. SHAP Explanations & 20-Alert Pilot Export with Strict Contract Validation (KOU-04 & KOU-05)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import WorkstreamConfig
from .audit_datasets import run_audit
from .data_loader import load_dataset
from .preprocessing import prepare_data_and_splits
from .train_traditional import train_traditional_models
from .train_dl import train_dl_models, TORCH_AVAILABLE
from .export_pilot import export_pilot_alerts


def run_pipeline(config_path: str = "config/reference.json") -> int:
    t_start = time.perf_counter()
    print("=" * 70)
    print("PROJECT CLARITY — KOUSHIK DETECTION WORKSTREAM RUNNER")
    print("=" * 70)

    cfg = WorkstreamConfig.load(config_path)
    print(f"Configuration: {config_path}")
    print(f"Dataset Name:  {cfg.dataset_name} (Synthetic Reference: {cfg.is_synthetic})")
    print(f"Input Path:    {cfg.input_path}")
    print(f"Target Column: {cfg.target_column}")

    # Step 1: Audit
    print("\n[STEP 1/4] Running Dataset Audit...")
    audit_res = run_audit(cfg)
    if audit_res.get("status") == "BLOCKED — DATASET NOT AVAILABLE":
        print("\n[BLOCKED] Dataset is not available. Stopping pipeline as expected for unpopulated dataset.")
        return 0

    # Step 2: Load Data & Preprocess
    print("\n[STEP 2/4] Loading Data and Creating Splits...")
    df, _ = load_dataset(cfg)
    splits = prepare_data_and_splits(df, cfg)

    # Step 3: Train Traditional Ensemble
    print("\n[STEP 3/4] Training Traditional 3-Model Ensemble (RF, XGBoost, MLP)...")
    trad_res = train_traditional_models(splits, cfg)

    # Step 4: Train DL Ensemble
    if TORCH_AVAILABLE and cfg.dl.enabled:
        print("\n[STEP 4/4] Training DL 3-Model Ensemble (CNN, LSTM, Transformer)...")
        dl_res = train_dl_models(splits, cfg)
    else:
        print("\n[STEP 4/4] Skipping DL ensemble (PyTorch not available or disabled).")

    # Step 5: Export Pilot & Validate
    print("\n[STEP 5/5] Exporting Stratified 20-Alert Pilot and Validating Contracts...")
    pilot_res = export_pilot_alerts(cfg)

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 70)
    print(f"SUCCESS: Pipeline execution completed in {total_time:.2f} seconds.")
    print("All outputs, metrics, and schema-validated alert pilots generated.")
    print("=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run complete detection workstream pipeline.")
    parser.add_argument("--config", default="config/reference.json", help="Path to config file")
    args = parser.parse_args()

    sys.exit(run_pipeline(args.config))


if __name__ == "__main__":
    main()
