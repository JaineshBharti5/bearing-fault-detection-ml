"""
run_pipeline.py — Master orchestrator
=======================================
Runs the full bearing fault detection pipeline end-to-end:

  Step 1 → Generate CWRU-style vibration dataset
  Step 2 → Extract FFT / RMS / Kurtosis features
  Step 3 → Train Random Forest + compress to TFLite
  Step 4 → Simulate edge deployment

Run:
    python run_pipeline.py
"""

import sys
import time
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BANNER = """
╔══════════════════════════════════════════════════════════╗
║      WHEEL BEARING FAULT DETECTION SYSTEM                ║
║      CWRU Dataset • Random Forest • TFLite Edge AI       ║
╚══════════════════════════════════════════════════════════╝
"""

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

def main():
    print(BANNER)
    t_total = time.time()

    # ─ Step 1: Generate Dataset ─────────────────────────────────────────────
    section("STEP 1 / 4 — Generating Vibration Dataset")
    from data_generator import generate_dataset, CLASS_MAP
    X, y, t = generate_dataset()

    # ─ Step 2: Extract Features ─────────────────────────────────────────────
    section("STEP 2 / 4 — Feature Extraction (FFT · RMS · Kurtosis …)")
    from feature_extractor import build_feature_matrix
    df_feats = build_feature_matrix(X, y, CLASS_MAP)

    # ─ Step 3: Train + TFLite ───────────────────────────────────────────────
    section("STEP 3 / 4 — Training Random Forest + TFLite Compression")
    from train_model import main as train_main
    rf, scaler, tflite_path, results = train_main()

    # ─ Step 4: Edge Deployment ──────────────────────────────────────────────
    section("STEP 4 / 4 — Edge Deployment Simulation")
    from edge_deploy import main as edge_main
    edge_main()

    elapsed = time.time() - t_total
    print(f"\n✓  Full pipeline completed in {elapsed:.1f}s")
    print(f"\nOutput files:")
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git"]]
        for fn in files:
            if fn.endswith((".csv", ".pkl", ".tflite", ".json", ".py", ".md")):
                path = os.path.join(root, fn)
                size = os.path.getsize(path)
                print(f"  {path:<55} {size/1024:>8.1f} KB")


if __name__ == "__main__":
    main()
