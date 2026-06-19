"""
run_all.py
----------
Runs the entire NovaPay fraud detection project end-to-end, in order:
  1. generate_dataset.py   -> creates data/novapay_transactions.csv
  2. 01_eda.py              -> Task 1 + Task 2 (business framing + EDA charts)
  3. 02_model_pipeline.py   -> Task 3 + 4 + 5 (preprocessing, modeling, dollar impact)

Task 6 (CFO summary) is a static deliverable at outputs/task6_cfo_summary.txt
since it is a write-up, not a computation — but it's written FROM the
numbers this pipeline produces.

Usage (from the project root, with venv activated):
    python run_all.py

If you already have a real novapay_transactions.csv, place it in data/
BEFORE running this, and skip step 1 by running:
    python 01_eda.py
    python 02_model_pipeline.py
"""

import subprocess
import sys

STEPS = [
    ("Generating synthetic dataset", "generate_dataset.py"),
    ("Running EDA (Task 1 & 2)", "01_eda.py"),
    ("Running model pipeline (Task 3, 4 & 5)", "02_model_pipeline.py"),
]


def run_step(label, script):
    print("\n" + "=" * 70)
    print(f"STEP: {label}  ({script})")
    print("=" * 70)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n[FAILED] {script} exited with code {result.returncode}. Stopping.")
        sys.exit(1)


if __name__ == "__main__":
    for label, script in STEPS:
        run_step(label, script)

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETE")
    print("=" * 70)
    print("""
Check these outputs:
  outputs/task1_business_framing.txt
  outputs/figures/                      (10 charts)
  outputs/model_comparison.csv
  outputs/dollar_impact_report.txt
  outputs/task6_cfo_summary.txt
  outputs/best_model.pkl
""")
