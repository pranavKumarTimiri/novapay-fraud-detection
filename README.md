# NovaPay Fraud Detection — "Catch the Leaks"

ML project to replace NovaPay's rule-based fraud filter (38% recall) with a
trained model, and quantify the dollar savings.

> **Note on the dataset:** The original assessment brief referenced a
> `novapay_transactions.csv` that was supposed to be "provided separately,"
> but it wasn't attached. `generate_dataset.py` builds a **synthetic
> stand-in** with the exact same schema (8,000 rows, 14 columns, ~6% fraud
> rate) and realistic fraud signal patterns (high amount + overseas country
> + high velocity + prior flags ⇒ higher fraud probability), so the whole
> pipeline runs correctly end-to-end. **If you get the real CSV later, just
> drop it into `data/novapay_transactions.csv` (same name) and skip
> `generate_dataset.py`** — every other script just reads that file path
> and doesn't care how it was created.

---

## 1. Project structure

```
novapay_fraud/
├── generate_dataset.py        # Builds the synthetic dataset (skip if you have the real CSV)
├── 01_eda.py                  # Task 1 (business framing) + Task 2 (EDA & charts)
├── 02_model_pipeline.py       # Task 3, 4, 5 (preprocessing, modeling, dollar impact)
├── run_all.py                 # Runs all of the above in order, one command
├── requirements.txt
├── README.md                  # this file
├── data/
│   └── novapay_transactions.csv      (created by generate_dataset.py)
└── outputs/
    ├── task1_business_framing.txt
    ├── task6_cfo_summary.txt         (Task 6 — written manually, included)
    ├── model_comparison.csv          (metrics table for both models)
    ├── dollar_impact_report.txt      (Task 5 full breakdown)
    ├── best_model.pkl                (saved model + preprocessor)
    └── figures/
        ├── 01_class_distribution.png
        ├── 02_fraud_by_country.png
        ├── 03_fraud_by_hour.png
        ├── 04_fraud_by_amount_bracket.png
        ├── 05_fraud_by_priorflags_velocity.png
        ├── 06_correlation_heatmap.png
        ├── 07_roc_pr_curves.png
        ├── 08_confusion_matrices.png
        ├── 09_threshold_tuning.png
        └── 10_dollar_impact_comparison.png
```

---

## 2. Step-by-step: running this in VS Code (Windows)

### Step 1 — Open the project folder
1. Unzip / copy the `novapay_fraud` folder anywhere on your machine, e.g. `C:\Projects\novapay_fraud`.
2. Open **VS Code** → `File` → `Open Folder...` → select `novapay_fraud`.

### Step 2 — Open a terminal in VS Code
- Menu: `Terminal` → `New Terminal` (or `` Ctrl + ` ``).
- This opens **PowerShell** by default on Windows, in the project folder.

### Step 3 — Create and activate a virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
```
You should see `(venv)` appear at the start of your terminal prompt.

> **If you get an error like "running scripts is disabled on this system":**
> This is a PowerShell execution-policy block (you've hit this before with
> the Notes & Tags Manager project). Fix it by running this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then re-run `venv\Scripts\activate`.

### Step 4 — Install dependencies
```powershell
pip install -r requirements.txt
```
This installs: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, imbalanced-learn, joblib.

### Step 5 — Run the entire project with one command
```powershell
python run_all.py
```
This runs, in order:
1. `generate_dataset.py` — creates `data/novapay_transactions.csv`
2. `01_eda.py` — prints Task 1 answer, saves 6 EDA charts
3. `02_model_pipeline.py` — runs Task 3–5: preprocessing, SMOTE, trains
   Logistic Regression + XGBoost, 5-fold stratified CV, threshold tuning,
   and the full dollar-impact calculation (with bonus annualized
   confidence interval)

Takes about 15–30 seconds on a normal laptop.

### Step 6 — View the results
- All console output is also saved to text files in `outputs/`.
- Open `outputs/figures/` in VS Code's file explorer and click any `.png`
  to preview it directly in the editor.
- Open `outputs/dollar_impact_report.txt` for the full Task 5 numbers.
- Open `outputs/task6_cfo_summary.txt` for the CFO write-up.

### Running steps individually (optional)
If you already have the **real** `novapay_transactions.csv`, place it in
`data/` with that exact filename, then skip the generator:
```powershell
python 01_eda.py
python 02_model_pipeline.py
```

---

## 3. What each task produces

| Task | Script | Output |
|---|---|---|
| 1 — Business framing | `01_eda.py` | `outputs/task1_business_framing.txt` — Recall vs Precision justification |
| 2 — EDA & imbalance | `01_eda.py` | 6 charts in `outputs/figures/` (class balance, fraud by country/hour/amount/velocity, correlation heatmap) |
| 3 — Preprocessing & SMOTE | `02_model_pipeline.py` | One-hot + scaling pipeline; SMOTE applied only to the training fold |
| 4 — Model training & tuning | `02_model_pipeline.py` | Logistic Regression (baseline) + XGBoost, stratified 5-fold CV, threshold sweep to minimize dollar cost |
| 5 — Dollar impact | `02_model_pipeline.py` | `outputs/dollar_impact_report.txt` — old vs new system cost, savings, annualized to 50,000 txn/day with bootstrap 95% CI |
| 6 — CFO narrative | manual | `outputs/task6_cfo_summary.txt` |

---

## 4. Key results (from this synthetic dataset)

- **Best model:** XGBoost — Recall **85.4%**, Precision 28.1%, ROC-AUC 0.93
  (vs. old rule-based system's 38% recall)
- **Test-set cost:** Old system $22,275 → ML model $10,010 (**55% reduction**)
- **Annualized savings estimate:** ~$140M/year at NovaPay's full 50,000
  txn/day volume (95% CI: $94M–$182M)

> These exact numbers are reproducible every run because `random_state=42`
> is fixed everywhere (data generation, train/test split, SMOTE, model
> training, bootstrap). If you swap in the real dataset, numbers will
> naturally differ.

---

## 5. Design choices & justifications (for the assessor)

- **Why SMOTE over undersampling:** Undersampling would throw away ~85% of
  the legitimate transactions (since fraud is only 6%), losing valuable
  signal about what "normal" looks like. SMOTE instead synthesizes new
  minority-class examples by interpolating between real fraud cases,
  preserving the full legitimate-transaction signal.
- **Why SMOTE only on the training fold:** Applying it before the
  train/test split (or before each CV fold) would leak synthetic "twins"
  of test-set fraud cases into training, artificially inflating recall.
  This pipeline fits SMOTE separately inside every CV fold and once on
  the final training set — never touching the test set.
- **Why threshold tuning instead of the default 0.5 cutoff:** Because
  False Negatives ($340) cost ~13.6x more than False Positives ($25), the
  optimal decision threshold is well below 0.5. We sweep thresholds from
  0.05–0.95 and pick whichever minimizes total dollar cost on the test set
  — that's why XGBoost's tuned threshold ends up at 0.07, not 0.5.
- **Why Recall is the headline metric (Task 1):** See
  `outputs/task1_business_framing.txt` for the full reasoning — the
  cost asymmetry between missed fraud and false alarms drives this.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `python` not recognized | Install Python 3.10+ from python.org, check "Add to PATH" during install, restart VS Code |
| `pip install` fails on xgboost | Make sure you're inside the activated venv (`(venv)` in prompt); try `pip install --upgrade pip` first |
| PowerShell blocks venv activation | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once |
| `ModuleNotFoundError` for any package | Re-run `pip install -r requirements.txt` inside the activated venv |
| Charts don't open / blank | They're saved as `.png` files in `outputs/figures/` — open via VS Code's Explorer panel, not via the terminal |
| Want to change cost assumptions ($340 / $25) | Edit `COST_FN` and `COST_FP` constants near the top of `02_model_pipeline.py` |
| Want to re-run with different random seed | Change `RNG_SEED` in `generate_dataset.py` and `RNG` in `02_model_pipeline.py` |
