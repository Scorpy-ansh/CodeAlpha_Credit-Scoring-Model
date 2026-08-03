# Credit Scoring Model

End-to-end credit risk classification pipeline that predicts default risk and compares multiple ML models (Logistic Regression, Decision Tree, Random Forest, XGBoost). The script supports a synthetic dataset (offline) and real datasets with automatic download (no login required).

## What This Project Does

- Loads a dataset (synthetic / UCI Taiwan / German Credit)
- Performs feature engineering + preprocessing
- Trains and evaluates multiple classifiers
- Generates plots (EDA, ROC curves, confusion matrices, feature importance, model comparison)
- Exports evaluation metrics to CSV

## Project Structure

- `credit_scoring_model.py` — main training/evaluation pipeline
- `datasets/` — local dataset cache (includes `german_credit.csv`)
- `credit_scoring_output_*/` — example output artifacts (plots + metrics)

## Setup

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

Optional (improves UCI download reliability):

```bash
pip install ucimlrepo
```

## Run

Synthetic dataset (default):

```bash
python credit_scoring_model.py
```

Real datasets (auto-downloaded):

```bash
python credit_scoring_model.py --dataset german_github
python credit_scoring_model.py --dataset uci_taiwan
```

## Outputs

The script writes results into `credit_scoring_output/`:

- `evaluation_results.csv` — metrics for all models
- `roc_curves.png` — ROC-AUC curves
- `confusion_matrices.png` — confusion matrices per model
- `model_comparison.png` — bar chart comparison across metrics
- `importance_*.png` — per-model feature importance
- `eda_plots.png` — dataset overview plots

Existing `credit_scoring_output_GERMAN/` and `credit_scoring_output_UCI_TAIWAN/` folders are example outputs already generated.

## Notes

- XGBoost is imported by default in `credit_scoring_model.py`, so `xgboost` must be installed.
- UCI Taiwan dataset fetching tries `ucimlrepo` first; if unavailable it falls back to a ZIP/XLS workflow.
