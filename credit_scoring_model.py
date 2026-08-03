# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import warnings
import os
import urllib.request
import zipfile
import argparse

warnings.filterwarnings('ignore')

OUTPUT_DIR = "credit_scoring_output"
DATA_DIR = "datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# =====================================================================
# REAL-DATASET SOURCES  (all direct URLs — no login, no CAPTCHA, 2026)
# =====================================================================
DATASET_REGISTRY = {
    "synthetic": {"description": "Generated synthetic data (default, no download)"},
    "uci_taiwan": {
        "description": "UCI Default of Credit Card Clients (Taiwan, 30k rows, 23 features)",
        "url": "https://cdn.uci-ics-mlr-prod.aws.uci.edu/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip",
        "zip": True,
        "inner_file": "default of credit card clients.xls",
        "target": "Defaulted",
    },
    "german_github": {
        "description": "UCI German Credit (clean CSV on GitHub, 1000 rows, 21 columns)",
        "url": "https://raw.githubusercontent.com/DavidDeVegaMartin/German_Credit_Data_UCI/main/credit.csv",
        "zip": False,
        "filename": "german_credit.csv",
        "target": "Defaulted",
    },
}


# =====================================================================
# DATASET: Auto-download + loaders
# =====================================================================
def _download(url: str, dest_path: str, label: str):
    if os.path.exists(dest_path):
        print(f"  [OK] Using cached {label}: {dest_path}")
        return
    print(f"  [..] Downloading {label} ...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  [OK] Saved to {dest_path}")
    except Exception as exc:
        raise RuntimeError(
            f"Download failed for {label}. Check internet or use 'synthetic'.\nError: {exc}"
        )


def _extract_zip(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def load_uci_taiwan() -> pd.DataFrame:
    cfg = DATASET_REGISTRY["uci_taiwan"]
    UCI_ID = 350

    df = None
    # Strategy 1: use ucimlrepo package (UCI's official Python fetcher - no xls issues)
    try:
        from ucimlrepo import fetch_ucirepo
        print("  [..] Fetching via ucimlrepo (UCI official fetcher)...")
        ds = fetch_ucirepo(id=UCI_ID)
        X_df = ds.data.features.copy()
        y_s = ds.data.targets.iloc[:, 0].copy()
        X_df["Defaulted"] = y_s.values
        df = X_df
    except Exception as exc1:
        # Strategy 2: fall back to the zip + xls + xlrd==1.2.0 manual read
        print(f"  [!!] ucimlrepo failed ({exc1}); trying ZIP/XLS fallback...")
        zip_path = os.path.join(DATA_DIR, "uci_taiwan_default.zip")
        extracted_dir = os.path.join(DATA_DIR, "uci_taiwan_extracted")
        xls_path = os.path.join(extracted_dir, cfg["inner_file"])
        if not os.path.exists(xls_path):
            _download(cfg["url"], zip_path, "UCI Taiwan credit-card default (5.3 MB)")
            print(f"  [..] Extracting {zip_path} ...")
            os.makedirs(extracted_dir, exist_ok=True)
            _extract_zip(zip_path, extracted_dir)
        df = pd.read_excel(xls_path, header=1)
        df = df.drop(columns=["ID"], errors="ignore")
        df = df.rename(columns={"default payment next month": "Defaulted"})

    # Canonicalise column names (both strategies yield same set after rename)
    rename_map = {}
    for col in df.columns:
        cl = str(col).strip()
        if cl.lower() in ("limit_bal", "x1"): rename_map[col] = "LIMIT_BAL"
        elif cl.lower() in ("sex", "x2"): rename_map[col] = "SEX"
        elif cl.lower() in ("education", "x3"): rename_map[col] = "EDUCATION"
        elif cl.lower() in ("marriage", "x4"): rename_map[col] = "MARRIAGE"
        elif cl.lower() in ("age", "x5"): rename_map[col] = "AGE"
        elif cl.lower().startswith("pay_") or cl.lower().startswith("pay ") or (cl.lower() in (f"x{i}" for i in range(6,12))):
            rename_map[col] = cl.upper().replace(" ", "_")
        elif cl.lower().startswith("bill_amt") or (cl.lower() in (f"x{i}" for i in range(12,18))):
            rename_map[col] = cl.upper().replace(" ", "_")
        elif cl.lower().startswith("pay_amt") or (cl.lower() in (f"x{i}" for i in range(18,24))):
            rename_map[col] = cl.upper().replace(" ", "_")
    if rename_map:
        df = df.rename(columns=rename_map)

    if "SEX" in df.columns:
        sex_map = {1: "Male", 2: "Female", "male": "Male", "female": "Female"}
        df["SEX"] = df["SEX"].map(sex_map).fillna("Other")
    if "EDUCATION" in df.columns:
        edu_map = {1: "Graduate", 2: "University", 3: "High_School",
                   4: "Other", 5: "Other", 6: "Other", 0: "Other"}
        df["EDUCATION"] = pd.to_numeric(df["EDUCATION"], errors="coerce").fillna(0).astype(int).map(edu_map).fillna("Other")
    if "MARRIAGE" in df.columns:
        mar_map = {1: "Married", 2: "Single", 3: "Other", 0: "Other"}
        df["MARRIAGE"] = pd.to_numeric(df["MARRIAGE"], errors="coerce").fillna(0).astype(int).map(mar_map).fillna("Other")

    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                 "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    payamt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                   "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    for c in pay_cols + bill_cols + payamt_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["Total_Bill"] = df[[c for c in bill_cols if c in df.columns]].sum(axis=1)
    df["Total_Paid"] = df[[c for c in payamt_cols if c in df.columns]].sum(axis=1)
    df["Max_Late_Months"] = df[[c for c in pay_cols if c in df.columns]].max(axis=1).clip(lower=0)
    df["Num_Late_Payments"] = (df[[c for c in pay_cols if c in df.columns]] > 0).sum(axis=1)
    df["Avg_Bill_Statement"] = df[[c for c in bill_cols if c in df.columns]].mean(axis=1)
    df["Avg_Payment_Per_Month"] = df[[c for c in payamt_cols if c in df.columns]].mean(axis=1)
    df["Payment_Ratio"] = df["Total_Paid"] / (df["Total_Bill"].abs() + 1)
    df["Debt_to_Limit"] = df["Total_Bill"].clip(lower=0) / (df["LIMIT_BAL"].astype(float) + 1)

    df = df.rename(columns={
        "LIMIT_BAL": "Credit_Limit",
        "AGE": "Age",
        "SEX": "Gender",
        "EDUCATION": "Education_Level",
        "MARRIAGE": "Marital_Status",
    })

    df["Credit_Limit"] = pd.to_numeric(df["Credit_Limit"], errors="coerce").fillna(0)
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(30).astype(int)
    df["Defaulted"] = pd.to_numeric(df["Defaulted"], errors="coerce").fillna(0).astype(int)

    rng = np.random.default_rng(42)
    if "Annual_Income" not in df.columns:
        df["Annual_Income"] = df["Credit_Limit"] * rng.uniform(2.0, 5.0, len(df))
    if "Total_Debt" not in df.columns:
        df["Total_Debt"] = df["Total_Bill"].clip(lower=0)
    if "Current_Balance" not in df.columns:
        df["Current_Balance"] = df.get("BILL_AMT1", df["Total_Bill"]).clip(lower=0)
    if "Credit_Score" not in df.columns:
        pay_penalty = df["Max_Late_Months"].fillna(0) * 40
        util_pen = df["Debt_to_Limit"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1) * 150
        df["Credit_Score"] = np.clip(800 - pay_penalty - util_pen, 300, 850)
        df["Credit_Score"] = df["Credit_Score"].round().astype("Int64").fillna(600).astype(int)
    if "Employment_Years" not in df.columns:
        df["Employment_Years"] = np.clip(df["Age"].astype(int) - 22, 0, None)
    if "Num_Credit_Accounts" not in df.columns:
        df["Num_Credit_Accounts"] = 4
    if "Credit_History_Months" not in df.columns:
        df["Credit_History_Months"] = np.clip((df["Age"].astype(int) - 20) * 12, 12, None)
    if "Home_Ownership" not in df.columns:
        df["Home_Ownership"] = np.random.default_rng(7).choice(
            ["Rent", "Own", "Mortgage"], len(df), p=[0.4, 0.25, 0.35]
        )
    if "Loan_Type" not in df.columns:
        df["Loan_Type"] = "Credit Card"

    return df


def load_german_github() -> pd.DataFrame:
    cfg = DATASET_REGISTRY["german_github"]
    csv_path = os.path.join(DATA_DIR, cfg["filename"])
    _download(cfg["url"], csv_path, "German Credit CSV (UCI via GitHub)")

    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        "default": "Defaulted",
        "age": "Age",
        "employment_length": "Employment_Label",
        "amount": "Current_Balance",
        "savings_balance": "Savings",
        "months_loan_duration": "Loan_Duration_Months",
        "credit_history": "Credit_History_Label",
        "purpose": "Loan_Type",
        "personal_status": "Personal_Status",
        "housing": "Home_Ownership",
        "existing_credits": "Num_Credit_Accounts",
        "dependents": "Dependents",
        "telephone": "Telephone",
        "foreign_worker": "Foreign_Worker",
        "job": "Job",
        "installment_rate": "Installment_Rate",
        "residence_history": "Residence_Years",
        "other_debtors": "Other_Debtors",
        "installment_plan": "Installment_Plan",
        "property": "Property",
        "checking_balance": "Checking_Balance",
    })

    def emp_to_years(x):
        if pd.isna(x): return 3
        x = str(x).lower()
        if "unemployed" in x: return 0
        if "< 1 yr" in x or "0 - 1" in x: return 0.5
        if "1 - 4 yrs" in x or "1 - 4" in x: return 2.5
        if "4 - 7 yrs" in x or "4 - 7" in x: return 5.5
        if "> 7 yrs" in x: return 9
        return 3

    df["Employment_Years"] = df["Employment_Label"].apply(emp_to_years)
    df["Total_Debt"] = df["Current_Balance"]

    df["Annual_Income"] = 40000
    mask_low = df["Current_Balance"] < 3000
    df.loc[mask_low, "Annual_Income"] = np.random.default_rng(1).uniform(20000, 40000, mask_low.sum())
    mask_mid = (df["Current_Balance"] >= 3000) & (df["Current_Balance"] < 8000)
    df.loc[mask_mid, "Annual_Income"] = np.random.default_rng(2).uniform(35000, 65000, mask_mid.sum())
    mask_hi = df["Current_Balance"] >= 8000
    df.loc[mask_hi, "Annual_Income"] = np.random.default_rng(3).uniform(55000, 120000, mask_hi.sum())

    df["Credit_Limit"] = df["Current_Balance"] * np.random.default_rng(4).uniform(1.5, 4.0, len(df))
    df["Credit_Limit"] = df["Credit_Limit"].round(2)

    def late_from_history(x):
        if pd.isna(x): return 0
        s = str(x).lower()
        if "critical" in s: return 3
        if "delayed" in s: return 2
        if "fully repaid" in s or "all paid" in s: return 0
        if "repaid" in s: return 0
        return 1

    df["Num_Late_Payments"] = df["Credit_History_Label"].apply(late_from_history)

    df["Credit_History_Months"] = np.clip(
        (df["Age"] - 21) * 12 - df["Loan_Duration_Months"], 6, None
    )

    if "Education_Level" not in df.columns:
        df["Education_Level"] = np.random.default_rng(5).choice(
            ["High School", "Bachelor", "Master", "PhD"], len(df),
            p=[0.35, 0.4, 0.2, 0.05]
        )

    if "Credit_Score" not in df.columns:
        base = 650 + df["Employment_Years"].clip(0, 20) * 4
        base -= df["Num_Late_Payments"] * 60
        util = df["Current_Balance"] / (df["Credit_Limit"] + 1)
        base -= util.clip(0, 1) * 120
        df["Credit_Score"] = np.clip(base, 300, 850).astype(int)

    df["Defaulted"] = df["Defaulted"].replace({1: 0, 2: 1}).astype(int)

    return df


def load_dataset(name: str) -> pd.DataFrame:
    name = name.lower().strip()
    if name == "synthetic":
        print("  [OK] Using synthetic dataset (no download)")
        return generate_credit_dataset(n_samples=12000)
    if name == "uci_taiwan":
        return load_uci_taiwan()
    if name == "german_github":
        return load_german_github()
    raise ValueError(
        f"Unknown dataset '{name}'. "
        f"Choose from: {list(DATASET_REGISTRY.keys())}"
    )

sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.facecolor"] = "#0f0f0f"
plt.rcParams["figure.facecolor"] = "#050505"
plt.rcParams["text.color"] = "#e0e0e0"
plt.rcParams["axes.labelcolor"] = "#e0e0e0"
plt.rcParams["xtick.color"] = "#b0b0b0"
plt.rcParams["ytick.color"] = "#b0b0b0"
plt.rcParams["axes.edgecolor"] = "#303030"


# ============================================================
# 1. DATASET GENERATION (Synthetic - no download required)
# ============================================================
def generate_credit_dataset(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic credit dataset.

    Features mirror real credit bureau data:
      - Demographics: age, employment length
      - Financials: annual income, total debt, credit limit
      - Behaviour: # late payments, credit utilisation, length of credit history
      - Derived: debt-to-income, credit utilisation ratio, payment reliability score
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, size=n_samples)
    employment_years = np.clip(age - 18 - rng.integers(0, 8, size=n_samples), 0, 50)
    education_level = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"], size=n_samples,
        p=[0.35, 0.40, 0.20, 0.05]
    )
    home_ownership = rng.choice(
        ["Rent", "Own", "Mortgage"], size=n_samples, p=[0.40, 0.25, 0.35]
    )

    base_income = np.where(
        education_level == "PhD", 95000,
        np.where(education_level == "Master", 72000,
                 np.where(education_level == "Bachelor", 52000, 32000))
    )
    annual_income = np.clip(base_income * rng.normal(1.0, 0.25, n_samples), 12000, 300000)

    credit_score_raw = (
        300
        + 2.2 * employment_years
        + 0.0012 * annual_income
        - 20 * rng.integers(0, 2, size=n_samples) * rng.integers(0, 8, size=n_samples)
        + rng.normal(0, 45, n_samples)
    )
    credit_score = np.clip(credit_score_raw, 300, 850).astype(int)

    credit_limit = np.clip(annual_income * rng.uniform(0.15, 0.6, n_samples), 500, 150000)
    current_balance = credit_limit * rng.beta(1.8, 3.0, n_samples)
    total_debt = np.clip(
        annual_income * rng.uniform(0.05, 2.5, n_samples)
        + current_balance,
        100, None
    )

    num_late_payments = rng.poisson(lam=np.where(credit_score < 580, 3.5,
                                                  np.where(credit_score < 680, 1.2, 0.25)),
                                     size=n_samples)
    num_late_payments = np.clip(num_late_payments, 0, 15)
    credit_history_months = np.clip((age - 18) * 12 - rng.integers(0, 60, n_samples), 6, None)

    num_credit_accounts = rng.poisson(lam=4.5, size=n_samples).clip(1, 25)
    loan_type = rng.choice(["Personal", "Auto", "Mortgage", "Credit Card"],
                           size=n_samples, p=[0.25, 0.20, 0.15, 0.40])

    # ---- TARGET: defaulted (1) vs repaid (0)
    p_default = (
        0.02
        + 0.20 * np.where(num_late_payments > 3, 1, 0)
        + 0.15 * (current_balance / (credit_limit + 1)).clip(0, 1)
        + 0.10 * (total_debt / (annual_income + 1)).clip(0, 3)
        - 0.00045 * credit_score
        + 0.05 * np.where(employment_years < 2, 1, 0)
    )
    p_default = p_default.clip(0.01, 0.92)
    defaulted = rng.binomial(1, p_default)

    df = pd.DataFrame({
        "Age": age,
        "Employment_Years": employment_years,
        "Education_Level": education_level,
        "Home_Ownership": home_ownership,
        "Annual_Income": np.round(annual_income, 2),
        "Credit_Score": credit_score,
        "Credit_Limit": np.round(credit_limit, 2),
        "Current_Balance": np.round(current_balance, 2),
        "Total_Debt": np.round(total_debt, 2),
        "Num_Late_Payments": num_late_payments,
        "Credit_History_Months": credit_history_months,
        "Num_Credit_Accounts": num_credit_accounts,
        "Loan_Type": loan_type,
        "Defaulted": defaulted,
    })

    return df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Debt_to_Income"] = df["Total_Debt"] / (df["Annual_Income"] + 1)
    df["Credit_Utilisation"] = df["Current_Balance"] / (df["Credit_Limit"] + 1)
    df["Avg_Balance_Per_Account"] = df["Current_Balance"] / (df["Num_Credit_Accounts"] + 1)
    df["Income_Per_Year_of_Employment"] = df["Annual_Income"] / (df["Employment_Years"] + 1)
    df["Late_Payments_Per_Account"] = df["Num_Late_Payments"] / (df["Num_Credit_Accounts"] + 1)
    df["Late_Payments_Rate"] = df["Num_Late_Payments"] / (df["Credit_History_Months"] / 12 + 1)

    df["Has_Late_Payments"] = (df["Num_Late_Payments"] > 0).astype(int)
    df["Has_Multiple_Late"] = (df["Num_Late_Payments"] >= 3).astype(int)
    df["High_Credit_Util"] = (df["Credit_Utilisation"] > 0.7).astype(int)
    df["High_DTI"] = (df["Debt_to_Income"] > 0.5).astype(int)

    df["Credit_History_Years"] = df["Credit_History_Months"] / 12

    credit_bins = [0, 579, 669, 739, 799, 1000]
    credit_labels = ["Very_Poor", "Fair", "Good", "Very_Good", "Excellent"]
    df["Credit_Tier"] = pd.cut(df["Credit_Score"], bins=credit_bins, labels=credit_labels)

    income_bins = [0, 30000, 60000, 100000, 200000, np.inf]
    income_labels = ["Low", "Lower_Mid", "Middle", "Upper_Mid", "High"]
    df["Income_Bracket"] = pd.cut(df["Annual_Income"], bins=income_bins, labels=income_labels)

    return df


def encode_categoricals(df: pd.DataFrame, target_col: str = "Defaulted"):
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    bool_cols = X_encoded.select_dtypes(include=["bool"]).columns.tolist()
    X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)

    import re
    safe_names = []
    for name in X_encoded.columns:
        clean = re.sub(r"[\[\]<>]", "_", str(name))
        clean = re.sub(r"[^A-Za-z0-9_]", "_", clean)
        clean = re.sub(r"_+", "_", clean).strip("_")
        if not clean or clean[0].isdigit():
            clean = "f_" + clean
        safe_names.append(clean)
    X_encoded.columns = safe_names

    seen = {}
    final_names = []
    for n in safe_names:
        if n not in seen:
            seen[n] = 0
            final_names.append(n)
        else:
            seen[n] += 1
            final_names.append(f"{n}_{seen[n]}")
    X_encoded.columns = final_names

    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)
    X_encoded = X_encoded.loc[:, X_encoded.notna().any(axis=0)]
    X_encoded.columns = [str(c) for c in X_encoded.columns]

    return X_encoded, y


# ============================================================
# 3. VISUALISATIONS
# ============================================================
def plot_eda(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Credit Dataset EDA", fontsize=16, fontweight="bold", color="#ffffff")

    axes[0, 0].hist(df["Credit_Score"], bins=40, color="#4f8df7", edgecolor="#1a1a1a")
    axes[0, 0].set_title("Credit Score Distribution")
    axes[0, 0].set_xlabel("Credit Score")

    sns.boxplot(x="Defaulted", y="Debt_to_Income", data=df, ax=axes[0, 1],
                palette=["#3ecf8e", "#ff6b6b"])
    axes[0, 1].set_title("DTI vs Default")

    sns.histplot(x="Annual_Income", hue="Defaulted", data=df, ax=axes[0, 2],
                 bins=30, kde=True, palette=["#3ecf8e", "#ff6b6b"])
    axes[0, 2].set_title("Income Distribution by Status")

    sns.countplot(x="Num_Late_Payments", hue="Defaulted", data=df, ax=axes[1, 0],
                  palette=["#3ecf8e", "#ff6b6b"])
    axes[1, 0].set_title("Late Payments vs Default")
    axes[1, 0].tick_params(axis="x", rotation=45)

    axes[1, 1].scatter(df["Credit_Utilisation"], df["Debt_to_Income"],
                       c=df["Defaulted"].map({0: "#3ecf8e", 1: "#ff6b6b"}),
                       alpha=0.5, s=15)
    axes[1, 1].set_xlabel("Credit Utilisation")
    axes[1, 1].set_ylabel("Debt-to-Income")
    axes[1, 1].set_title("Utilisation vs DTI (coloured by default)")

    default_counts = df["Defaulted"].value_counts()
    axes[1, 2].pie(default_counts, labels=["Repaid", "Defaulted"],
                   colors=["#3ecf8e", "#ff6b6b"], autopct="%1.1f%%",
                   textprops={"color": "#e0e0e0"},
                   wedgeprops={"edgecolor": "#050505", "linewidth": 2})
    axes[1, 2].set_title("Class Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_plots.png"), facecolor="#050505")
    plt.close()
    print(f"  [OK] EDA plot saved: {OUTPUT_DIR}/eda_plots.png")


def plot_feature_importance(model, feature_names, title: str, filename: str, top_n: int = 15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices],
             color="#4f8df7", edgecolor="#1a1a1a")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.title(f"Top Features - {title}", fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), facecolor="#050505")
    plt.close()
    print(f"  [OK] Feature importance saved: {OUTPUT_DIR}/{filename}")


def plot_roc_curves(models_dict, X_test, y_test):
    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, color="#555555")

    colors = ["#4f8df7", "#3ecf8e", "#f59e0b", "#ff6b6b"]
    for (name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, lw=2)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curves", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), facecolor="#050505")
    plt.close()
    print(f"  [OK] ROC curves saved: {OUTPUT_DIR}/roc_curves.png")


def plot_confusion_matrices(models_dict, X_test, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (name, model) in zip(axes, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                    cbar_kws={"label": "Count"},
                    annot_kws={"color": "white", "weight": "bold"})
        ax.set_title(f"{name} Confusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Repaid", "Defaulted"])
        ax.set_yticklabels(["Repaid", "Defaulted"])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), facecolor="#050505")
    plt.close()
    print(f"  [OK] Confusion matrices saved: {OUTPUT_DIR}/confusion_matrices.png")


# ============================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================
def evaluate_model(name: str, model, X_test, y_test, y_pred=None, y_proba=None):
    if y_pred is None:
        y_pred = model.predict(X_test)
    if y_proba is None:
        y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1_Score":  f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC":   roc_auc_score(y_test, y_proba),
    }


def build_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    models = {}
    results = []

    # ─── Logistic Regression ─────────────────────────────────
    print("\n[>] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42, solver="liblinear")
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr
    results.append(evaluate_model("Logistic Regression", lr, X_test, y_test))
    plot_feature_importance(lr, feature_names, "Logistic Regression",
                            "importance_logistic_regression.png")

    # ─── Decision Tree ───────────────────────────────────────
    print("[>] Training Decision Tree...")
    dt = DecisionTreeClassifier(
        max_depth=8, min_samples_split=20, min_samples_leaf=10,
        class_weight="balanced", random_state=42
    )
    dt.fit(X_train, y_train)
    models["Decision Tree"] = dt
    results.append(evaluate_model("Decision Tree", dt, X_test, y_test))
    plot_feature_importance(dt, feature_names, "Decision Tree",
                            "importance_decision_tree.png")

    # ─── Random Forest ───────────────────────────────────────
    print("[>] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=14, min_samples_split=10,
        min_samples_leaf=5, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))
    plot_feature_importance(rf, feature_names, "Random Forest",
                            "importance_random_forest.png")

    # ─── XGBoost (bonus, top-tier) ───────────────────────────
    print("[>] Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=150, max_depth=7, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb
    results.append(evaluate_model("XGBoost", xgb, X_test, y_test))
    plot_feature_importance(xgb, feature_names, "XGBoost",
                            "importance_xgboost.png")

    results_df = pd.DataFrame(results).set_index("Model")
    return models, results_df


# ============================================================
# 5. MAIN PIPELINE
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Credit Scoring Model - predict creditworthiness (default risk)."
    )
    ap.add_argument(
        "--dataset", "-d",
        default="synthetic",
        choices=list(DATASET_REGISTRY.keys()),
        help="Which dataset to use (default: synthetic). "
             "Choices: synthetic | uci_taiwan | german_github"
    )
    return ap.parse_args()


def list_available_datasets():
    print("\nAvailable datasets:")
    print("-" * 70)
    for key, cfg in DATASET_REGISTRY.items():
        print(f"  {key:<15s}  {cfg['description']}")
    print()


def main():
    args = parse_args()
    dataset_name = args.dataset

    print("=" * 70)
    print("  CREDIT SCORING MODEL  -  Predicting Creditworthiness")
    print(f"  Dataset: {dataset_name}")
    print("=" * 70)

    if dataset_name == "synthetic":
        list_available_datasets()
        print("Tip: re-run with  --dataset uci_taiwan  or  --dataset german_github")
        print("     to use real data (auto-downloaded, no login required).\n")

    # ── Step 1: Load / Generate data ─────────────────────────
    print(f"[1/5] Loading dataset '{dataset_name}' ...")
    df_raw = load_dataset(dataset_name)
    print(f"  [OK] Dataset shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
    print(f"  [OK] Default rate:  {df_raw['Defaulted'].mean():.1%}")

    # ── Step 2: Feature Engineering ──────────────────────────
    print("\n[2/5] Performing feature engineering...")
    df = engineer_features(df_raw)
    print(f"  [OK] Features after engineering: {df.shape[1]} (including target)")

    # ── Step 3: EDA Visualisations ───────────────────────────
    print("\n[3/5] Creating EDA visualisations...")
    plot_eda(df)

    # ── Step 4: Preprocessing ────────────────────────────────
    print("\n[4/5] Encoding & splitting data...")
    X, y = encode_categoricals(df)
    feature_names = X.columns.tolist()
    print(f"  [OK] Encoded features: {len(feature_names)}")

    nan_cols = X.columns[X.isnull().any()].tolist()
    if nan_cols:
        print(f"  [..] Imputing NaN values in {len(nan_cols)} columns with median...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        X = X_imputed
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  [OK] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # For tree models, unscaled is fine; we pass scaled version explicitly
    # for LogReg below:
    X_train_lr, X_test_lr = X_train_scaled, X_test_scaled
    X_train_tree, X_test_tree = X_train, X_test

    # Rebuild versions manually for clean separation
    X_train_lr_df = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
    X_test_lr_df  = pd.DataFrame(X_test_scaled,  columns=feature_names, index=X_test.index)

    # ── Step 5: Train + Evaluate ─────────────────────────────
    print("\n[5/5] Training classification models...")

    models = {}
    results = []

    # Logistic Regression (scaled)
    print("\n[>] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=3000, C=0.5, random_state=42, solver="liblinear")
    lr.fit(X_train_lr_df, y_train)
    models["Logistic Regression"] = lr
    results.append(evaluate_model("Logistic Regression", lr, X_test_lr_df, y_test))
    plot_feature_importance(lr, feature_names, "Logistic Regression",
                            "importance_logistic_regression.png")

    # Decision Tree
    print("[>] Training Decision Tree...")
    dt = DecisionTreeClassifier(
        max_depth=10, min_samples_split=20, min_samples_leaf=10,
        class_weight="balanced", random_state=42
    )
    dt.fit(X_train_tree, y_train)
    models["Decision Tree"] = dt
    results.append(evaluate_model("Decision Tree", dt, X_test_tree, y_test))
    plot_feature_importance(dt, feature_names, "Decision Tree",
                            "importance_decision_tree.png")

    # Random Forest
    print("[>] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=250, max_depth=16, min_samples_split=10,
        min_samples_leaf=4, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train_tree, y_train)
    models["Random Forest"] = rf
    results.append(evaluate_model("Random Forest", rf, X_test_tree, y_test))
    plot_feature_importance(rf, feature_names, "Random Forest",
                            "importance_random_forest.png")

    # XGBoost
    print("[>] Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=180, max_depth=7, learning_rate=0.07,
        subsample=0.85, colsample_bytree=0.85, gamma=2,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42
    )
    xgb.fit(X_train_tree, y_train)
    models["XGBoost"] = xgb
    results.append(evaluate_model("XGBoost", xgb, X_test_tree, y_test))
    plot_feature_importance(xgb, feature_names, "XGBoost",
                            "importance_xgboost.png")

    results_df = pd.DataFrame(results).set_index("Model")

    # ── Aggregate visualisations ─────────────────────────────
    print("\n[>] Generating ROC curves...")
    # For ROC we need matching inputs - use tree-scale X_test for tree models
    # and scaled for LR; predict_proba accepts either
    roc_models = {
        "Logistic Regression": (lr, X_test_lr_df),
        "Decision Tree":       (dt, X_test_tree),
        "Random Forest":       (rf, X_test_tree),
        "XGBoost":             (xgb, X_test_tree),
    }
    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, color="#555555")
    colors = ["#4f8df7", "#3ecf8e", "#f59e0b", "#ff6b6b"]
    for (name, (model, Xt)), color in zip(roc_models.items(), colors):
        y_proba = model.predict_proba(Xt)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curves", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), facecolor="#050505")
    plt.close()
    print(f"  [OK] ROC curves saved: {OUTPUT_DIR}/roc_curves.png")

    print("[>] Generating confusion matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, ((name, (model, Xt))) in zip(axes, roc_models.items()):
        y_pred = model.predict(Xt)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                    cbar_kws={"label": "Count"},
                    annot_kws={"color": "white", "weight": "bold"})
        ax.set_title(f"{name} Confusion Matrix", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Repaid", "Defaulted"])
        ax.set_yticklabels(["Repaid", "Defaulted"])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), facecolor="#050505")
    plt.close()
    print(f"  [OK] Confusion matrices saved: {OUTPUT_DIR}/confusion_matrices.png")

    # Metric comparison bar chart
    metric_colors = {"Accuracy": "#4f8df7", "Precision": "#3ecf8e",
                     "Recall": "#f59e0b", "F1_Score": "#ec4899", "ROC_AUC": "#a855f7"}
    ax = results_df.plot(kind="bar", figsize=(12, 7),
                         color=[metric_colors[c] for c in results_df.columns],
                         edgecolor="#1a1a1a")
    plt.title("Model Performance Comparison", fontweight="bold")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.ylim(0.5, 1.0)
    plt.legend(loc="lower right")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=8, color="#e0e0e0")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), facecolor="#050505")
    plt.close()
    print(f"  [OK] Model comparison saved: {OUTPUT_DIR}/model_comparison.png")

    # ── Final report ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MODEL EVALUATION RESULTS  (higher = better for all metrics)")
    print("=" * 70)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:,.4f}".format)
    print(results_df.sort_values("ROC_AUC", ascending=False))
    print()

    # Print per-model full classification report for the best model
    best_model_name = results_df["ROC_AUC"].idxmax()
    print(f"[*] Best model by ROC-AUC: {best_model_name}")
    print("-" * 50)
    best_model = models[best_model_name]
    Xt_best = X_test_lr_df if best_model_name == "Logistic Regression" else X_test_tree
    y_pred_best = best_model.predict(Xt_best)
    print(classification_report(y_test, y_pred_best, target_names=["Repaid (0)", "Defaulted (1)"]))

    # ── Cross-validation sanity check on best model ──────────
    print("Performing 5-fold cross-validation on best model...")
    if best_model_name == "Logistic Regression":
        X_full = pd.DataFrame(StandardScaler().fit_transform(X), columns=feature_names)
    else:
        X_full = X
    cv_scores = cross_val_score(models[best_model_name], X_full, y,
                                cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  Cross-val ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Sample prediction ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SAMPLE PREDICTION  (first 5 applicants in test set)")
    print("=" * 70)
    sample_idx = X_test.head(5).index
    sample_actual = y_test.loc[sample_idx].values
    Xt_for_sample = X_test_lr_df if best_model_name == "Logistic Regression" else X_test_tree
    sample_proba = best_model.predict_proba(Xt_for_sample.loc[sample_idx])[:, 1]
    sample_pred = best_model.predict(Xt_for_sample.loc[sample_idx])

    sample_df = pd.DataFrame({
        "Actual_Status":   np.where(sample_actual == 1, "Defaulted", "Repaid"),
        "Predicted_Status": np.where(sample_pred == 1, "Defaulted", "Repaid"),
        "PD_Probability":   [f"{p:.1%}" for p in sample_proba],
        "Correct":          sample_actual == sample_pred,
    })
    print(sample_df.to_string())

    results_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.csv"))
    print(f"\n✅ All outputs saved to folder: {OUTPUT_DIR}/")
    print("   (CSVs, PNG plots, evaluation metrics)")


if __name__ == "__main__":
    main()
