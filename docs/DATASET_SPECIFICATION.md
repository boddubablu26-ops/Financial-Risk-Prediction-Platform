# Dataset Specification & Data Management Guide

- **Project:** Financial Risk Prediction Platform (FRP-ML)
- **Document Version:** 1.0.0
- **Dataset Selected:** Modern Credit Risk Benchmark Dataset (`credit_risk_dataset.csv`)
- **Benchmark Source:** Industry-Standard Credit Risk Lending Benchmark (32,581 records)

---

## 1. Dataset Selection & Justification

To build a real-world, enterprise-grade financial risk model, selecting the right dataset is critical. Historically, educational ML projects used the **Statlog German Credit Dataset (1994)**. However, modern financial risk modeling requires a significantly more robust, granular, and realistic dataset.

### Comparative Evaluation

| Evaluation Criteria | Legacy German Credit Dataset (1994) | Modern Credit Risk Benchmark (Selected) | Why the Modern Dataset is Superior |
| :--- | :--- | :--- | :--- |
| **Sample Size** | 1,000 records | **32,581 records** | 32x more data; enables gradient boosting and cross-validation without high variance. |
| **Feature Realism** | Coded abstract categories (`A11`, `A12`, etc.) | **Explicit financial metrics** (Income, Loan Amount, Interest Rate, Credit History) | Directly interpretable; aligns with modern commercial lending and fintech underwriting. |
| **Data Quality Challenges** | Pre-cleaned, artificial | **Realistic missing values & outliers** (e.g. missing interest rates, employment lengths) | Forces engineering team to implement production-grade imputation and outlier routines. |
| **Class Imbalance** | 70% Non-Default / 30% Default | **78.2% Low Risk (0) / 21.8% High Risk (1)** | Reflects real-world banking default rates; stresses ROC-AUC, PR-AUC, and cost-weighted loss. |
| **Explainability (XAI)** | Obscure feature names | **Domain features (Debt-to-Income, Default on File)** | Yields intuitive, audit-compliant SHAP force and waterfall explanations for loan officers. |

---

## 2. Complete Data Dictionary

The dataset comprises **12 core variables** (11 features and 1 target label).

| Column Name | Data Type | Role | Value Domain / Example Range | Missingness | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `person_age` | Integer | Feature | 20 – 79 years (outliers > 80) | 0.0% | Chronological age of applicant. |
| `person_income` | Float / Int | Feature | $4,000 – $2,000,000+ | 0.0% | Annual gross income of the applicant in USD. |
| `person_home_ownership` | String (Cat) | Feature | `RENT`, `MORTGAGE`, `OWN`, `OTHER` | 0.0% | Residential ownership status. |
| `person_emp_length` | Float | Feature | 0.0 – 45.0 years | **~2.7% (895 rows)** | Duration in years with current or continuous employer. |
| `loan_intent` | String (Cat) | Feature | `PERSONAL`, `EDUCATION`, `MEDICAL`, `VENTURE`, `HOMEIMPROVEMENT`, `DEBTCONSOLIDATION` | 0.0% | Stated primary purpose of the credit facility. |
| `loan_grade` | String (Ord) | Feature | `A`, `B`, `C`, `D`, `E`, `F`, `G` | 0.0% | Credit bureau assigned risk rating (A = Lowest risk, G = Highest risk). |
| `loan_amnt` | Float / Int | Feature | $500 – $35,000 | 0.0% | Principal loan amount requested in USD. |
| `loan_int_rate` | Float | Feature | 5.42% – 23.22% | **~9.6% (3,116 rows)**| Contractual interest rate percentage. |
| `loan_percent_income`| Float | Feature | 0.00 – 0.83 (Ratio) | 0.0% | Debt burden ratio: `loan_amnt / person_income`. |
| `cb_person_default_on_file` | String (Binary) | Feature | `Y`, `N` | 0.0% | Historical record of previous default on credit file. |
| `cb_person_cred_hist_length`| Integer | Feature | 2 – 30 years | 0.0% | Length of active credit history registered in credit bureau. |
| **`loan_status`** | **Integer** | **Target** | **`0` (Non-Default), `1` (Default)** | **0.0%** | **Target variable. 0: Low Risk / Approved; 1: High Risk / Defaulted.** |

---

## 3. Data Cleaning & Validation Rules

### 3.1 Schema & Boundary Validation Rules

```python
# Pydantic / Validation Boundary Rules
VALIDATION_RULES = {
    "person_age": {"min": 18, "max": 100, "action": "Filter outliers where age > 100"},
    "person_income": {"min": 1000, "max": 10_000_000, "action": "Must be positive"},
    "person_home_ownership": {"allowed": ["RENT", "MORTGAGE", "OWN", "OTHER"]},
    "person_emp_length": {"min": 0.0, "max": 60.0, "action": "Filter emp_length > age - 14"},
    "loan_intent": {"allowed": [
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", 
        "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
    ]},
    "loan_grade": {"allowed": ["A", "B", "C", "D", "E", "F", "G"]},
    "loan_amnt": {"min": 500, "max": 100_000},
    "loan_int_rate": {"min": 3.0, "max": 40.0},
    "loan_status": {"allowed": [0, 1]},
    "cb_person_default_on_file": {"allowed": ["Y", "N"]},
    "cb_person_cred_hist_length": {"min": 0, "max": 60}
}
```

### 3.2 Missing Value Imputation Strategy
1. **`person_emp_length` (~2.7% missing):**
   - Impute missing values using the **median employment length grouped by applicant age bracket** (or global training median: typically 4.0 years).
   - Zero data leakage: Median statistics must be computed strictly on the training partition.
2. **`loan_int_rate` (~9.6% missing):**
   - Interest rate correlates strongly with `loan_grade` ($r \approx 0.75$).
   - Impute missing `loan_int_rate` using the **median interest rate for the applicant's corresponding `loan_grade`** calculated from the training split.

### 3.3 Outlier Remediation Strategy
- **Age Anomalies:** Any record with `person_age > 100` is an operational data entry error and must be removed or clamped to 80.
- **Employment Length Anomalies:** If `person_emp_length > (person_age - 14)`, drop or clamp to realistic career duration.
- **Extreme Income Ratios:** Verify `loan_percent_income <= 1.0` (loan amount should not exceed 100% of annual income for standard retail loans).

---

## 4. Derived Feature Engineering Rules

The feature pipeline must generate the following derived financial features:

1. **Calculated Loan-to-Income (LTI):**
   $$\text{LTI} = \frac{\text{loan\_amnt}}{\text{person\_income}}$$
2. **Annual Debt Service / Interest Burden Index:**
   $$\text{Interest\_Burden} = \frac{\text{loan\_amnt} \times (\text{loan\_int_rate} / 100)}{\text{person\_income}}$$
3. **Credit History Maturity Ratio:**
   $$\text{Cred\_to\_Age\_Ratio} = \frac{\text{cb\_person\_cred\_hist\_length}}{\text{person\_age}}$$
4. **Historical Risk Multiplier:**
   - Binary encoding: `cb_person_default_on_file == 'Y'` converted to `1`, `'N'` to `0`.
   - Interaction feature: `Historical_Risk_Weight = Default_Flag * (loan_grade_numeric)`.

---

## 5. Dataset Storage & Directory Structure

```
data/
├── raw/
│   └── credit_risk_dataset.csv         # Original untouched dataset (32,581 rows)
├── processed/
│   ├── train_features.csv             # Scaled and encoded training features
│   ├── test_features.csv              # Scaled and encoded test features
│   ├── train_target.csv               # Training labels (y_train)
│   └── test_target.csv                # Test labels (y_test)
└── sample/
    └── sample_applicant.json          # Single sample payload for API verification
```

---

## 6. Dataset Acquisition & Fallback Generation Script

To enable the dev team to immediately work without external download blocks or missing files, an automated acquisition script `scripts/download_or_generate_dataset.py` will be provided. The script:
1. First attempts to locate or download the official 32,581-record benchmark CSV.
2. If offline or unavailable, automatically generates a statistically identical, high-fidelity financial benchmark dataset preserving all feature correlations, distributions, missing value rates, and the 21.8% default rate.
