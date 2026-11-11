# Product Requirements Document (PRD)

## Project Title: Finance Real-World Machine Learning Project — Financial Risk Prediction
**Version:** 1.0  
**Project Type:** Machine Learning / Financial Analytics  
**Priority:** Medium–High  
**Document Status:** Approved for Engineering Implementation  
**Target Delivery Date:** Q4 2026  

---

## 1. Project Overview

The objective of this project is to develop an **end-to-end Machine Learning application** that accurately predicts financial risk based on historical financial records, credit profiles, and applicant demographics. 

The system analyzes financial records, discovers latent behavioral patterns, classifies applicants or financial entities into well-defined risk tiers (e.g., High Risk vs. Low Risk, with granular risk probability scores), and generates transparent, interpretable explanations for each decision.

By bridging complex algorithmic modeling with explainable AI (SHAP) and an intuitive web dashboard, this platform enables financial analysts, credit underwriters, and risk managers to make consistent, data-driven, and audit-compliant risk assessment decisions.

---

## 2. Business Problem & Project Objectives

### 2.1 The Business Problem
Financial institutions face substantial losses from loan defaults, credit card delinquencies, and inaccurate risk ratings. Existing manual underwriting is slow and subjective, while legacy credit scorecards often fail to capture complex non-linear interactions across emerging financial variables (e.g., disposable income ratios, recent delinquency velocity, or employment stability). Furthermore, modern regulations (e.g., Fair Credit Reporting Act, Equal Credit Opportunity Act) mandate that automated credit decisions must provide clear reasons for adverse actions.

### 2.2 Key Objectives
1. **Automated Pipeline:** Build a robust, reproducible data processing and machine learning pipeline capable of ingesting batch CSV datasets or single-record inputs.
2. **Data Cleansing & Profiling:** Identify and resolve missing values, outliers, duplicate records, and structural anomalies while providing automated Exploratory Data Analysis (EDA).
3. **Domain-Specific Feature Engineering:** Transform raw financial variables into high-signal risk metrics such as Debt-to-Income (DTI), Loan-to-Income (LTI), Financial Stress Score, and Delinquency Severity.
4. **Multi-Model Benchmarking:** Train, tune, and evaluate at least five classification algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost, and Support Vector Machine) across standardized cross-validation folds.
5. **Business-Aligned Evaluation:** Measure model performance using business-critical metrics (ROC-AUC, PR-AUC, F1-Score, Recall) and evaluate cost-benefit trade-offs (penalizing False Negatives more heavily than False Positives).
6. **Transparent Explainability:** Leverage SHAP (SHapley Additive exPlanations) to provide global feature importance across the portfolio and local waterfall explanations for individual applicant decisions.
7. **Production Serving:** Present predictions, explanations, and model analytics via a high-performance REST API (FastAPI) and an interactive web dashboard (Streamlit).

---

## 3. Target Users & Personas

| User Persona | Role & Responsibilities | Core System Needs |
| :--- | :--- | :--- |
| **Financial Analyst / Underwriter** | Evaluates loan applications, sets credit limits, assesses borrower risk profile. | Rapid single-record risk prediction, probability score, and top 3–5 plain-language risk factors influencing the decision. |
| **Risk Assessment Manager** | Oversees portfolio risk exposure, establishes underwriting policies, monitors default rates. | Portfolio-wide risk category distributions, aggregate loss metrics, model threshold tuning, and global feature importance. |
| **Data Scientist / ML Engineer** | Develops, benchmarks, tunes, and maintains predictive models and feature pipelines. | Cross-validation benchmarks, ROC-AUC comparison, confusion matrices, SHAP summary plots, and serialized artifact tracking. |
| **Compliance & Audit Officer** | Ensures lending decisions comply with regulatory requirements (fair lending, adverse action notice). | Verifiable local feature attribution (SHAP waterfall), decision audit trail, and fairness checks across demographic segments. |
| **Executive / Business Decision-Maker** | Approves lending strategy, capital allocation, and automated credit thresholds. | Executive summary KPI cards, model accuracy guarantees, and estimated default reduction figures. |

---

## 4. Core Features and Functional Requirements

### 4.1 Data Upload and Processing
* **CSV Ingestion:** Support uploading historical financial datasets via drag-and-drop in the dashboard or via the REST API endpoint `POST /api/v1/data/upload`.
* **Schema Validation:** Automatically check for required columns, correct data types (numeric, categorical, datetime), and format constraints using Pydantic schemas.
* **Duplicate Detection:** Identify and eliminate duplicate applicant records based on composite unique keys or exact row matching.
* **Missing Value Imputation:**
  * Median imputation for skewed financial variables (e.g., income, loan amount, debt).
  * Mode/constant imputation for categorical attributes (e.g., employment status, home ownership).
  * Flagging of missingness indicators when informative.
* **Outlier Auditing & Capping:** Detect extreme or erroneous financial anomalies (e.g., negative income, age > 100, DTI > 100%) and apply domain-informed Winsorization or boundary filtering.
* **Clean Dataset Persistence:** Save validated and cleaned outputs to `data/processed/clean_financial_data.parquet` and `.csv`.

### 4.2 Exploratory Data Analysis (EDA)
* **Dataset Profiling:** Display total row count, feature count, memory usage, and completeness percentage.
* **Summary Statistics:** Tabulate mean, standard deviation, median, IQR, min, and max for all numerical financial variables.
* **Distribution Visualizations:** Interactive histograms and KDE plots for critical predictors (Income, Loan Amount, Interest Rate, Credit History Length).
* **Correlation Heatmap:** Interactive Pearson and Spearman correlation matrices highlighting collinearity between predictors and risk flags.
* **Target Distribution Analysis:** Visual representation of class balance (e.g., Low Risk vs. High Risk proportion) to quantify class imbalance severity.
* **Outlier Boxplots:** Categorical boxplots illustrating financial distributions split by risk category.

### 4.3 Feature Engineering
* **Derived Financial Indicators:**
  * $\text{Debt-to-Income (DTI)} = \frac{\text{Total Monthly Debt Obligations}}{\text{Monthly Gross Income}}$
  * $\text{Loan-to-Income (LTI)} = \frac{\text{Requested Loan Amount}}{\text{Annual Gross Income}}$
  * $\text{Disposable Income} = \text{Monthly Income} - \text{Estimated Monthly Debt Payments}$
  * $\text{Delinquency Severity Index} = f(\text{Past Due 30-59 Days}, \text{Past Due 60-89 Days}, \text{Past Due 90+ Days})$
  * $\text{Credit Line Utilization} = \frac{\text{Total Revolving Balance}}{\text{Total Credit Limit}}$
  * $\text{Financial Stress Composite Score} = \text{Weighted combination of DTI, Utilization, and Delinquency History}$
* **Categorical Encoding:** One-Hot Encoding for nominal variables (e.g., loan purpose, housing status) and Target / Weight of Evidence (WoE) encoding for high-cardinality features.
* **Feature Scaling:** RobustScaler or StandardScaler applied strictly inside transformation pipelines to prevent data leakage.
* **Multicollinearity & Selection:** Filter collinear pairs ($|r| > 0.85$), evaluate Mutual Information scores, and select the optimal feature subset via Recursive Feature Elimination (RFE) or tree importance.

### 4.4 Machine Learning Model Development
The system must train, tune, and compare **5 classification algorithms**:
1. **Logistic Regression (LR):** Baseline linear classifier with L1/L2 regularization and class weight balancing.
2. **Decision Tree (DT):** Interpretable tree model with cost-complexity pruning (`ccp_alpha`) and depth restrictions.
3. **Random Forest (RF):** Ensemble of bagging trees optimizing variance reduction and non-linear interactions.
4. **XGBoost (XGB):** Gradient boosted decision trees with `scale_pos_weight`, regularization ($\gamma, \lambda$), and early stopping.
5. **Support Vector Machine (SVM):** Non-linear classification via Radial Basis Function (RBF) kernel with probability calibration via Platt scaling.

#### Training Protocol
* **Data Splitting:** Stratified 80/20 train/test split with deterministic seed (`random_state=42`).
* **Cross-Validation:** 5-Fold Stratified Cross-Validation on the training partition.
* **Imbalance Treatment:** Implement both algorithmic weighting (`class_weight='balanced'`, `scale_pos_weight`) and synthetic sampling (SMOTE/ADASYN) within CV pipelines.
* **Hyperparameter Optimization:** Automated tuning using Optuna or GridSearchCV.

### 4.5 Model Evaluation & Selection
Each model must be evaluated on the unseen test set using:
* **Accuracy**
* **Precision (Positive Predictive Value)**
* **Recall (Sensitivity / Default Detection Rate)**
* **F1-Score and Macro F1**
* **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**
* **PR-AUC (Precision-Recall Area Under Curve)**
* **Confusion Matrix (TP, FP, TN, FN)**

#### Business Decision Rule
In credit risk modeling, a **False Negative (predicting Low Risk for a borrower who subsequently defaults)** results in principal loss and is approximately $5\times$ to $10\times$ more expensive than a **False Positive (predicting High Risk for a creditworthy borrower, causing unnecessary manual review)**. Therefore, the champion model must maximize **Recall and ROC-AUC** while maintaining an acceptable precision threshold ($> 60\%$).

### 4.6 Risk Prediction Engine
* **Input Interfaces:** Accept single-record financial attributes via web form or batch records via CSV/JSON.
* **Pipeline Application:** Automatically apply the fitted preprocessing and feature engineering transformer pipeline.
* **Model Inference:** Generate predicted risk class (`High Risk` vs. `Low Risk` or tiered: `Low`, `Medium`, `High`) and continuous risk probability ($0.00\% - 100.00\%$).
* **Threshold Adjustment:** Support adjustable decision thresholds (default: 0.50, customizable to 0.35–0.40 for conservative underwriting).
* **Automated Risk Driver Extraction:** Output the top 3–5 primary positive and negative factors driving the specific prediction.

### 4.7 Model Explainability (XAI)
* **Global Explainability:**
  * SHAP Beeswarm summary plots illustrating feature importance and directional impact across the entire validation cohort.
  * Mean absolute SHAP value ranking bar chart.
* **Local Explainability:**
  * Interactive SHAP Waterfall plot for any evaluated individual applicant.
  * SHAP Force plot demonstrating push-pull dynamics of positive risk drivers vs. mitigating credit factors.
* **Linguistic Explanation Generator:** Translates mathematical SHAP log-odds contributions into plain-English analyst advisories:
  * *Example:* "Applicant assigned **High Risk (84.2% Probability)** primarily due to: (1) High Debt-to-Income ratio of 54% (+0.42 log-odds), (2) 2 instances of 90+ day past-due payments (+0.38 log-odds), and (3) Credit line utilization of 88% (+0.25 log-odds)."

---

## 5. Dashboard Requirements

The user interface will be developed in **Streamlit**, organized into 5 dedicated, high-impact sections:

1. **Overview Dashboard:**
   * High-level KPI cards: Total Applicants Analyzed, High-Risk Flag Rate, Champion Model Accuracy, Champion ROC-AUC.
   * Risk Category breakdown donut chart.
   * Model Performance Summary Leaderboard.
2. **Exploratory Data Analysis (EDA):**
   * Dynamic feature distribution plots with risk category overlay.
   * Interactive correlation matrix with threshold slider.
   * Outlier boxplots and missing value audit tables.
3. **Model Performance Studio:**
   * Side-by-side comparative table of all 5 models (Accuracy, Precision, Recall, F1, ROC-AUC).
   * Interactive ROC and Precision-Recall curve comparisons.
   * Interactive Confusion Matrix with custom classification threshold slider.
4. **Interactive Risk Predictor:**
   * Clean financial input form with grouped sections: Personal Demographics, Income & Employment, Debt Obligations, Credit History.
   * Live "Assess Risk" trigger.
   * Real-time Risk Gauge (0–100%), Status Badge (`LOW RISK` / `MEDIUM RISK` / `HIGH RISK`), and Confidence Interval.
   * Top Risk Factors callout box.
5. **Explainability & Audit Studio:**
   * Global SHAP importance rankings.
   * Individual SHAP Waterfall plot for the current prediction.
   * Interactive "What-If" Scenario Simulator (e.g., "What if the applicant pays down debt by \$5,000?").

---

## 6. Technology Stack Architecture

| Layer | Recommended Technology | Justification |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ / 3.12 | Industry standard for machine learning, data engineering, and analytics. |
| **Data Processing** | Pandas, NumPy, Scipy | High-performance tabular manipulation, vectorized arithmetic, and statistical profiling. |
| **Data Visualization** | Plotly, Seaborn, Matplotlib | Dynamic, interactive charts for dashboard and publication-quality static plots. |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM | Production-grade algorithms, pipelines, cross-validation, and metrics. |
| **Explainability** | SHAP (SHapley Additive exPlanations) | Mathematically rigorous game-theoretic feature attribution. |
| **Backend API** | FastAPI, Pydantic, Uvicorn | High-throughput asynchronous REST API with automatic OpenAPI / Swagger documentation. |
| **Frontend UI** | Streamlit | Rapid, reactive analytical dashboard with rich data widgets. |
| **Model Persistence** | Joblib, JSON metadata | Thread-safe, compressed serialization of Scikit-Learn pipelines and models. |
| **Environment** | Python `venv`, Pip | Isolated, repeatable dependency environment. |

---

## 7. Non-Functional Requirements (NFRs)

1. **Inference Latency:** Single-record API inference (including preprocessing and SHAP local attribution) must complete in under **100 milliseconds**; standard prediction without SHAP in under **25 milliseconds**.
2. **Input Reliability & Resilience:** Invalid values (e.g., string in numeric field, negative income) must trigger descriptive HTTP 422 validation errors without crashing the service.
3. **Model Reproducibility:** Fixed seeds (`random_state=42`) across data splits, pipeline transforms, and algorithm training must guarantee 100% deterministic model replication.
4. **Security & Data Privacy:** No Personally Identifiable Information (PII) such as full names, Social Security Numbers, or raw tax IDs stored unencrypted. Synthetic or anonymized identifiers used exclusively.
5. **Auditability & Versioning:** Every serialized model file must be paired with an immutable `metadata.json` capturing training timestamp, dataset hash, Git commit SHA, hyperparameter config, and benchmark metric scores.
6. **Code Quality & Testing:** Minimum 80% unit test coverage across data cleaning, feature engineering, and inference modules using `pytest`.

---

## 8. Expected Deliverables

The development team is expected to produce:
1. `src/data/`: Data ingestion, Pydantic validation, cleaning, and preprocessing pipeline.
2. `src/features/`: Feature engineering transformers and selection pipeline.
3. `src/models/`: Multi-model training, hyperparameter optimization, and evaluation suite.
4. `src/explainability/`: Global and local SHAP explanation module and natural language generator.
5. `models/`: Best-performing serialized model pipeline (`champion_model.joblib`) and `model_metadata.json`.
6. `api/`: FastAPI REST API service with OpenAPI documentation.
7. `dashboard/`: Multi-page interactive Streamlit dashboard.
8. `docs/`: Complete engineering documentation suite (specifications, data dictionary, tracker, architecture).
9. `tests/`: Automated unit and integration test suite.
10. `run_pipeline.py`: Single-command master orchestration script.

---

## 9. Acceptance Criteria

The system will achieve final production sign-off when:
- [ ] Historical financial dataset is ingested, validated, cleaned, and summarized with zero unhandled exceptions.
- [ ] All 5 classification models (LR, DT, RF, XGB, SVM) are trained, cross-validated, and benchmarked on identical test sets.
- [ ] Champion model achieves **ROC-AUC $\ge 0.85$** and **Recall $\ge 0.75$** on the test partition.
- [ ] FastAPI endpoint `/api/v1/predict` accepts financial applicant payloads and returns predicted risk category, probability, and key factors within 100ms.
- [ ] SHAP explainability module generates valid global feature rankings and individual waterfall plots.
- [ ] Streamlit dashboard renders all 5 views cleanly, error-free, and responsively.
- [ ] Automated test suite passes 100% of test cases.
- [ ] Project documentation and setup instructions are verified by an independent developer onboarding run.
