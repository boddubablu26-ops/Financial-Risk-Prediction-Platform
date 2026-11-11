# Master Project Implementation Tracker

## Project: Financial Risk Prediction Platform (`Project 3`)
**Version:** 1.0  
**Current Date:** 2026-11-11  
**Project Lead:** Machine Learning Engineering Lead  
**Repository Path:** `A:\chanikya_projects\project 3`  

---

## 1. Executive Summary & Purpose

This tracker provides an end-to-end execution audit and verification system for the **Financial Risk Prediction Platform**. It ensures that every planned engineering, data science, modeling, API, dashboard, and testing deliverable defined in the PRD is tracked with utmost precision from inception to final sign-off.

---

## 2. Milestone Phase Summary

| Phase | Milestone Description | Target Deliverables | Status | Completion % |
| :---: | :--- | :--- | :---: | :---: |
| **0** | **Requirements & Architectural Documentation** | Full documentation suite in `docs/` | **COMPLETED** | **100%** |
| **1** | **Environment Setup & Dataset Acquisition** | Python `venv`, locked `requirements.txt`, raw benchmark CSV | **COMPLETED** | **100%** |
| **2** | **Data Ingestion, Validation & Preprocessing** | Pydantic validator, cleaning pipeline, train/test split | **COMPLETED** | **100%** |
| **3** | **Exploratory Data Analysis (EDA) Module** | Profiling script, summary distributions, correlation matrix | **COMPLETED** | **100%** |
| **4** | **Financial Feature Engineering & Selection** | Ratios (DTI, LTI), WoE/OneHot, Scaler, RFE feature selector | **COMPLETED** | **100%** |
| **5** | **Multi-Model Machine Learning Development** | 5 algorithms (LR, DT, RF, XGB, SVM) trained & tuned | **COMPLETED** | **100%** |
| **6** | **Model Evaluation, Comparison & Cost Analysis** | Benchmark metrics table, ROC/PR curves, champion selection | **COMPLETED** | **100%** |
| **7** | **Model Explainability Engine (SHAP)** | TreeSHAP/KernelSHAP, waterfall plots, linguistic generator | **COMPLETED** | **100%** |
| **8** | **Inference API (FastAPI) & Interactive Dashboard (Streamlit)** | REST API microservice, 5-tab web dashboard, user guides | **COMPLETED** | **100%** |
| **9** | **Testing, Verification & Quality Sign-Off** | Pytest suite, latency validation, end-to-end CLI runner | **COMPLETED** | **100%** |

*Status Legend:*
- **COMPLETED:** Built, rigorously tested, and passing all acceptance gates.
- **IN PROGRESS:** Actively under active development.
- **PENDING:** Scheduled for development in sequential order.
- **BLOCKED:** Blocked pending an external dependency or decision.

---

## 3. Itemized Task Checklist & Verification Gates

### Phase 0: Requirements & Architectural Documentation
*Objective: Establish exhaustive, unambiguous system specifications and engineering guidance.*

- [x] **0.1 Master Documentation Suite Index:** Create `docs/README.md` defining documentation architecture and navigation.
- [x] **0.2 Product Requirements Document:** Finalize `docs/PRD.md` with business problem, personas, functional requirements 4.1–4.7, and NFRs.
- [x] **0.3 Implementation Tracker:** Build `docs/PROJECT_TRACKER.md` with milestone gates, itemized verification criteria, and RACI matrix.
- [x] **0.4 Dataset & Data Dictionary:** Author `docs/DATASET_AND_DATA_DICTIONARY.md` analyzing candidate datasets, defining modern 2024/2025 financial risk schema, and field constraints.
- [x] **0.5 Data Pipeline & Cleaning Specification:** Author `docs/DATA_PIPELINE_AND_PREPROCESSING_SPEC.md` specifying Pydantic validation, outlier capping, missing value rules, and split protocols.
- [x] **0.6 Feature Engineering Specification:** Author `docs/FEATURE_ENGINEERING_SPEC.md` detailing DTI, LTI, Delinquency Index, encodings, and feature selection.
- [x] **0.7 Machine Learning Modeling Specification:** Author `docs/ML_MODELING_AND_EVALUATION_SPEC.md` covering all 5 classifiers, CV strategy, hyperparameter tuning, and cost-matrix evaluation.
- [x] **0.8 Model Explainability (SHAP) Specification:** Author `docs/MODEL_EXPLAINABILITY_SHAP_SPEC.md` specifying global/local SHAP analysis and natural language risk driver generation.
- [x] **0.9 Backend API Specification:** Author `docs/API_BACKEND_SPEC.md` specifying FastAPI endpoints, request/response JSON schemas, and error responses.
- [x] **0.10 Frontend Dashboard Specification:** Author `docs/DASHBOARD_FRONTEND_SPEC.md` specifying Streamlit 5-tab user interface, layouts, and component states.
- [x] **0.11 System Architecture & Deployment Guide:** Author `docs/SYSTEM_ARCHITECTURE_AND_DEPLOYMENT_GUIDE.md` detailing directory layout, virtual environment, and CI/CD testing.

---

### Phase 1: Environment Setup & Dataset Acquisition
*Objective: Prepare isolated Python runtime and obtain high-quality financial risk dataset.*

- [x] **1.1 Virtual Environment Initialization:** Create Python 3.12 virtual environment in `.venv/` or `venv/`.  
  *Verification:* `python -m venv venv && venv\Scripts\python --version` returns Python 3.12.x.
- [x] **1.2 Dependency Management:** Create locked `requirements.txt` containing Pandas, NumPy, Scikit-learn, XGBoost, SHAP, FastAPI, Uvicorn, Streamlit, Plotly, Pydantic, and Pytest.  
  *Verification:* `pip install -r requirements.txt` succeeds with zero unresolved conflicts.
- [x] **1.3 Project Directory Scaffolding:** Create `data/raw/`, `data/processed/`, `models/`, `src/`, `api/`, `dashboard/`, `tests/`.
- [x] **1.4 Dataset Acquisition Script:** Develop `src/data/download_dataset.py` to acquire modern credit risk dataset (e.g. 32k–50k rows) with realistic financial indicators, including a high-fidelity synthetic benchmark fallback generator for deterministic offline runs.  
  *Verification:* `python src/data/download_dataset.py` writes `data/raw/financial_risk_records.csv` (>30,000 rows).

---

### Phase 2: Data Ingestion, Validation & Preprocessing Pipeline
*Objective: Build production-grade data cleansing and validation module.*

- [x] **2.1 Pydantic Ingestion Validator:** Implement `src/data/validator.py` defining schema constraints for each financial attribute (income > 0, age between 18 and 100, valid loan intent categories).  
  *Verification:* Unit tests verify invalid rows are caught and categorized.
- [x] **2.2 Data Cleansing Module:** Implement `src/data/cleaner.py` handling duplicate removal, median imputation for numerical features, mode for categorical features, and domain-informed outlier capping.  
  *Verification:* Cleaned dataset contains 0 unexpected nulls and 0 duplicate IDs.
- [x] **2.3 Stratified Splitting Utility:** Implement `src/data/splitter.py` performing 80/20 train/test split stratified on `risk_flag` with `random_state=42`.  
  *Verification:* Class balance ratio in train set matches test set to within $\pm 0.5\%$.
- [x] **2.4 Clean Data Persistence:** Ensure cleaned datasets are persisted to `data/processed/train.parquet` and `data/processed/test.parquet`.

---

### Phase 3: Exploratory Data Analysis (EDA) Module
*Objective: Provide automated statistical profiling and data health visualizations.*

- [x] **3.1 Statistical Profiler:** Implement `src/data/eda.py` computing summary statistics, missingness matrices, and skewness coefficients.
- [x] **3.2 Distribution Visualizer:** Generate interactive histograms for Income, Debt, Credit Score, and Loan Amount segmented by Risk Category.
- [x] **3.3 Correlation & Multicollinearity Matrix:** Compute Pearson and Spearman correlation matrices; output high-collinearity pairs ($|r| > 0.80$).
- [x] **3.4 Target Imbalance Audit:** Generate visualization quantifying proportion of High Risk vs. Low Risk cases.

---

### Phase 4: Financial Feature Engineering & Selection Pipeline
*Objective: Generate derived financial risk indicators and build leak-free transformation pipelines.*

- [x] **4.1 Derived Financial Ratios Transformer:** Implement `src/features/transformers.py` calculating:
  - Debt-to-Income (DTI)
  - Loan-to-Income (LTI)
  - Estimated Disposable Monthly Income
  - Delinquency Severity Index
  - Credit Line Utilization Ratio
  - Financial Health Composite Score
- [x] **4.2 Scikit-Learn Preprocessing Pipeline:** Assemble `ColumnTransformer` applying `OneHotEncoder(handle_unknown='ignore')` to categoricals and `RobustScaler()` to numeric features.  
  *Verification:* Pipeline correctly transforms test set without re-fitting.
- [x] **4.3 Feature Selection Engine:** Implement `src/features/selector.py` executing Mutual Information and Tree-based importance ranking to isolate the top $k$ predictive features.

---

### Phase 5: Multi-Model Machine Learning Development
*Objective: Train, tune, and cross-validate 5 distinct classification models.*

- [x] **5.1 Logistic Regression Model:** Implement L1/L2 regularized baseline with `class_weight='balanced'`.
- [x] **5.2 Decision Tree Model:** Implement tree classifier with cost-complexity pruning (`ccp_alpha`) to prevent overfitting.
- [x] **5.3 Random Forest Classifier:** Implement ensemble bagging model with `n_estimators=200`, `max_depth=12`.
- [x] **5.4 XGBoost Classifier:** Implement gradient boosted trees with `scale_pos_weight`, `learning_rate=0.05`, and early stopping.
- [x] **5.5 Support Vector Machine (SVM):** Implement SVM with RBF kernel and `CalibratedClassifierCV` for true risk probability outputs.
- [x] **5.6 5-Fold Stratified Cross-Validation:** Train all 5 models inside cross-validation loops ensuring zero data leakage across folds.
- [x] **5.7 Hyperparameter Optimization:** Implement Optuna or GridSearch tuning for top-performing candidates.

---

### Phase 6: Model Evaluation, Comparison & Cost Analysis
*Objective: Rigorously benchmark model performance under financial decision constraints.*

- [x] **6.1 Metric Computation Suite:** Compute Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC for all 5 models on the held-out test split.
- [x] **6.2 Comparative Benchmark Table:** Generate unified performance comparison table.
- [x] **6.3 Confusion Matrix & Cost Analysis:** Calculate expected financial loss matrix (Cost of False Negative = \$10,000 default loss; Cost of False Positive = \$500 review cost).
- [x] **6.4 Decision Threshold Tuning:** Optimize probability threshold $\tau$ (e.g., 0.35 vs. 0.50) to maximize F1 or minimize total financial loss.
- [x] **6.5 Champion Model Selection & Serialization:** Select highest-performing model (target: ROC-AUC $\ge 0.85$, Recall $\ge 0.75$), serialize pipeline as `models/champion_model.joblib`, and generate `models/model_metadata.json`.

---

### Phase 7: Model Explainability Engine (SHAP)
*Objective: Build global and local interpretability modules for regulatory compliance.*

- [x] **7.1 Global SHAP Explainer:** Implement `src/explainability/shap_engine.py` using `shap.TreeExplainer` or `KernelExplainer` to compute shap values across test cohort.
- [x] **7.2 Summary & Beeswarm Visuals:** Generate global feature importance bar plots and directional beeswarm plots.
- [x] **7.3 Local Waterfall Generator:** Compute exact per-feature contributions for any single applicant record.
- [x] **7.4 Natural Language Risk Driver Generator:** Implement linguistic rule engine converting top positive/negative SHAP values into user-friendly bullet points.  
  *Verification:* Example output matches: "High debt-to-income ratio (48%), low credit history length (2 yrs), and 2 recent delinquent payments."

---

### Phase 8: Inference API (FastAPI) & Interactive Dashboard (Streamlit)
*Objective: Deliver enterprise-ready prediction interfaces for analysts and automated systems.*

- [x] **8.1 FastAPI Microservice (`api/main.py`):**
  - `GET /health` — health check and model version.
  - `POST /api/v1/predict` — single-record prediction with probability and key risk drivers.
  - `POST /api/v1/predict/batch` — bulk CSV/JSON batch risk scoring.
  - `GET /api/v1/explain/{id}` — local SHAP breakdown.
  - `GET /api/v1/model/metrics` — champion model benchmarks.
- [x] **8.2 Streamlit Web Application (`dashboard/app.py`):**
  - **Tab 1: Executive Overview:** High-level metrics, portfolio risk distribution, model leaderboard.
  - **Tab 2: Exploratory Data Analysis:** Interactive feature distributions, correlation heatmap, outlier audits.
  - **Tab 3: Model Performance Studio:** Side-by-side comparison table, interactive ROC/PR curves, confusion matrix with threshold slider.
  - **Tab 4: Interactive Risk Predictor:** Applicant input form, live risk gauge, risk tier status badge, top risk factors.
  - **Tab 5: Explainability & Audit:** Global SHAP importance, local waterfall chart, and interactive "What-If" simulator.

---

### Phase 9: Testing, Verification & Master CLI Runner
*Objective: Ensure complete test coverage and single-command execution.*

- [x] **9.1 Automated Test Suite (`tests/`):**
  - `test_data_cleaning.py`: Validates missing value handling and outlier bounds.
  - `test_feature_engineering.py`: Validates calculation of DTI, LTI, and scaling.
  - `test_model_inference.py`: Validates model prediction output shapes and probabilities ($0.0 \le p \le 1.0$).
  - `test_api_endpoints.py`: Tests FastAPI `/health` and `/predict` endpoints using `TestClient`.
- [x] **9.2 Master CLI Orchestration Script (`run_pipeline.py`):**
  - Orchestrates end-to-end flow: Download Data $\rightarrow$ Clean & Validate $\rightarrow$ Engineer Features $\rightarrow$ Train 5 Models $\rightarrow$ Evaluate & Save Champion $\rightarrow$ Generate SHAP Visuals.
- [x] **9.3 Independent Developer Onboarding Walkthrough:** Verify that a developer can clone the repo, install dependencies, and run `run_pipeline.py` without errors.

---

## 4. RACI Matrix (Roles & Responsibilities)

| Milestone / Deliverable | Data Scientist (DS) | ML Engineer (MLE) | Backend Eng (BE) | Frontend Eng (FE) | Risk Lead (RL) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Dataset Selection & Ingestion | **A / R** | C | C | I | C |
| Data Validation & Cleaning Pipeline | C | **A / R** | C | I | I |
| Financial Feature Engineering | **A / R** | R | I | I | C |
| 5-Model Training & Benchmarking | **A / R** | R | I | I | C |
| Cost-Sensitive Threshold Tuning | R | C | I | I | **A** |
| SHAP Explainability & NLP Drivers | **A / R** | R | C | I | C |
| FastAPI Microservice Development | I | C | **A / R** | C | I |
| Streamlit Interactive Dashboard | C | C | C | **A / R** | C |
| Pytest Test Suite & CI/CD Pipeline | C | **A / R** | R | R | I |

*Legend: R = Responsible, A = Accountable, C = Consulted, I = Informed.*

---

## 5. Risk Register & Mitigation Strategy

| Risk ID | Identified Risk | Severity | Probability | Mitigation Strategy |
| :---: | :--- | :---: | :---: | :--- |
| **R-01** | Extreme class imbalance in raw data (e.g. <10% default rate) causing model bias. | High | High | Implement Stratified K-Fold CV, `scale_pos_weight` in XGBoost, `class_weight='balanced'` in RF/LR, and evaluate using PR-AUC and Recall. |
| **R-02** | Data leakage between train and test splits during feature scaling or imputation. | Critical | Med | Enforce Scikit-Learn `Pipeline` and `ColumnTransformer` architecture; strictly call `.fit()` only on training folds. |
| **R-03** | SHAP `KernelExplainer` computational latency exceeding real-time API SLA (<100ms). | High | High | Use `TreeExplainer` for tree models (XGBoost/RF) which operates in $<10\text{ms}$; cache background reference sets. |
| **R-04** | Disparate impact / bias across demographic groups (e.g., age, gender). | High | Low | Conduct algorithmic fairness audits; exclude protected attributes from model inputs; verify equalized odds across demographic subgroups. |
| **R-05** | Production drift in macroeconomic conditions (e.g., interest rate surges). | Med | Med | Log all inference inputs and prediction probabilities; establish drift detection monitors (Kolmogorov-Smirnov test on DTI/Income). |
