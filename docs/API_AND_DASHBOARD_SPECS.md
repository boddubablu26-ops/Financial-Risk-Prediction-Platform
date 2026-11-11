# API & Dashboard Specifications

- **System:** Financial Risk Prediction Platform (FRP-ML)
- **Document Version:** 1.0.0
- **Status:** Approved / Engineering Specification
- **Frameworks:** FastAPI (Backend REST API) & Streamlit (Frontend Dashboard)

---

## 1. REST API Specifications (FastAPI)

### 1.1 General API Architecture
- **Base URL:** `http://localhost:8000/api/v1`
- **Interactive Documentation:**
  - Swagger UI: `http://localhost:8000/docs`
  - ReDoc: `http://localhost:8000/redoc`
- **Content Type:** `application/json` (except `/predict/batch` which accepts `multipart/form-data`)

---

### 1.2 Data Schemas (Pydantic Models)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ApplicantInput(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Age of applicant in years", example=32)
    person_income: float = Field(..., gt=0, description="Annual gross income in USD", example=65000.0)
    person_home_ownership: str = Field(..., description="Home ownership: RENT, MORTGAGE, OWN, OTHER", example="RENT")
    person_emp_length: float = Field(..., ge=0.0, le=60.0, description="Years of continuous employment", example=4.5)
    loan_intent: str = Field(..., description="Purpose: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION", example="DEBTCONSOLIDATION")
    loan_grade: str = Field(..., description="Credit risk grade: A, B, C, D, E, F, G", example="C")
    loan_amnt: float = Field(..., gt=0, description="Loan principal requested in USD", example=12000.0)
    loan_int_rate: float = Field(..., gt=0, le=40.0, description="Annual interest rate percentage", example=11.49)
    cb_person_default_on_file: str = Field(..., description="Prior historical default: Y or N", example="N")
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60, description="Credit history length in years", example=6)

class RiskFactor(BaseModel):
    feature: str
    impact: str  # "Increases Risk" or "Mitigates Risk"
    shap_value: float
    description: str

class PredictionResponse(BaseModel):
    status: str = "success"
    model_name: str
    model_version: str
    risk_category: str          # "Low Risk" or "High Risk"
    is_default_predicted: bool
    risk_probability: float     # 0.000 to 1.000 (e.g. 0.825)
    risk_score: int             # 0 to 1000 scale (e.g. 825)
    decision_threshold: float   # e.g. 0.35
    risk_tier: str              # "Prime", "Near-Prime", "Sub-Prime", "Deep Sub-Prime"
    key_risk_drivers: List[RiskFactor]
    recommendation: str         # "APPROVE", "MANUAL_REVIEW", "REJECT"

class ModelMetricItem(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    is_champion: bool

class LeaderboardResponse(BaseModel):
    champion_model: str
    evaluation_dataset: str
    models: List[ModelMetricItem]
```

---

### 1.3 Endpoint Definitions

#### Endpoint 1: Health & System Status
- **Method & Path:** `GET /health`
- **Description:** Verifies service health, active model load status, and memory availability.
- **Response `200 OK`:**
```json
{
  "status": "healthy",
  "active_model": "XGBoost Classifier",
  "model_version": "1.0.0",
  "pipeline_loaded": true,
  "timestamp": "2026-11-11T10:15:00Z"
}
```

#### Endpoint 2: Single Applicant Risk Prediction
- **Method & Path:** `POST /predict`
- **Request Body:** `ApplicantInput` (JSON)
- **Response `200 OK`:**
```json
{
  "status": "success",
  "model_name": "XGBoost Classifier",
  "model_version": "1.0.0",
  "risk_category": "High Risk",
  "is_default_predicted": true,
  "risk_probability": 0.824,
  "risk_score": 824,
  "decision_threshold": 0.35,
  "risk_tier": "Sub-Prime",
  "key_risk_drivers": [
    {
      "feature": "loan_percent_income",
      "impact": "Increases Risk",
      "shap_value": 0.342,
      "description": "High debt-to-income ratio (0.185)"
    },
    {
      "feature": "loan_grade",
      "impact": "Increases Risk",
      "shap_value": 0.210,
      "description": "Sub-prime loan grade: C"
    },
    {
      "feature": "person_home_ownership",
      "impact": "Mitigates Risk",
      "shap_value": -0.095,
      "description": "Residential status: MORTGAGE"
    }
  ],
  "recommendation": "REJECT"
}
```

#### Endpoint 3: Batch CSV Risk Scoring
- **Method & Path:** `POST /predict/batch`
- **Request:** `multipart/form-data` with `file: UploadFile` (CSV file containing applicant records).
- **Response `200 OK`:** Returns structured JSON summary + downloadable scored CSV URL or stream.
```json
{
  "status": "success",
  "total_records_processed": 500,
  "high_risk_count": 112,
  "low_risk_count": 388,
  "average_risk_probability": 0.238,
  "results": [
    {
      "record_id": 0,
      "risk_category": "Low Risk",
      "risk_probability": 0.124,
      "recommendation": "APPROVE"
    }
  ]
}
```

#### Endpoint 4: Detailed SHAP Explanation
- **Method & Path:** `POST /explain`
- **Request Body:** `ApplicantInput` (JSON)
- **Response `200 OK`:** Full feature attribution breakdown, baseline risk probability, and ranked importance vectors.

#### Endpoint 5: Model Leaderboard & Comparison
- **Method & Path:** `GET /models`
- **Response `200 OK`:** Multi-model benchmark leaderboard comparing all 5 trained models.

---

## 2. Interactive Analytical Dashboard (Streamlit)

The frontend is implemented as a modular multi-page Streamlit application located in `src/dashboard/`.

### 2.1 Navigation & Page Structure

```
src/dashboard/
├── app.py                      # Main entrypoint, global styling & sidebar branding
└── pages/
    ├── 1_📊_Overview.py        # Executive summary, KPI metrics, portfolio breakdown
    ├── 2_🔍_Data_Analysis.py   # Exploratory Data Analysis (EDA) & distribution plots
    ├── 3_🏆_Model_Performance.py # 5-Model Leaderboard, ROC curves, confusion matrices
    ├── 4_🎯_Risk_Prediction.py  # Interactive borrower form & Batch CSV uploader
    └── 5_💡_Explainability.py  # Global SHAP beeswarm & Local applicant waterfall
```

---

### 2.2 Detailed Page Specifications

#### Page 1: Overview Dashboard (`1_📊_Overview.py`)
- **Top Metric Cards:**
  - *Total Records Analyzed:* `32,581`
  - *Portfolio Default Rate:* `21.8%`
  - *Champion Model:* `XGBoost Classifier`
  - *Test ROC-AUC Score:* `0.942`
- **Visuals:**
  - Interactive donut chart showing Low Risk (78.2%) vs High Risk (21.8%).
  - Breakdown by Loan Intent (Debt Consolidation, Education, Medical, Venture).
  - Quick summary table of key portfolio risks.

#### Page 2: Exploratory Data Analysis (`2_🔍_Data_Analysis.py`)
- **Dataset Summary:** Interactive data table showing descriptive statistics (`describe()`).
- **Missing Value Profile:** Missingness bar chart showing `loan_int_rate` (9.6%) and `person_emp_length` (2.7%).
- **Interactive Histograms / KDEs:**
  - Distribution of `person_income` with log-scale toggle.
  - Distribution of `loan_amnt` and `loan_int_rate`.
- **Correlation Heatmap:** Interactive correlation matrix highlighting collinearity between Loan Amount, Income, and Loan Grade.
- **Outlier Box Plots:** Box plots of Loan-to-Income by Loan Status (0 vs 1).

#### Page 3: Model Performance & Comparison (`3_🏆_Model_Performance.py`)
- **Model Leaderboard Table:**
  - Columns: Model Name, Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Status (Champion).
  - Interactive sorting by any metric.
- **Visual Curves:**
  - Multi-model ROC Curves plot with individual AUC values.
  - Precision-Recall (PR) Curves plot.
- **Confusion Matrix Heatmaps:**
  - Toggle between models to display True Positives, True Negatives, False Positives, False Negatives.
- **Cost-Sensitive Threshold Optimizer:**
  - Interactive slider for decision threshold (0.10 to 0.70) showing dynamically updated financial loss curve.

#### Page 4: Risk Prediction & Scoring (`4_🎯_Risk_Prediction.py`)
- **Tab 1: Single Applicant Risk Scoring:**
  - Input form split into columns:
    - *Personal Profile:* Age (18-100), Annual Income ($), Home Ownership (Dropdown), Employment Length (Years).
    - *Credit & Loan Details:* Loan Intent (Dropdown), Loan Grade (A-G), Loan Amount ($), Interest Rate (%), Prior Default on File (Radio Y/N), Credit History Length (Years).
  - *Action Button:* `"Evaluate Financial Risk"`
  - *Output Display:*
    - Large Risk Badge: Green `LOW RISK (APPROVED)` or Red `HIGH RISK (DEFAULT HAZARD)`.
    - Risk Probability Gauge (0% to 100%).
    - Key Risk Indicators summary callout.
- **Tab 2: Batch CSV Scoring:**
  - File uploader accepting CSV.
  - Preview first 5 rows.
  - Progress bar while scoring.
  - Summary metrics: Total Scored, % Flagged High Risk.
  - Download Button: `"Download Scored CSV with Risk Probabilities"`.

#### Page 5: Explainability Studio (`5_💡_Explainability.py`)
- **Global Explainability Section:**
  - SHAP Summary Beeswarm plot: Displays top 10 features and whether higher or lower values increase risk.
  - Mean Absolute SHAP feature importance bar plot.
- **Individual Applicant Explainability Section:**
  - Interactive SHAP Waterfall Plot for the applicant evaluated on Page 4.
  - Positive contributors (red arrows pushing score higher) vs negative contributors (blue arrows pulling score lower).
  - Natural language narrative synthesis:
    - *"Key Risk Factor: Loan amount represents 38% of annual income, which significantly increases default hazard."*
    - *"Mitigating Factor: Zero prior credit bureau defaults on file provides positive credit backing."*

---

## 3. Visual & Design Standards

- **Color Palette:**
  - Primary Brand: Deep Indigo (`#4F46E5`)
  - Low Risk / Positive: Emerald Green (`#10B981`)
  - High Risk / Danger: Crimson Red (`#EF4444`)
  - Warning / Moderate: Amber (`#F59E0B`)
  - Neutral Background: Clean Slate (`#0F172A` / `#F8FAFC`)
- **Design Principles:**
  - Clean typography, high contrast, responsive grid layouts.
  - Zero jargon in user-facing recommendation badges.
  - Clear visual demarcation between training metrics and live inference.
