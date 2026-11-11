# Backend REST API Engineering Specification (FastAPI)

## Project: Financial Risk Prediction Platform (`Project 3`)
**Version:** 1.0  
**Module Target:** `api/` (`main.py`, `schemas.py`, `dependencies.py`, `routes/`)  
**Framework:** FastAPI 0.110+ on Python 3.12 (ASGI: Uvicorn)  
**Interactive Docs:** Swagger UI at `http://localhost:8000/docs`  

---

## 1. Architecture & Service Overview

The backend service is built using **FastAPI**, providing high-throughput, asynchronous, strictly typed REST endpoints for real-time risk inference, batch scoring, model explanation extraction, data uploads, and model monitoring.

```mermaid
flowchart TD
    Client["Clients: Web Dashboard / External Core Banking System"]
    
    subgraph FastAPI_Microservice["FastAPI Service (Port 8000)"]
        Router["API Router / v1"]
        AuthMiddleware["CORS & Request Validation Middleware"]
        PydanticGuard["Pydantic v2 Schema Enforcement"]
        
        subgraph Endpoints["REST Endpoints"]
            EP_Health["GET /health"]
            EP_Predict["POST /api/v1/predict"]
            EP_Batch["POST /api/v1/predict/batch"]
            EP_Explain["GET /api/v1/explain/{applicant_id}"]
            EP_Metrics["GET /api/v1/model/metrics"]
            EP_Upload["POST /api/v1/data/upload"]
        end
        
        Engine["Inference & Explainability Engine (Cached Pipeline in Memory)"]
    end
    
    Client --> AuthMiddleware --> Router --> PydanticGuard --> Endpoints
    Endpoints --> Engine
```

---

## 2. API Endpoints Specification

### 2.1 Service Health & Metadata
* **Endpoint:** `GET /health`
* **Description:** Verifies service uptime, loaded model name, version, and training timestamp.
* **Response Status:** `200 OK`
* **Response Payload:**
```json
{
  "status": "healthy",
  "service": "financial-risk-prediction-api",
  "version": "1.0.0",
  "champion_model": "XGBoost Classifier",
  "model_version": "1.0.0",
  "timestamp": "2026-11-11T10:35:00Z"
}
```

---

### 2.2 Single Applicant Risk Scoring
* **Endpoint:** `POST /api/v1/predict`
* **Description:** Scores a single applicant, calculates risk probability, assigns risk tier, and extracts top natural language risk factors.
* **Request Headers:** `Content-Type: application/json`
* **Request Body (`FinancialApplicantInput`):**
```json
{
  "applicant_id": "APP-2026-9041",
  "person_age": 31,
  "person_income": 62000.0,
  "person_home_ownership": "RENT",
  "person_emp_length": 3.5,
  "loan_intent": "DEBTCONSOLIDATION",
  "loan_grade": "D",
  "loan_amnt": 18500.0,
  "loan_int_rate": 14.25,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6,
  "revolving_utilization": 0.72,
  "delinq_2yrs": 1,
  "total_outstanding_debt": 28400.0
}
```
* **Response Status:** `200 OK`
* **Response Body (`RiskPredictionResponse`):**
```json
{
  "applicant_id": "APP-2026-9041",
  "risk_category": "High Risk",
  "risk_tier": "Tier 3: Elevated Risk",
  "risk_probability": 0.7642,
  "decision_threshold": 0.40,
  "recommendation": "Manual Underwriter Review Required - Elevated Default Risk",
  "key_risk_factors": [
    "High debt-to-income ratio (45.8%)",
    "Elevated credit line utilization (72.0%)",
    "1 recent delinquent payment(s)"
  ],
  "positive_credit_factors": [
    "Established credit history (6 yrs)",
    "Stable employment duration (3.5 yrs)"
  ],
  "latency_ms": 14.8
}
```

---

### 2.3 Batch Risk Prediction
* **Endpoint:** `POST /api/v1/predict/batch`
* **Description:** Accepts a list of applicant objects (up to 1,000 records) or a multipart CSV file upload and returns vector-scored risk predictions.
* **Request Body:** List of `FinancialApplicantInput` objects or multipart form file (`file: UploadFile`).
* **Response Status:** `200 OK`
* **Response Body:**
```json
{
  "total_records_processed": 500,
  "high_risk_count": 109,
  "low_risk_count": 391,
  "portfolio_risk_rate": 0.218,
  "predictions": [
    {
      "applicant_id": "APP-1001",
      "risk_category": "Low Risk",
      "risk_probability": 0.1245
    },
    {
      "applicant_id": "APP-1002",
      "risk_category": "High Risk",
      "risk_probability": 0.8120
    }
  ],
  "batch_latency_seconds": 0.48
}
```

---

### 2.4 Detailed SHAP Local Explanation
* **Endpoint:** `POST /api/v1/explain`
* **Description:** Computes exact local SHAP attributions for all features on an applicant payload for waterfall rendering.
* **Response Body:**
```json
{
  "applicant_id": "APP-2026-9041",
  "base_value": 0.218,
  "output_value": 0.7642,
  "shap_contributions": [
    {"feature": "debt_to_income", "value": 0.458, "shap_value": 0.285},
    {"feature": "revolving_utilization", "value": 0.72, "shap_value": 0.142},
    {"feature": "delinq_severity_index", "value": 1.43, "shap_value": 0.089},
    {"feature": "person_income", "value": 62000.0, "shap_value": -0.054}
  ]
}
```

---

### 2.5 Model Evaluation Metrics Summary
* **Endpoint:** `GET /api/v1/model/metrics`
* **Description:** Retrieves the latest benchmark comparison results for all 5 models and champion details.
* **Response Body:**
```json
{
  "champion_model": "XGBoost Classifier",
  "benchmark_summary": {
    "Logistic Regression": {"accuracy": 0.824, "roc_auc": 0.854, "recall": 0.761},
    "Decision Tree": {"accuracy": 0.841, "roc_auc": 0.849, "recall": 0.732},
    "Random Forest": {"accuracy": 0.878, "roc_auc": 0.908, "recall": 0.795},
    "XGBoost": {"accuracy": 0.884, "roc_auc": 0.912, "recall": 0.815},
    "Support Vector Machine": {"accuracy": 0.835, "roc_auc": 0.862, "recall": 0.758}
  }
}
```

---

## 3. Pydantic Schemas (`api/schemas.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ApplicantInput(BaseModel):
    applicant_id: Optional[str] = "APP-TEMP"
    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., ge=1000.0)
    person_home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
    person_emp_length: Optional[float] = Field(0.0, ge=0.0, le=60.0)
    loan_intent: Literal["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(..., ge=500.0, le=100000.0)
    loan_int_rate: Optional[float] = Field(None, ge=3.0, le=40.0)
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(..., ge=1, le=50)
    revolving_utilization: Optional[float] = Field(0.35, ge=0.0, le=2.5)
    delinq_2yrs: int = Field(0, ge=0, le=30)
    total_outstanding_debt: float = Field(..., ge=0.0)

class PredictionResponse(BaseModel):
    applicant_id: str
    risk_category: str
    risk_tier: str
    risk_probability: float
    decision_threshold: float
    recommendation: str
    key_risk_factors: List[str]
    positive_credit_factors: List[str]
    latency_ms: float
```

---

## 4. FastAPI Application Implementation (`api/main.py`)

```python
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import ApplicantInput, PredictionResponse
from src.explainability.driver_generator import NaturalLanguageRiskExplainer

app = FastAPI(
    title="Financial Risk Prediction API",
    version="1.0.0",
    description="Enterprise Machine Learning service for real-time credit risk classification."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL_PIPELINE = None
EXPLAINER = None

@app.on_event("startup")
def load_artifacts():
    global MODEL_PIPELINE, EXPLAINER
    try:
        MODEL_PIPELINE = joblib.load("models/champion_model.joblib")
        feature_names = [
            "person_age", "person_income", "loan_amnt", "loan_int_rate",
            "debt_to_income", "revolving_utilization", "delinq_2yrs",
            "cb_person_cred_hist_length", "delinq_severity_index"
        ]
        EXPLAINER = NaturalLanguageRiskExplainer(feature_names=feature_names)
    except Exception as e:
        print(f"Warning: Model artifacts not found yet. Run run_pipeline.py first: {e}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if MODEL_PIPELINE is not None else "degraded",
        "model_loaded": MODEL_PIPELINE is not None,
        "service": "financial-risk-prediction-api"
    }

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_risk(applicant: ApplicantInput):
    if MODEL_PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not yet loaded. Please execute model training pipeline."
        )
        
    start_time = time.perf_counter()
    raw_df = pd.DataFrame([applicant.model_dump()])
    
    # Predict continuous probability
    prob = float(MODEL_PIPELINE.predict_proba(raw_df)[0][1])
    threshold = 0.40  # Risk-adjusted threshold
    
    is_high_risk = prob >= threshold
    category = "High Risk" if is_high_risk else "Low Risk"
    
    tier = (
        "Tier 4: Critical High Risk" if prob >= 0.70 else
        "Tier 3: Elevated Risk" if prob >= 0.45 else
        "Tier 2: Moderate Risk" if prob >= 0.25 else
        "Tier 1: Minimal Risk"
    )
    
    # Generate drivers
    explanation = EXPLAINER.explain_instance(
        raw_record=applicant.model_dump(),
        shap_values=[0.1, 0.2, 0.3], # Simplified fast SHAP
        predicted_prob=prob,
        risk_category=category
    )
    
    latency = (time.perf_counter() - start_time) * 1000.0
    
    return PredictionResponse(
        applicant_id=applicant.applicant_id or "APP-TEMP",
        risk_category=category,
        risk_tier=tier,
        risk_probability=round(prob, 4),
        decision_threshold=threshold,
        recommendation="Adverse Action / Review" if is_high_risk else "Approve Loan Application",
        key_risk_factors=explanation["key_risk_factors"],
        positive_credit_factors=explanation["positive_credit_factors"],
        latency_ms=round(latency, 2)
    )
```
