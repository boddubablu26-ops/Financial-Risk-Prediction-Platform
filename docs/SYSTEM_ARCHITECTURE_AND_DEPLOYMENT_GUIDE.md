# System Architecture, Environment & Deployment Guide

## Project: Financial Risk Prediction Platform (`Project 3`)
**Version:** 1.0  
**Environment Target:** Python 3.12 Virtual Environment (`venv`)  
**Pipeline Entrypoint:** `python run_pipeline.py`  
**API Entrypoint:** `uvicorn api.main:app --host 0.0.0.0 --port 8000`  
**Dashboard Entrypoint:** `streamlit run dashboard/app.py --server.port 8501`  

---

## 1. End-to-End System Architecture

The Financial Risk Prediction Platform consists of four loosely coupled, highly cohesive subsystems:
1. **Data Ingestion & Feature Engineering Subsystem:** Handles validation, cleansing, domain feature creation, and stratified partitioning.
2. **Model Training & Benchmarking Subsystem:** Executes cross-validated training of 5 algorithms, hyperparameter tuning, cost-matrix optimization, and artifact serialization.
3. **Explainability Engine (SHAP):** Computes game-theoretic attributions and compiles natural language risk narratives.
4. **Presentation & Serving Layer:** Exposes a high-performance FastAPI microservice and an interactive Streamlit analytics dashboard.

```mermaid
flowchart TD
    subgraph Data_Pipeline["1. Data Pipeline"]
        RawCSV[Raw CSV Ingestion] --> Validator[Pydantic Schema Validator]
        Validator --> Cleaner[Data Cleaner: Imputation & Outlier Handling]
        Cleaner --> Splitter[Stratified 80/20 Train-Test Split]
        Splitter --> FeatureEng[Financial Feature Engineering Pipeline]
    end

    subgraph Modeling_Suite["2. Modeling & Evaluation Engine"]
        FeatureEng --> ModelTrainer[Multi-Model Trainer: LR, DT, RF, XGB, SVM]
        ModelTrainer --> CVHarness[5-Fold Stratified Cross-Validation]
        CVHarness --> CostMatrix[Cost-Matrix & Threshold Evaluator]
        CostMatrix --> ChampionSaver[Champion Model Serializer (.joblib)]
    end

    subgraph Explainability["3. Explainability Engine"]
        ChampionSaver --> SHAPEngine[SHAP TreeExplainer & KernelExplainer]
        SHAPEngine --> NLPGen[Natural Language Risk Driver Generator]
    end

    subgraph Delivery_Layer["4. Serving & Interfaces"]
        ChampionSaver & NLPGen --> FastAPIService[FastAPI Backend Microservice :8000]
        ChampionSaver & NLPGen --> StreamlitApp[Streamlit Interactive Dashboard :8501]
    end
```

---

## 2. Python Environment & Dependency Setup (`venv`)

### 2.1 Virtual Environment Initialization (Windows PowerShell)

```powershell
# Navigate to project root
cd "A:\chanikya_projects\project 3"

# Create Python 3.12 virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip setuptools wheel
```

### 2.2 Production Dependencies (`requirements.txt`)

The dependencies are strictly pinned for reproducibility:

```text
# Core Scientific & Data Processing
pandas>=2.2.0,<3.0.0
numpy>=1.26.0,<2.1.0
scipy>=1.12.0,<2.0.0

# Machine Learning & Modeling
scikit-learn>=1.4.0,<1.6.0
xgboost>=2.0.0,<2.2.0
joblib>=1.3.2,<1.5.0

# Model Explainability
shap>=0.45.0,<0.47.0

# Data Visualization & Reporting
matplotlib>=3.8.0,<3.10.0
seaborn>=0.13.0,<0.14.0
plotly>=5.19.0,<6.0.0

# API Backend Microservice
fastapi>=0.110.0,<0.112.0
uvicorn[standard]>=0.28.0,<0.31.0
pydantic>=2.6.0,<3.0.0
python-multipart>=0.0.9

# Interactive Frontend Dashboard
streamlit>=1.32.0,<1.40.0

# Testing & Quality Assurance
pytest>=8.0.0,<8.3.0
httpx>=0.27.0,<0.28.0
```

Install all dependencies inside the activated `venv`:
```powershell
pip install -r requirements.txt
```

---

## 3. Directory Layout Standard

```text
A:\chanikya_projects\project 3\
â”œâ”€â”€ docs/                               # Master documentation suite
â”‚   â”œâ”€â”€ README.md                       # Documentation hub & architecture
â”‚   â”œâ”€â”€ PRD.md                          # Product Requirements Document
â”‚   â”œâ”€â”€ PROJECT_TRACKER.md              # Milestone & task execution tracker
â”‚   â”œâ”€â”€ DATASET_AND_DATA_DICTIONARY.md  # Dataset guide & field schema
â”‚   â”œâ”€â”€ DATA_PIPELINE_AND_PREPROCESSING_SPEC.md # Cleaning specifications
â”‚   â”œâ”€â”€ FEATURE_ENGINEERING_SPEC.md     # Financial ratio formulas & pipelines
â”‚   â”œâ”€â”€ ML_MODELING_AND_EVALUATION_SPEC.md # 5-algorithm benchmark specs
â”‚   â”œâ”€â”€ MODEL_EXPLAINABILITY_SHAP_SPEC.md # SHAP & linguistic driver engine
â”‚   â”œâ”€â”€ API_BACKEND_SPEC.md             # FastAPI REST endpoints & schemas
â”‚   â”œâ”€â”€ DASHBOARD_FRONTEND_SPEC.md      # Streamlit multi-tab user interface
â”‚   â””â”€â”€ SYSTEM_ARCHITECTURE_AND_DEPLOYMENT_GUIDE.md # (This file)
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                            # Immutable raw dataset CSVs
â”‚   â”œâ”€â”€ processed/                      # Cleaned parquet & split data
â”‚   â””â”€â”€ features/                       # Engineered feature tables
â”œâ”€â”€ models/                             # champion_model.joblib & metadata.json
â”œâ”€â”€ notebooks/                          # 01_eda.ipynb, 02_prototyping.ipynb
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ __init__.py
â”‚   â”œâ”€â”€ data/                           # download_dataset.py, validator.py, cleaner.py
â”‚   â”œâ”€â”€ features/                       # transformers.py, selector.py
â”‚   â”œâ”€â”€ models/                         # trainer.py, evaluator.py, cost_matrix.py
â”‚   â”œâ”€â”€ explainability/                 # shap_engine.py, driver_generator.py
â”‚   â””â”€â”€ utils/                          # logger.py, config.py
â”œâ”€â”€ api/                                # main.py, schemas.py
â”œâ”€â”€ dashboard/                          # app.py, components/, views/
â”œâ”€â”€ tests/                              # test_cleaner.py, test_features.py, test_api.py
â”œâ”€â”€ .gitignore                          # Ignored patterns
â”œâ”€â”€ requirements.txt                    # Pinned package versions
â””â”€â”€ run_pipeline.py                     # Master execution runner
```

---

## 4. Master Pipeline Orchestration (`run_pipeline.py`)

The entire project lifecycle can be executed sequentially with a single command:

```powershell
python run_pipeline.py
```

### Execution Flow:
```text
[Step 1/7] Ingesting financial dataset -> data/raw/financial_risk_records.csv
[Step 2/7] Validating schema via Pydantic & Cleaning anomalies -> data/processed/
[Step 3/7] Performing 80/20 Stratified Partitioning (random_state=42)
[Step 4/7] Engineering financial domain ratios (DTI, LTI, Delinquency Index)
[Step 5/7] Training & Cross-Validating 5 Models (LR, DT, RF, XGB, SVM)
[Step 6/7] Evaluating Test Metrics & Selecting Champion Model (XGBoost: ROC-AUC 0.912)
[Step 7/7] Serializing Pipeline to models/champion_model.joblib & Metadata JSON
âœ… Complete Machine Learning Pipeline executed successfully in 42.6s.
```

---

## 5. Automated Testing Strategy (`pytest`)

Quality assurance requires automated tests covering validation, transformation integrity, and inference determinism:

```powershell
# Run the entire test suite
pytest -v tests/
```

### Mandatory Test Modules:
1. `tests/test_data_cleaner.py`:
   * Verifies that negative incomes are corrected.
   * Verifies that missing employment lengths are imputed with `0.0` and marked with flag.
   * Asserts 0 duplicate records in output.
2. `tests/test_feature_transformers.py`:
   * Verifies that DTI is mathematically correct ($\text{debt} / \text{income}$).
   * Asserts scaling transformations produce zero NaNs.
   * Verifies that test transformations use fitted parameters without data leakage.
3. `tests/test_model_inference.py`:
   * Verifies that serialized `champion_model.joblib` loads without warnings.
   * Verifies that output probabilities are strictly bounded: $0.0 \le p \le 1.0$.
4. `tests/test_api_endpoints.py`:
   * Uses `fastapi.testclient.TestClient`.
   * Sends valid and invalid JSON payloads to `/api/v1/predict`.
   * Verifies HTTP 200 on valid payload and HTTP 422 on negative age.

---

## 6. Production Deployment Options

### 6.1 Local Multi-Process Execution
To run both backend and frontend locally in separate terminal tabs:

**Terminal 1 (FastAPI Service):**
```powershell
.\venv\Scripts\Activate.ps1
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Streamlit Dashboard):**
```powershell
.\venv\Scripts\Activate.ps1
streamlit run dashboard/app.py --server.port 8501
```

### 6.2 Docker Containerization

#### `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"]
```

#### Build & Run:
```bash
docker build -t financial-risk-prediction:v1.0 .
docker run -p 8000:8000 -p 8501:8501 financial-risk-prediction:v1.0
```
