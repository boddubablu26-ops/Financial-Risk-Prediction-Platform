# Financial Risk Prediction Platform (FRP-ML)

An end-to-end enterprise machine learning platform for assessing credit default and financial risk using multi-model benchmarking (Logistic Regression, Decision Trees, Random Forest, XGBoost, SVM) with explainable AI (SHAP), a high-performance FastAPI backend, and an interactive Streamlit dashboard.

---

## 📖 Complete Documentation

The complete project specification, architecture blueprints, data dictionaries, ML designs, and implementation trackers are maintained in the [`docs/`](docs/) directory:

- 📋 [**Product Requirements Document (PRD)**](docs/PRD.md)
- 🏗️ [**System Architecture & Technical Design**](docs/ARCHITECTURE.md)
- 📊 [**Dataset Specification & Benchmark Guide**](docs/DATASET_SPECIFICATION.md)
- 🧠 [**Machine Learning Pipeline Design**](docs/ML_PIPELINE_DESIGN.md)
- 🔌 [**API & Dashboard Specifications**](docs/API_AND_DASHBOARD_SPECS.md)
- ✅ [**Implementation Tracker & Verification Matrix**](docs/TRACKER.md)
- 🛠️ [**Developer Setup & Execution Guide**](docs/DEVELOPMENT_GUIDE.md)

---

## ⚡ Quick Start

### 1. Setup Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
pip install -r requirements.txt
```

### 2. Acquire or Generate Benchmark Dataset
```powershell
python scripts/download_or_generate_dataset.py
```

### 3. Run Machine Learning Pipeline
```powershell
python -m src.models.train
```

### 4. Launch FastAPI REST Server
```powershell
uvicorn src.api.main:app --reload --port 8000
```
Visit Swagger UI: `http://localhost:8000/docs`

### 5. Launch Streamlit Analytics Dashboard
```powershell
streamlit run src/dashboard/app.py
```
Visit Dashboard: `http://localhost:8501`

---

## 📁 Repository Structure

```
project 3/
├── docs/                      # Full Technical Documentation & Specifications
├── data/                      # Raw and preprocessed financial datasets
├── models/                    # Serialized models (.joblib) and evaluation metadata
├── src/                       # Application Source Code
│   ├── data/                  # Ingestion, validation, cleaning
│   ├── features/              # Feature engineering & scikit-learn transformers
│   ├── models/                # Multi-model training and registry
│   ├── evaluation/            # Metrics, ROC/PR curves, cost optimization
│   ├── explainability/        # SHAP local and global explainers
│   ├── api/                   # FastAPI application & REST endpoints
│   └── dashboard/             # Multi-page Streamlit analytics dashboard
├── scripts/                   # Dataset generation and helper scripts
├── tests/                     # Unit, integration, and API tests
└── requirements.txt           # Pinned project dependencies
```
