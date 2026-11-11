# Developer Setup & Engineering Workflow Guide

- **Project:** Financial Risk Prediction Platform (FRP-ML)
- **Document Version:** 1.0.0
- **Status:** Active Developer Guide
- **Operating Systems Supported:** Windows 10/11, Linux (Ubuntu 20.04+), macOS (Apple Silicon / Intel)

---

## 1. Prerequisites

Before starting development, ensure the following tools are installed:
- **Python:** Version `3.10.x`, `3.11.x`, or `3.12.x` (Verify via `python --version`)
- **Git:** Version `2.30+` (Verify via `git --version`)
- **PowerShell (Windows)** or **Bash/Zsh (Linux/macOS)**

---

## 2. Virtual Environment Setup

Always develop inside an isolated Python virtual environment to avoid dependency conflicts.

### Step 2.1: Clone or Navigate to the Workspace
```powershell
# Windows PowerShell
cd "A:\chanikya_projects\project 3"
```

### Step 2.2: Create the Virtual Environment (`venv`)
```powershell
# Create venv named '.venv'
python -m venv .venv
```

### Step 2.3: Activate the Virtual Environment

**On Windows (PowerShell):**
```powershell
# If execution policy blocks script execution, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt `cmd`):**
```cmd
.venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
source .venv/bin/activate
```

*(Once activated, your terminal prompt should display `(.venv)`)*

---

## 3. Dependency Installation

Upgrade `pip` and install all pinned production and development dependencies:

```powershell
# Upgrade package installer
python -m pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

---

## 4. Project Directory Hierarchy

The project follows a standard, modular Python package structure:

```
project 3/
├── .venv/                              # Python virtual environment (gitignored)
├── docs/                               # Comprehensive project documentation
│   ├── README.md                       # Documentation hub & reading index
│   ├── PRD.md                          # Product Requirements Document
│   ├── ARCHITECTURE.md                 # System Architecture & Technical Design
│   ├── DATASET_SPECIFICATION.md        # Dataset guide, data dictionary & cleaning rules
│   ├── ML_PIPELINE_DESIGN.md           # 5-Model ML specs, CV, cost matrix, SHAP
│   ├── API_AND_DASHBOARD_SPECS.md      # FastAPI & Streamlit specifications
│   ├── TRACKER.md                      # Milestone & Task Implementation Tracker
│   └── DEVELOPMENT_GUIDE.md            # This setup and execution guide
├── data/                               # Dataset storage
│   ├── raw/                            # Untouched raw benchmark dataset
│   ├── processed/                      # Preprocessed train/test arrays
│   └── sample/                         # Sample applicant JSON payloads
├── models/                             # Serialized models and evaluation artifacts
│   ├── saved_models/                   # .joblib model files
│   └── metadata/                       # JSON metrics, confusion matrices, SHAP summaries
├── src/                                # Core application source code
│   ├── __init__.py
│   ├── data/                           # Ingestion, validation, cleaning, splitting
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── validator.py
│   │   └── cleaner.py
│   ├── features/                       # Transformers, scaling, encoding pipelines
│   │   ├── __init__.py
│   │   ├── transformers.py
│   │   └── feature_pipeline.py
│   ├── models/                         # Model definitions, CV training, tuning
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── model_registry.py
│   │   └── hyperparameter_tuning.py
│   ├── evaluation/                     # Metric calculations, cost optimization, curves
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── cost_matrix.py
│   │   └── plots.py
│   ├── explainability/                 # SHAP TreeExplainer, waterfall, narratives
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   └── report_generator.py
│   ├── api/                            # FastAPI REST application
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── predict.py
│   │       ├── explain.py
│   │       └── models.py
│   └── dashboard/                      # Streamlit interactive application
│       ├── app.py                      # Main entrypoint
│       └── views/
│           ├── overview.py
│           ├── data_analysis.py
│           ├── model_performance.py
│           ├── risk_prediction.py
│           └── explainability.py
├── scripts/                            # Helper and utility scripts
│   └── download_or_generate_dataset.py # Automated benchmark dataset acquisition
├── tests/                              # Automated test suite
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
├── requirements.txt                    # Pinned Python dependencies
├── .gitignore                          # Standard git ignore definitions
└── README.md                           # Master project README
```

---

## 5. Standard Execution Workflow

Follow these steps in sequence when implementing and testing the system:

### Step 5.1: Acquire or Generate the Dataset
Execute the benchmark dataset acquisition script:
```powershell
python scripts/download_or_generate_dataset.py
```
*Expected Output: Confirms creation of `data/raw/credit_risk_dataset.csv` with 32,581 records.*

### Step 5.2: Execute the ML Training & Evaluation Pipeline
Run the multi-model training pipeline:
```powershell
python -m src.models.train
```
*Expected Output:*
- Fits preprocessor pipeline on training data.
- Trains 5 classification models using 5-Fold Stratified Cross-Validation.
- Evaluates models on the holdout test set.
- Generates `models/metadata/model_comparison.json`.
- Identifies and serializes the Champion Model pipeline to `models/saved_models/champion_pipeline.joblib`.
- Pre-computes SHAP explainer objects to `models/saved_models/shap_explainer.joblib`.

### Step 5.3: Run the FastAPI REST Server
Start the backend API using Uvicorn:
```powershell
uvicorn src.api.main:app --reload --port 8000
```
- Open your browser to `http://localhost:8000/docs` to test endpoints via Swagger UI.

### Step 5.4: Launch the Interactive Streamlit Dashboard
In a separate terminal (with `.venv` activated):
```powershell
streamlit run src/dashboard/app.py
```
- Open `http://localhost:8501` to interact with the full dashboard across all 5 pages.

---

## 6. Automated Testing & Verification

Run the full automated test suite using `pytest`:

```powershell
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest --cov=src tests/ --cov-report=term-missing
```

### Test Suite Structure
1. `tests/test_data.py`:
   - Validates input bounds, detects out-of-range ages, verifies missing value imputation.
2. `tests/test_features.py`:
   - Verifies custom ratio calculations (LTI, interest burden, credit-to-age).
   - Validates output shapes and confirms zero NaNs after ColumnTransformer.
3. `tests/test_models.py`:
   - Verifies serialized champion model loads properly.
   - Tests deterministic probability generation for test samples.
4. `tests/test_api.py`:
   - Tests FastAPI `/health`, `/predict`, `/explain`, and `/models` using `httpx.AsyncClient` / `TestClient`.

---

## 7. Code Quality & Standards

- **Formatting & Linting:** Adhere to PEP 8. Use `ruff` or `flake8` for linting.
- **Type Annotations:** All functions must have explicit type annotations for parameters and return types (e.g. `def predict_risk(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:`).
- **Docstrings:** Use Google-style docstrings for all classes and public functions.
- **Random Seeds:** Always set `random_state=42` across all scikit-learn, XGBoost, and train_test_split functions to ensure deterministic execution.

