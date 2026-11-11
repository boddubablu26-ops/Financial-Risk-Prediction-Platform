# Financial Risk Prediction Platform (FRP-ML)
## Master Documentation Hub

Welcome to the technical documentation repository for the **Financial Risk Prediction Platform**. This documentation suite has been engineered to provide the development team with complete, unambiguous architectural blueprints, specifications, algorithms, validation matrices, and operational setup guides to build the product with utmost accuracy.

---

## 📚 Core System Specifications

| Document | Description | Key Topics Covered |
| :--- | :--- | :--- |
| [**PRD.md**](PRD.md) | **Product Requirements Document** | Business context, target personas, functional requirements (FR 1.1–8.2), non-functional requirements, acceptance criteria. |
| [**ARCHITECTURE.md**](ARCHITECTURE.md) | **System Architecture & Technical Design** | High-level system diagram, layered component design, sequence workflows, technology stack justification, security & reproducibility. |
| [**DATASET_SPECIFICATION.md**](DATASET_SPECIFICATION.md) | **Dataset Specification & Management** | Benchmark comparison (why 32,581 records vs German credit), 12-column data dictionary, imputation strategies, outlier bounds, derived features. |
| [**ML_PIPELINE_DESIGN.md**](ML_PIPELINE_DESIGN.md) | **Machine Learning & Modeling Design** | 5 classification algorithms, hyperparameter spaces, 5-fold stratified CV, class imbalance handling, cost-sensitive threshold tuning, SHAP architecture. |
| [**API_AND_DASHBOARD_SPECS.md**](API_AND_DASHBOARD_SPECS.md) | **API & Dashboard Specifications** | FastAPI schemas & REST endpoints (`/predict`, `/explain`, `/models`), 5-page Streamlit dashboard wireframes, UI styling standards. |
| [**TRACKER.md**](TRACKER.md) | **Implementation Tracker & Verification Matrix** | 57 granular tasks across 10 development phases, priority tags, definition of done (DoD), and live progress tracking. |
| [**DEVELOPMENT_GUIDE.md**](DEVELOPMENT_GUIDE.md) | **Developer Setup & Workflow Guide** | Step-by-step virtual environment (`venv`) setup, dependency installation, execution commands, testing with pytest, and coding standards. |

---

## 🔬 Deep-Dive Module Specifications

For specialized engineering assignments, deep-dive specifications are also available:

- 📊 [**DATASET_AND_DATA_DICTIONARY.md**](DATASET_AND_DATA_DICTIONARY.md): Granular statistical distributions, variable encodings, and domain metrics.
- 🧹 [**DATA_PIPELINE_AND_PREPROCESSING_SPEC.md**](DATA_PIPELINE_AND_PREPROCESSING_SPEC.md): Zero-leakage data cleaning, imputation, and validation specs.
- 🧮 [**FEATURE_ENGINEERING_SPEC.md**](FEATURE_ENGINEERING_SPEC.md): Mathematical formulations for financial debt-to-income and burden indices.
- 🤖 [**ML_MODELING_AND_EVALUATION_SPEC.md**](ML_MODELING_AND_EVALUATION_SPEC.md): Hyperparameter tuning grids, cost-sensitive loss curves, and evaluation matrices.
- 🔍 [**MODEL_EXPLAINABILITY_SHAP_SPEC.md**](MODEL_EXPLAINABILITY_SHAP_SPEC.md): SHAP TreeExplainer, waterfall decomposition, and compliance reporting.
- 🌐 [**API_BACKEND_SPEC.md**](API_BACKEND_SPEC.md): Pydantic request/response schemas and ASGI deployment architecture.
- 🖥️ [**DASHBOARD_FRONTEND_SPEC.md**](DASHBOARD_FRONTEND_SPEC.md): Page-by-page Streamlit wireframes, reactive state management, and UX design.
- 🚀 [**SYSTEM_ARCHITECTURE_AND_DEPLOYMENT_GUIDE.md**](SYSTEM_ARCHITECTURE_AND_DEPLOYMENT_GUIDE.md): Production containerization, health checks, and CI/CD pipelines.
- 📋 [**PROJECT_TRACKER.md**](PROJECT_TRACKER.md): Extended sprint backlog and quality checklist.

---

## 🏛️ Architecture Snapshot

```
                       ┌─────────────────────────┐
                       │   Raw Financial Data    │
                       │ (32,581 Benchmark CSV)  │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Ingestion & Validation  │
                       │    (Data Cleaner)       │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Preprocessing Pipeline  │
                       │(Ratios + Encode + Scale)│
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │   Multi-Model Engine    │
                       │(5 Models + 5-Fold CV)   │
                       └────────────┬────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────┐
                       │ Evaluation & Cost Matrix│
                       │ (Champion Optimization) │
                       └────────────┬────────────┘
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
          ┌───────────────────────┐   ┌───────────────────────┐
          │  FastAPI REST Server  │   │  Streamlit Dashboard  │
          │   (/predict, /explain)│   │  (5 Analytical Pages) │
          └───────────┬───────────┘   └───────────┬───────────┘
                      │                           │
                      └─────────────┬─────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │ Explainability Engine   │
                       │ (SHAP Local & Global)   │
                       └─────────────────────────┘
```

---

## 🎯 Quick Navigation for Developers

- **To understand the product goals & scope:** Read [`docs/PRD.md`](PRD.md).
- **To understand the system structure & design patterns:** Read [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).
- **To review column definitions, missing value rules, and ratios:** Read [`docs/DATASET_SPECIFICATION.md`](DATASET_SPECIFICATION.md).
- **To review model hyperparameters, tuning, and SHAP logic:** Read [`docs/ML_PIPELINE_DESIGN.md`](ML_PIPELINE_DESIGN.md).
- **To inspect API contracts and UI wireframes:** Read [`docs/API_AND_DASHBOARD_SPECS.md`](API_AND_DASHBOARD_SPECS.md).
- **To check current tasks, sprint backlog, and acceptance criteria:** Read [`docs/TRACKER.md`](TRACKER.md).
- **To set up Python venv, install packages, and run code:** Read [`docs/DEVELOPMENT_GUIDE.md`](DEVELOPMENT_GUIDE.md).
