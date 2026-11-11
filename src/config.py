"""Project paths, seeds, and schema constants for the FRP-ML platform."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
N_CV_FOLDS: int = 5
MODEL_VERSION: str = "1.0.0"
API_VERSION: str = "1.0.0"
SERVICE_NAME: str = "financial-risk-prediction-api"

# Asymmetric lending costs from the modeling specification.
FP_COST: float = 1.0
FN_COST: float = 5.0
THRESHOLD_MIN: float = 0.10
THRESHOLD_MAX: float = 0.60
THRESHOLD_STEP: float = 0.02

RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "credit_risk_dataset.csv"
SAMPLE_APPLICANT_PATH: Path = PROJECT_ROOT / "data" / "sample" / "sample_applicant.json"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models" / "saved_models"
METADATA_DIR: Path = PROJECT_ROOT / "models" / "metadata"
PLOTS_DIR: Path = METADATA_DIR / "plots"

CHAMPION_PATH: Path = MODELS_DIR / "champion_pipeline.joblib"
CHAMPION_ALIAS_PATH: Path = PROJECT_ROOT / "models" / "champion_model.joblib"
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.joblib"
SHAP_EXPLAINER_PATH: Path = MODELS_DIR / "shap_explainer.joblib"
SHAP_BACKGROUND_PATH: Path = MODELS_DIR / "shap_background.npz"
COMPARISON_PATH: Path = METADATA_DIR / "model_comparison.json"
BEST_MODEL_PATH: Path = METADATA_DIR / "best_model_info.json"
BEST_MODEL_ALIAS_PATH: Path = PROJECT_ROOT / "models" / "model_metadata.json"
CONFUSION_PATH: Path = METADATA_DIR / "confusion_matrices.json"
SHAP_SUMMARY_PATH: Path = METADATA_DIR / "shap_global_summary.json"
SHAP_SAMPLES_PATH: Path = METADATA_DIR / "shap_samples.json"
EDA_SUMMARY_PATH: Path = METADATA_DIR / "eda_summary.json"
FEATURE_SELECTION_PATH: Path = METADATA_DIR / "feature_selection.json"
PREPROCESSING_METADATA_PATH: Path = PROCESSED_DIR / "preprocessing_metadata.json"

TARGET_COLUMN: str = "loan_status"

# Canonical 12-column credit-risk benchmark (11 predictors + target).
RAW_FEATURE_COLUMNS: list[str] = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

REQUIRED_TRAINING_COLUMNS: list[str] = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]

HOME_OWNERSHIP_VALUES: tuple[str, ...] = ("RENT", "MORTGAGE", "OWN", "OTHER")
LOAN_INTENT_VALUES: tuple[str, ...] = (
    "PERSONAL",
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "HOMEIMPROVEMENT",
    "DEBTCONSOLIDATION",
)
LOAN_GRADE_VALUES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
GRADE_ORDER: list[str] = list(LOAN_GRADE_VALUES)
DEFAULT_ON_FILE_VALUES: tuple[str, ...] = ("Y", "N")

# Skewed money and ratio features use RobustScaler. Near-symmetric features use StandardScaler.
ROBUST_FEATURES: list[str] = [
    "person_income",
    "loan_amnt",
    "loan_to_income_ratio",
    "interest_burden_index",
    "disposable_income",
]
STANDARD_FEATURES: list[str] = [
    "person_age",
    "person_emp_length",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "cred_hist_age_ratio",
    "stability_index",
    "payment_to_income",
    "financial_stress_score",
    "emp_length_missing_flag",
    "historical_risk_weight",
    "default_on_file_binary",
    "delinq_severity_index",
]
ORDINAL_FEATURES: list[str] = ["loan_grade"]
NOMINAL_FEATURES: list[str] = ["person_home_ownership", "loan_intent"]

AGE_BINS: list[int] = [0, 25, 35, 45, 60, 200]
AGE_LABELS: list[str] = ["le_25", "26_35", "36_45", "46_60", "61_plus"]

DISPLAY_NAMES: dict[str, str] = {
    "person_age": "Applicant age",
    "person_income": "Annual income",
    "person_emp_length": "Employment length",
    "person_home_ownership": "Home ownership",
    "loan_intent": "Loan purpose",
    "loan_grade": "Loan grade",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan as percent of income",
    "loan_to_income_ratio": "Loan-to-income ratio",
    "interest_burden_index": "Interest burden",
    "cred_hist_age_ratio": "Credit-history-to-age ratio",
    "cb_person_cred_hist_length": "Credit history length",
    "cb_person_default_on_file": "Prior default on file",
    "default_on_file_binary": "Prior default on file",
    "historical_risk_weight": "Historical default weighted by grade",
    "stability_index": "Employment stability",
    "payment_to_income": "Payment-to-income ratio",
    "disposable_income": "Disposable monthly income",
    "financial_stress_score": "Financial stress score",
    "emp_length_missing_flag": "Employment length missing",
    "delinq_severity_index": "Delinquency severity",
    "debt_to_income": "Debt-to-income ratio",
    "revolving_utilization": "Credit line utilization",
    "delinq_2yrs": "Recent delinquencies",
    "total_outstanding_debt": "Outstanding debt",
}
