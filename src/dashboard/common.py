"""Shared Streamlit styling, sidebar, and cached artifact loaders."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import (
    COMPARISON_PATH,
    EDA_SUMMARY_PATH,
    RAW_DATA_PATH,
    SAMPLE_APPLICANT_PATH,
    SHAP_SAMPLES_PATH,
    SHAP_SUMMARY_PATH,
)
from src.models.model_registry import read_json
from src.models.predictor import RiskRuntime

BRAND = "#4F46E5"
LOW = "#10B981"
HIGH = "#EF4444"
AMBER = "#F59E0B"
INK = "#0F172A"

MODEL_COLORS = {
    "Logistic Regression": "#4F46E5",
    "Decision Tree": "#0EA5E9",
    "Random Forest": "#10B981",
    "XGBoost Classifier": "#EF4444",
    "Support Vector Machine": "#F59E0B",
}


def ensure_project_root(anchor: str, parents_up: int) -> None:
    """Put the repository root on ``sys.path`` for Streamlit page scripts.

    Args:
        anchor: ``__file__`` of the calling script.
        parents_up: How many parents to climb to the repository root.
    """

    root = str(Path(anchor).resolve().parents[parents_up])
    if root not in sys.path:
        sys.path.insert(0, root)


def apply_theme() -> None:
    """Inject the platform color system."""

    st.markdown(
        f"""
        <style>
        .stApp {{ background: #F8FAFC; color: {INK}; }}
        h1, h2, h3 {{ letter-spacing: -0.02em; }}
        .frp-hero {{
            background: linear-gradient(120deg, #0F172A 0%, #312E81 60%, {BRAND} 100%);
            color: white; border-radius: 16px; padding: 22px 26px; margin-bottom: 16px;
        }}
        .frp-hero h1 {{ color: white; margin: 0 0 6px 0; font-size: 1.8rem; }}
        .frp-hero p {{ margin: 0; color: #E2E8F0; }}
        .badge {{
            display: inline-block; padding: 8px 14px; border-radius: 999px;
            font-weight: 700; letter-spacing: 0.04em;
        }}
        .badge-low {{ background: #D1FAE5; color: #065F46; }}
        .badge-high {{ background: #FEE2E2; color: #991B1B; }}
        div[data-testid="stMetric"] {{
            background: white; border: 1px solid #E2E8F0; border-radius: 14px;
            padding: 10px 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Brand the sidebar and show whether the champion model is loaded."""

    runtime = get_runtime()
    st.sidebar.markdown("### Financial Risk Platform")
    st.sidebar.caption("Credit default scoring, model audit, and SHAP explanations.")
    if runtime.ready:
        st.sidebar.success(f"Champion loaded\n\n{runtime.model_name}")
        st.sidebar.caption(f"Decision threshold {runtime.threshold:.2f}")
    else:
        st.sidebar.error("Champion model is not trained yet.")
        st.sidebar.code("python -m src.models.train", language="powershell")


def hero(title: str, subtitle: str) -> None:
    """Render the page banner.

    Args:
        title: Page title.
        subtitle: One-line description.
    """

    st.markdown(
        f"<div class='frp-hero'><h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_runtime() -> RiskRuntime:
    """Load the champion pipeline once per Streamlit session process."""

    return RiskRuntime.load()


@st.cache_data(show_spinner=False)
def load_raw() -> pd.DataFrame:
    """Load the raw benchmark CSV."""

    if not RAW_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> dict[str, Any]:
    """Load a metadata JSON document."""

    return read_json(Path(path_str))


def comparison() -> dict[str, Any]:
    """Model leaderboard payload."""

    return load_json(str(COMPARISON_PATH))


def eda_summary() -> dict[str, Any]:
    """Profiling payload written by the training pipeline."""

    return load_json(str(EDA_SUMMARY_PATH))


def shap_summary() -> dict[str, Any]:
    """Global SHAP ranking."""

    return load_json(str(SHAP_SUMMARY_PATH))


def shap_samples() -> dict[str, Any]:
    """Beeswarm sample stored at training time."""

    return load_json(str(SHAP_SAMPLES_PATH))


def sample_applicant() -> dict[str, Any]:
    """Default form values."""

    payload = read_json(SAMPLE_APPLICANT_PATH)
    if payload:
        return payload
    return {
        "person_age": 30,
        "person_income": 65000.0,
        "person_home_ownership": "RENT",
        "person_emp_length": 5.0,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "C",
        "loan_amnt": 12500.0,
        "loan_int_rate": 11.89,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 6,
    }


def require_model() -> RiskRuntime | None:
    """Return the runtime, or stop the page with setup instructions."""

    runtime = get_runtime()
    if runtime.ready:
        return runtime
    st.warning("Train the champion model before using this page.")
    st.code("python -m src.models.train", language="powershell")
    return None
