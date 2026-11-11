"""Streamlit entry point for the Financial Risk Prediction Platform."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

VIEWS = Path(__file__).resolve().parent / "views"

st.set_page_config(
    page_title="FRP-ML | Financial Risk",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

navigation = st.navigation(
    [
        st.Page(
            VIEWS / "overview.py",
            title="Overview",
            icon=":material/dashboard:",
            default=True,
        ),
        st.Page(
            VIEWS / "data_analysis.py",
            title="Data Analysis",
            icon=":material/query_stats:",
        ),
        st.Page(
            VIEWS / "model_performance.py",
            title="Model Performance",
            icon=":material/leaderboard:",
        ),
        st.Page(
            VIEWS / "risk_prediction.py",
            title="Risk Prediction",
            icon=":material/crisis_alert:",
        ),
        st.Page(
            VIEWS / "explainability.py",
            title="Explainability",
            icon=":material/psychology:",
        ),
    ]
)
navigation.run()
