"""Page bodies for the five Streamlit analytical views."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import HOME_OWNERSHIP_VALUES, LOAN_GRADE_VALUES, LOAN_INTENT_VALUES
from src.dashboard.common import (
    AMBER,
    BRAND,
    HIGH,
    LOW,
    MODEL_COLORS,
    comparison,
    eda_summary,
    hero,
    load_raw,
    require_model,
    sample_applicant,
    shap_samples,
    shap_summary,
)
from src.eda.visualizations import (
    box_by_risk,
    correlation_heatmap,
    distribution_figure,
    intent_bar,
    missingness_bar,
    target_donut,
)
from src.models.predictor import recommendation_detail


def render_overview() -> None:
    """Executive portfolio snapshot and champion banner."""

    hero("Overview", "Portfolio default mix, champion quality, and the model leaderboard.")
    raw = load_raw()
    board = comparison()
    models = board.get("models") or []
    champion = next((item for item in models if item.get("is_champion")), None)
    default_rate = float(raw["loan_status"].mean()) if "loan_status" in raw.columns and len(raw) else None
    cards = st.columns(4)
    cards[0].metric("Records analyzed", f"{len(raw):,}" if len(raw) else "—")
    cards[1].metric("Portfolio default rate", f"{default_rate:.1%}" if default_rate is not None else "—")
    cards[2].metric("Champion model", champion["model_name"] if champion else "Not trained")
    cards[3].metric("Test ROC-AUC", f"{champion['roc_auc']:.3f}" if champion else "—")
    if len(raw) and "loan_status" in raw.columns:
        left, right = st.columns(2)
        left.plotly_chart(target_donut(raw), use_container_width=True)
        right.plotly_chart(intent_bar(raw), use_container_width=True)
    if models:
        st.subheader("Model leaderboard")
        st.dataframe(_leaderboard_frame(models), use_container_width=True, hide_index=True)
    else:
        st.info("Run the training pipeline to fill the leaderboard.")
    profile = eda_summary().get("cleaned") or {}
    if profile:
        st.caption(
            f"Cleaned modeling table: {profile.get('rows', 0):,} rows, "
            f"completeness {profile.get('completeness', 0):.1%}."
        )


def render_eda() -> None:
    """Distributions, missingness, correlation, and outlier boxes on the raw benchmark."""

    hero("Exploratory data analysis", "Raw benchmark profile, including the missing interest rates the cleaner must impute.")
    raw = load_raw()
    if raw.empty:
        st.warning("Raw dataset was not found. Run scripts/download_or_generate_dataset.py.")
        return
    summary = eda_summary().get("raw") or {}
    columns = st.columns(3)
    columns[0].metric("Rows", f"{len(raw):,}")
    columns[1].metric("Columns", f"{raw.shape[1]}")
    columns[2].metric("Completeness", f"{(1 - raw.isna().to_numpy().mean()):.1%}")
    st.plotly_chart(missingness_bar(raw), use_container_width=True)
    numeric = [column for column in ["person_income", "loan_amnt", "loan_int_rate", "person_age", "person_emp_length"] if column in raw.columns]
    choice = st.selectbox("Distribution", numeric, index=0)
    log_scale = st.checkbox("Log scale", value=choice == "person_income")
    st.plotly_chart(distribution_figure(raw, choice, log_x=log_scale), use_container_width=True)
    method = st.selectbox("Correlation method", ["pearson", "spearman"])
    st.plotly_chart(correlation_heatmap(raw, method=method), use_container_width=True)
    working = raw.copy()
    working["loan_to_income_ratio"] = working["loan_amnt"] / working["person_income"]
    st.plotly_chart(box_by_risk(working, "loan_to_income_ratio"), use_container_width=True)
    if summary.get("describe"):
        with st.expander("Descriptive statistics"):
            st.dataframe(pd.DataFrame(summary["describe"]).T, use_container_width=True)
    pairs = eda_summary().get("collinear_pairs_pearson") or []
    if pairs:
        st.subheader("Collinear pairs (|r| ≥ 0.80)")
        st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)


def render_performance() -> None:
    """Leaderboard, ROC, precision-recall, confusion matrix, and cost curve."""

    hero("Model performance", "Five classifiers compared on the same stratified holdout, with a cost-sensitive threshold.")
    models = comparison().get("models") or []
    if not models:
        st.warning("No evaluation file yet. Train the models to populate this studio.")
        return
    st.dataframe(_leaderboard_frame(models), use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    left.plotly_chart(_roc_figure(models), use_container_width=True)
    right.plotly_chart(_pr_figure(models), use_container_width=True)
    names = [item["model_name"] for item in models]
    champion_name = next((item["model_name"] for item in models if item.get("is_champion")), names[0])
    selected_name = st.selectbox("Confusion matrix", names, index=names.index(champion_name))
    selected = next(item for item in models if item["model_name"] == selected_name)
    st.plotly_chart(_confusion_figure(selected), use_container_width=True)
    st.subheader("Cost-sensitive threshold")
    st.caption("Total cost = 1 × false positives + 5 × false negatives. The red marker is the saved optimum.")
    curve = selected.get("cost_curve") or []
    if not curve:
        return
    thresholds = [point["threshold"] for point in curve]
    chosen = st.slider(
        "Decision threshold",
        min_value=float(min(thresholds)),
        max_value=float(max(thresholds)),
        value=float(selected["optimal_threshold"]),
        step=0.02,
    )
    point = min(curve, key=lambda item: abs(item["threshold"] - chosen))
    metrics = st.columns(4)
    metrics[0].metric("Expected cost", f"{point['expected_cost']:.0f}")
    metrics[1].metric("False negatives", f"{point['fn']}")
    metrics[2].metric("False positives", f"{point['fp']}")
    metrics[3].metric("Saved optimum", f"{selected['optimal_threshold']:.2f}")
    st.plotly_chart(_cost_figure(selected, chosen), use_container_width=True)


def render_prediction() -> None:
    """Single-applicant form and CSV batch scoring."""

    hero("Risk prediction", "Score one application or a CSV batch with the champion pipeline.")
    runtime = require_model()
    if runtime is None:
        return
    single, batch = st.tabs(["Single applicant", "Batch CSV"])
    defaults = sample_applicant()
    with single:
        with st.form("single-score"):
            left, right = st.columns(2)
            age = left.number_input("Age", min_value=18, max_value=100, value=int(defaults["person_age"]))
            income = left.number_input("Annual income ($)", min_value=1000.0, value=float(defaults["person_income"]), step=1000.0)
            home = left.selectbox("Home ownership", HOME_OWNERSHIP_VALUES, index=_index(HOME_OWNERSHIP_VALUES, defaults["person_home_ownership"]))
            employment = left.number_input("Employment length (years)", min_value=0.0, max_value=60.0, value=float(defaults["person_emp_length"]))
            intent = right.selectbox("Loan intent", LOAN_INTENT_VALUES, index=_index(LOAN_INTENT_VALUES, defaults["loan_intent"]))
            grade = right.selectbox("Loan grade", LOAN_GRADE_VALUES, index=_index(LOAN_GRADE_VALUES, defaults["loan_grade"]))
            amount = right.number_input("Loan amount ($)", min_value=500.0, max_value=100000.0, value=float(defaults["loan_amnt"]), step=100.0)
            rate = right.number_input("Interest rate (%)", min_value=0.0, max_value=40.0, value=float(defaults["loan_int_rate"]), step=0.1)
            default_on_file = right.radio("Prior default on file", ["N", "Y"], horizontal=True, index=0 if defaults["cb_person_default_on_file"] == "N" else 1)
            credit_length = right.number_input("Credit history (years)", min_value=0, max_value=60, value=int(defaults["cb_person_cred_hist_length"]))
            submitted = st.form_submit_button("Evaluate Financial Risk", type="primary")
        if submitted:
            record = {
                "person_age": int(age),
                "person_income": float(income),
                "person_home_ownership": home,
                "person_emp_length": float(employment),
                "loan_intent": intent,
                "loan_grade": grade,
                "loan_amnt": float(amount),
                "loan_int_rate": float(rate),
                "cb_person_default_on_file": default_on_file,
                "cb_person_cred_hist_length": int(credit_length),
            }
            _present_score(runtime, record)
    with batch:
        template = pd.DataFrame([defaults])
        st.download_button(
            "Download CSV template",
            template.to_csv(index=False).encode("utf-8"),
            file_name="applicant_template.csv",
            mime="text/csv",
        )
        upload = st.file_uploader("Applicant CSV", type="csv")
        if upload is not None:
            frame = pd.read_csv(upload)
            st.dataframe(frame.head(), use_container_width=True)
            if st.button("Score batch", type="primary"):
                scored = runtime.predict_dataframe(frame)
                high_rate = float((scored["risk_category"] == "High Risk").mean())
                metric_row = st.columns(3)
                metric_row[0].metric("Rows scored", f"{len(scored):,}")
                metric_row[1].metric("Flagged high risk", f"{high_rate:.1%}")
                metric_row[2].metric("Average probability", f"{scored['risk_probability'].mean():.1%}")
                st.dataframe(scored.head(50), use_container_width=True)
                st.download_button(
                    "Download scored CSV",
                    scored.to_csv(index=False).encode("utf-8"),
                    file_name="scored_applicants.csv",
                    mime="text/csv",
                )


def render_explainability() -> None:
    """Global SHAP rankings, a local waterfall, and a what-if rescore."""

    hero("Explainability studio", "Portfolio drivers and the factors behind an individual decision.")
    runtime = require_model()
    summary = shap_summary()
    ranking = summary.get("ranking") or []
    if ranking:
        st.subheader("Global feature importance")
        st.plotly_chart(_importance_bar(ranking), use_container_width=True)
        samples = shap_samples()
        if samples.get("points"):
            st.plotly_chart(_beeswarm(samples), use_container_width=True)
    else:
        st.info("Global SHAP artifacts appear after a tree-based champion is trained.")
    st.subheader("What-if scenario")
    st.caption("Hold the rest of the application constant and see how the default probability moves.")
    if runtime is None:
        return
    base = dict(st.session_state.get("frp_applicant") or sample_applicant())
    income_shift = st.slider(
        "Annual income",
        min_value=8000,
        max_value=250000,
        value=int(min(max(float(base.get("person_income", 65000)), 8000), 250000)),
        step=1000,
    )
    amount_shift = st.slider(
        "Loan amount",
        min_value=500,
        max_value=35000,
        value=int(min(max(float(base.get("loan_amnt", 12000)), 500), 35000)),
        step=100,
    )
    if st.button("Rescore scenario"):
        scenario = dict(base)
        scenario["person_income"] = float(income_shift)
        scenario["loan_amnt"] = float(amount_shift)
        _store_score(runtime, scenario)
    elif "frp_explanation" not in st.session_state:
        _store_score(runtime, base)
    st.subheader("Individual explanation")
    if st.session_state.get("frp_explanation"):
        _render_explanation(st.session_state["frp_explanation"], st.session_state.get("frp_score"))


def _store_score(runtime: Any, record: dict[str, Any]) -> None:
    explanation = runtime.explain_dataframe(pd.DataFrame([record]))[0]
    probability = float(explanation["risk_probability"])
    st.session_state["frp_applicant"] = record
    st.session_state["frp_score"] = {
        "risk_probability": probability,
        "risk_score": int(min(1000, max(0, round(probability * 1000)))),
        "risk_tier": explanation["risk_tier"],
        "decision_threshold": runtime.threshold,
        "recommendation": explanation["recommendation"],
        "risk_category": explanation["risk_category"],
    }
    st.session_state["frp_explanation"] = explanation


def _present_score(runtime: Any, record: dict[str, Any], store_only: bool = False) -> None:
    _store_score(runtime, record)
    if not store_only:
        _render_explanation(st.session_state["frp_explanation"], st.session_state["frp_score"])


def _render_explanation(explanation: dict[str, Any], score: dict[str, Any] | None) -> None:
    probability = float(explanation["risk_probability"])
    category = explanation["risk_category"]
    badge_class = "badge-high" if category == "High Risk" else "badge-low"
    recommendation = explanation.get("recommendation") or (score or {}).get("recommendation", "")
    st.markdown(
        f"<span class='badge {badge_class}'>{category} · {recommendation}</span>",
        unsafe_allow_html=True,
    )
    st.caption(recommendation_detail(str(recommendation)) if recommendation else "")
    left, right = st.columns([1, 1])
    left.plotly_chart(_gauge(probability), use_container_width=True)
    if score:
        right.metric("Risk score", f"{int(score.get('risk_score', probability * 1000))} / 1000")
        right.metric("Tier", str(score.get("risk_tier", explanation.get("risk_tier", "—"))))
        right.metric("Threshold", f"{float(score.get('decision_threshold', 0)):.2f}")
    st.write(explanation.get("adverse_action_notice") or explanation.get("summary_statement"))
    factor_cols = st.columns(2)
    factor_cols[0].markdown("**Factors increasing risk**")
    for item in explanation.get("key_risk_factors") or ["None dominant"]:
        factor_cols[0].write(f"• {item}")
    factor_cols[1].markdown("**Factors reducing risk**")
    for item in explanation.get("positive_credit_factors") or ["None material"]:
        factor_cols[1].write(f"• {item}")
    contributions = explanation.get("shap_contributions") or []
    if contributions:
        st.plotly_chart(_waterfall(contributions), use_container_width=True)


def _leaderboard_frame(models: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(models)
    columns = [
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "optimal_threshold",
        "expected_cost",
        "is_champion",
    ]
    existing = [column for column in columns if column in frame.columns]
    return frame[existing].sort_values("roc_auc", ascending=False)


def _roc_figure(models: list[dict[str, Any]]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash", "color": "#94A3B8"}))
    for item in models:
        figure.add_trace(
            go.Scatter(
                x=item["roc_curve"]["fpr"],
                y=item["roc_curve"]["tpr"],
                mode="lines",
                name=f"{item['model_name']} ({item['roc_auc']:.3f})",
                line={"color": MODEL_COLORS.get(item["model_name"], BRAND)},
            )
        )
    figure.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate", template="plotly_white")
    return figure


def _pr_figure(models: list[dict[str, Any]]) -> go.Figure:
    figure = go.Figure()
    for item in models:
        figure.add_trace(
            go.Scatter(
                x=item["pr_curve"]["recall_curve"],
                y=item["pr_curve"]["precision_curve"],
                mode="lines",
                name=f"{item['model_name']} ({item['pr_auc']:.3f})",
                line={"color": MODEL_COLORS.get(item["model_name"], BRAND)},
            )
        )
    figure.update_layout(title="Precision-recall curves", xaxis_title="Recall", yaxis_title="Precision", template="plotly_white")
    return figure


def _confusion_figure(model: dict[str, Any]) -> go.Figure:
    matrix = model["confusion_matrix"]
    values = [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]]
    figure = go.Figure(
        data=go.Heatmap(
            z=values,
            x=["Predicted low risk", "Predicted high risk"],
            y=["Actual low risk", "Actual high risk"],
            text=values,
            texttemplate="%{text}",
            colorscale=[[0, "#EEF2FF"], [1, BRAND]],
            showscale=False,
        )
    )
    figure.update_layout(title=f"{model['model_name']} at threshold {model['optimal_threshold']:.2f}", template="plotly_white")
    return figure


def _cost_figure(model: dict[str, Any], chosen: float) -> go.Figure:
    curve = model["cost_curve"]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[point["threshold"] for point in curve],
            y=[point["expected_cost"] for point in curve],
            mode="lines",
            name="Expected cost",
            line={"color": BRAND},
        )
    )
    figure.add_vline(x=model["optimal_threshold"], line_dash="dash", line_color=HIGH)
    figure.add_vline(x=chosen, line_dash="dot", line_color=AMBER)
    figure.update_layout(title="Expected portfolio cost", xaxis_title="Threshold", yaxis_title="1·FP + 5·FN", template="plotly_white")
    return figure


def _gauge(probability: float) -> go.Figure:
    color = LOW if probability < 0.25 else AMBER if probability < 0.5 else HIGH
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Predicted default probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 25], "color": "#D1FAE5"},
                    {"range": [25, 50], "color": "#FEF3C7"},
                    {"range": [50, 100], "color": "#FEE2E2"},
                ],
            },
        )
    )
    figure.update_layout(height=280, margin={"t": 40, "b": 10, "l": 20, "r": 20}, template="plotly_white")
    return figure


def _importance_bar(ranking: list[dict[str, Any]]) -> go.Figure:
    top = list(reversed(ranking[:12]))
    figure = go.Figure(
        go.Bar(
            x=[item["mean_abs_shap"] for item in top],
            y=[str(item["feature"]).replace("_", " ") for item in top],
            orientation="h",
            marker_color=BRAND,
        )
    )
    figure.update_layout(title="Mean absolute SHAP", template="plotly_white", xaxis_title="Mean |SHAP|")
    return figure


def _beeswarm(samples: dict[str, Any]) -> go.Figure:
    features = samples.get("features") or []
    figure = go.Figure()
    for feature_index, feature in enumerate(features):
        shap_values = []
        colors = []
        for point in samples["points"]:
            shap_values.append(point["shap"].get(feature, 0.0))
            colors.append(point["color"].get(feature, 0.0))
        figure.add_trace(
            go.Scatter(
                x=shap_values,
                y=[feature_index + ((index % 7) - 3) * 0.03 for index in range(len(shap_values))],
                mode="markers",
                name=feature.replace("_", " "),
                marker={"color": colors, "colorscale": "RdBu_r", "size": 7, "showscale": feature_index == 0},
                hovertemplate="%{x:.3f}<extra>" + feature + "</extra>",
            )
        )
    figure.update_layout(
        title="SHAP beeswarm (sample of the holdout)",
        yaxis={"tickmode": "array", "tickvals": list(range(len(features))), "ticktext": [feature.replace("_", " ") for feature in features]},
        template="plotly_white",
        xaxis_title="SHAP value (positive pushes toward default)",
        showlegend=False,
        height=480,
    )
    return figure


def _waterfall(contributions: list[dict[str, Any]]) -> go.Figure:
    ranked = sorted(contributions, key=lambda item: abs(float(item["shap_value"])), reverse=True)[:8]
    labels = [str(item.get("display_name") or item["feature"]) for item in ranked]
    values = [float(item["shap_value"]) for item in ranked]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[HIGH if value > 0 else LOW for value in values],
        )
    )
    figure.update_layout(
        title="Local contributions (positive increases default risk)",
        template="plotly_white",
        yaxis={"autorange": "reversed"},
        height=380,
    )
    return figure


def _index(options: tuple[str, ...] | list[str], value: str) -> int:
    try:
        return list(options).index(value)
    except ValueError:
        return 0
