"""Plotly figures for exploratory analysis of the credit-risk dataset."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import TARGET_COLUMN

LOW_RISK = "#10B981"
HIGH_RISK = "#EF4444"
BRAND = "#4F46E5"
INK = "#0F172A"


def _apply_layout(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK, "family": "Segoe UI, sans-serif"},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return figure


def target_donut(frame: pd.DataFrame) -> go.Figure:
    """Donut of low-risk versus high-risk labels.

    Args:
        frame: Table containing ``loan_status``.

    Returns:
        Plotly donut figure.
    """

    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Low Risk" if int(key) == 0 else "High Risk" for key in counts.index]
    colors = [LOW_RISK if int(key) == 0 else HIGH_RISK for key in counts.index]
    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=counts.to_numpy(),
                hole=0.62,
                marker={"colors": colors},
                textinfo="label+percent",
            )
        ]
    )
    return _apply_layout(figure, "Portfolio risk mix")


def missingness_bar(frame: pd.DataFrame) -> go.Figure:
    """Bar chart of missing-value rates by column.

    Args:
        frame: Table to profile. Raw data preserves the benchmark missingness.

    Returns:
        Plotly bar figure.
    """

    rates = (frame.isna().mean() * 100).sort_values(ascending=False)
    rates = rates[rates > 0]
    figure = px.bar(
        x=rates.to_numpy(),
        y=[str(name) for name in rates.index],
        orientation="h",
        labels={"x": "Missing (%)", "y": "Column"},
        color_discrete_sequence=[BRAND],
    )
    return _apply_layout(figure, "Missing value profile")


def distribution_figure(frame: pd.DataFrame, column: str, log_x: bool = False) -> go.Figure:
    """Histogram of a numeric column colored by risk label when available.

    Args:
        frame: Source table.
        column: Numeric column to plot.
        log_x: Apply a log scale to the x-axis.

    Returns:
        Plotly histogram.
    """

    plot_frame = frame.copy()
    if TARGET_COLUMN in plot_frame.columns:
        plot_frame["Risk"] = plot_frame[TARGET_COLUMN].map({0: "Low Risk", 1: "High Risk"})
        figure = px.histogram(
            plot_frame,
            x=column,
            color="Risk",
            barmode="overlay",
            nbins=40,
            opacity=0.75,
            color_discrete_map={"Low Risk": LOW_RISK, "High Risk": HIGH_RISK},
            log_x=log_x,
        )
    else:
        figure = px.histogram(plot_frame, x=column, nbins=40, color_discrete_sequence=[BRAND], log_x=log_x)
    return _apply_layout(figure, f"Distribution of {column}")


def correlation_heatmap(frame: pd.DataFrame, method: str = "pearson") -> go.Figure:
    """Interactive correlation heatmap for numeric columns.

    Args:
        frame: Source table.
        method: ``pearson`` or ``spearman``.

    Returns:
        Plotly heatmap.
    """

    numeric = frame.select_dtypes(include="number")
    correlation = numeric.corr(method=method)
    figure = px.imshow(
        correlation,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels={"color": method.title()},
    )
    return _apply_layout(figure, f"{method.title()} correlation")


def box_by_risk(frame: pd.DataFrame, column: str) -> go.Figure:
    """Box plot of a numeric feature split by risk label.

    Args:
        frame: Table with ``loan_status`` and ``column``.
        column: Numeric feature.

    Returns:
        Plotly box figure.
    """

    plot_frame = frame.copy()
    plot_frame["Risk"] = plot_frame[TARGET_COLUMN].map({0: "Low Risk", 1: "High Risk"})
    figure = px.box(
        plot_frame,
        x="Risk",
        y=column,
        color="Risk",
        color_discrete_map={"Low Risk": LOW_RISK, "High Risk": HIGH_RISK},
    )
    return _apply_layout(figure, f"{column} by risk category")


def intent_bar(frame: pd.DataFrame) -> go.Figure:
    """Count of applications by loan intent and risk label.

    Args:
        frame: Table with ``loan_intent``.

    Returns:
        Plotly bar figure.
    """

    plot_frame = frame.copy()
    if TARGET_COLUMN in plot_frame.columns:
        plot_frame["Risk"] = plot_frame[TARGET_COLUMN].map({0: "Low Risk", 1: "High Risk"})
        figure = px.histogram(
            plot_frame,
            x="loan_intent",
            color="Risk",
            barmode="group",
            color_discrete_map={"Low Risk": LOW_RISK, "High Risk": HIGH_RISK},
        )
    else:
        figure = px.histogram(plot_frame, x="loan_intent", color_discrete_sequence=[BRAND])
    return _apply_layout(figure, "Applications by loan intent")
