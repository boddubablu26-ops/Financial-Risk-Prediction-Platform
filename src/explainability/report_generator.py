"""Narrative report helpers built on the natural-language explainer."""

from __future__ import annotations

from typing import Any

from src.explainability.driver_generator import (
    NaturalLanguageRiskExplainer,
    build_adverse_action_notice,
)


def render_explanation(explanation: dict[str, Any]) -> str:
    """Render a multi-line analyst note from an explanation payload.

    Args:
        explanation: Output of ``NaturalLanguageRiskExplainer.explain_instance``.

    Returns:
        Plain-text report.
    """

    notice = build_adverse_action_notice(
        explanation["risk_category"],
        explanation["risk_probability"],
        explanation["key_risk_factors"],
    )
    mitigating = explanation.get("positive_credit_factors") or []
    mitigating_text = "; ".join(mitigating) if mitigating else "None material."
    return (
        f"{notice}\n"
        f"Summary: {explanation.get('summary_statement', '')}\n"
        f"Mitigating factors: {mitigating_text}"
    )


__all__ = [
    "NaturalLanguageRiskExplainer",
    "build_adverse_action_notice",
    "render_explanation",
]
