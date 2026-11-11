"""Plain-language translation of SHAP contributions into adverse-action reasons."""

from __future__ import annotations

from typing import Any

from src.config import DISPLAY_NAMES


class NaturalLanguageRiskExplainer:
    """Turn signed feature contributions into analyst-facing reasons."""

    def __init__(self, feature_names: list[str] | None = None) -> None:
        """Store an optional ordered feature list.

        Args:
            feature_names: Names aligned with a contribution vector. Aggregated
                explanations can pass names inside each contribution instead.
        """

        self.feature_names = feature_names or []

    def explain_instance(
        self,
        raw_record: dict[str, Any],
        shap_values: Any,
        predicted_prob: float,
        risk_category: str,
        top_k: int = 3,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Convert contributions into risk and mitigating phrases.

        Args:
            raw_record: Original applicant fields plus any derived ratios.
            shap_values: Signed contribution per feature.
            predicted_prob: Predicted default probability.
            risk_category: ``High Risk`` or ``Low Risk``.
            top_k: Number of reasons to keep on each side.
            feature_names: Names aligned with ``shap_values``.

        Returns:
            Narrative payload with phrases, a summary sentence, and raw contributions.
        """

        names = feature_names or self.feature_names
        values = [float(value) for value in list(shap_values)]
        contributions: list[dict[str, Any]] = []
        for name, shap_value in zip(names, values):
            contributions.append(
                {
                    "feature": name,
                    "shap_value": shap_value,
                    "raw_value": _lookup_value(name, raw_record),
                    "display_name": DISPLAY_NAMES.get(name, name.replace("_", " ")),
                }
            )
        positive = sorted(
            [item for item in contributions if item["shap_value"] > 0],
            key=lambda item: item["shap_value"],
            reverse=True,
        )[:top_k]
        mitigating = sorted(
            [item for item in contributions if item["shap_value"] < 0],
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )[:top_k]
        key_risk_factors = [self._format_risk_phrase(item) for item in positive]
        positive_credit_factors = [self._format_mitigating_phrase(item) for item in mitigating]
        summary = (
            f"Applicant classified as {risk_category} ({predicted_prob:.1%} probability). "
            f"Key risk drivers: {', '.join(key_risk_factors) if key_risk_factors else 'none dominant'}."
        )
        return {
            "risk_category": risk_category,
            "risk_probability": round(float(predicted_prob), 4),
            "summary_statement": summary,
            "key_risk_factors": key_risk_factors,
            "positive_credit_factors": positive_credit_factors,
            "raw_contributions": contributions,
        }

    def _format_risk_phrase(self, item: dict[str, Any]) -> str:
        feature = item["feature"]
        value = item["raw_value"]
        suffix = f" ({item['shap_value']:+.3f})"
        if feature in {"debt_to_income", "loan_to_income_ratio", "loan_percent_income"} and _is_number(value):
            return f"High loan-to-income burden ({float(value):.1%}){suffix}"
        if feature == "payment_to_income" and _is_number(value):
            return f"High payment-to-income ratio ({float(value):.1%}){suffix}"
        if "delinq" in feature and _is_number(value):
            return f"Elevated delinquency severity ({float(value):.2f}){suffix}"
        if "utilization" in feature and _is_number(value):
            return f"Elevated credit utilization ({float(value):.1%}){suffix}"
        if feature in {"default_on_file_binary", "cb_person_default_on_file", "historical_risk_weight"}:
            return f"Prior credit default on file{suffix}"
        if feature == "loan_grade":
            return f"Weaker loan grade ({value}){suffix}"
        if feature == "loan_int_rate" and _is_number(value):
            return f"Elevated interest rate ({float(value):.2f}%){suffix}"
        if feature == "interest_burden_index" and _is_number(value):
            return f"High interest burden ({float(value):.1%} of income){suffix}"
        if "income" in feature and "ratio" not in feature and "percent" not in feature and _is_number(value):
            return f"Limited annual income (${float(value):,.0f}){suffix}"
        if feature == "cred_hist_age_ratio" and _is_number(value):
            return f"Credit history is short relative to age ({float(value):.0%} of age){suffix}"
        if "cred_hist" in feature and _is_number(value):
            return f"Short credit history ({float(value):.0f} years){suffix}"
        if feature == "person_home_ownership":
            return f"Housing status increases risk ({value}){suffix}"
        if feature == "loan_intent":
            return f"Loan purpose increases risk ({value}){suffix}"
        if feature == "financial_stress_score" and _is_number(value):
            return f"High financial stress score ({float(value):.2f}){suffix}"
        if feature == "person_emp_length" and _is_number(value):
            return f"Short employment history ({float(value):.1f} years){suffix}"
        label = item["display_name"]
        return f"Adverse {label.lower()}{suffix}"

    def _format_mitigating_phrase(self, item: dict[str, Any]) -> str:
        feature = item["feature"]
        value = item["raw_value"]
        suffix = f" ({item['shap_value']:+.3f})"
        if "income" in feature and "ratio" not in feature and "percent" not in feature and _is_number(value):
            return f"Strong annual income (${float(value):,.0f}){suffix}"
        if feature in {"debt_to_income", "loan_to_income_ratio", "loan_percent_income"} and _is_number(value):
            return f"Modest loan relative to income ({float(value):.1%}){suffix}"
        if feature == "cred_hist_age_ratio" and _is_number(value):
            return f"Credit history covers a solid share of age ({float(value):.0%}){suffix}"
        if "cred_hist" in feature and _is_number(value):
            return f"Established credit history ({float(value):.0f} years){suffix}"
        if feature in {"default_on_file_binary", "cb_person_default_on_file"}:
            return f"No prior credit default on file{suffix}"
        if feature == "loan_grade":
            return f"Stronger loan grade ({value}){suffix}"
        if feature == "loan_int_rate" and _is_number(value):
            return f"Favorable interest rate ({float(value):.2f}%){suffix}"
        if feature == "person_emp_length" and _is_number(value):
            return f"Stable employment ({float(value):.1f} years){suffix}"
        if feature == "person_home_ownership":
            return f"Housing status reduces risk ({value}){suffix}"
        if feature == "financial_stress_score" and _is_number(value):
            return f"Low financial stress score ({float(value):.2f}){suffix}"
        label = item["display_name"]
        return f"Favorable {label.lower()}{suffix}"


def build_adverse_action_notice(
    risk_category: str,
    probability: float,
    key_risk_factors: list[str],
) -> str:
    """Build the single-paragraph notice used on the dashboard and in API text.

    Args:
        risk_category: ``High Risk`` or ``Low Risk``.
        probability: Predicted default probability.
        key_risk_factors: Ranked plain-language drivers.

    Returns:
        A one-paragraph adverse-action style statement.
    """

    if not key_risk_factors:
        drivers = "no single factor dominated the score"
    else:
        numbered = [f"({index}) {text}" for index, text in enumerate(key_risk_factors, start=1)]
        drivers = "; ".join(numbered)
    return (
        f"Applicant assigned {risk_category} ({probability:.1%} probability) "
        f"primarily due to: {drivers}."
    )


def _lookup_value(feature: str, record: dict[str, Any]) -> Any:
    if feature in record:
        return record[feature]
    aliases = {
        "default_on_file_binary": "cb_person_default_on_file",
        "loan_to_income_ratio": "loan_percent_income",
        "debt_to_income": "loan_percent_income",
    }
    alias = aliases.get(feature)
    if alias and alias in record:
        return record[alias]
    return None


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
