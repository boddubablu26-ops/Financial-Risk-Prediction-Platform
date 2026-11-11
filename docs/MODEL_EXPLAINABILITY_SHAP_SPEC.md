# Model Explainability & SHAP Engineering Specification

## Project: Financial Risk Prediction Platform (`Project 3`)
**Version:** 1.0  
**Module Target:** `src/explainability/` (`shap_engine.py`, `driver_generator.py`)  
**Integration:** FastAPI (`api/main.py`), Streamlit (`dashboard/app.py`)  

---

## 1. Overview & Regulatory Rationale

In high-stakes financial domains such as credit underwriting and risk classification, black-box predictions are unacceptable and often illegal. 

Under the **Fair Credit Reporting Act (FCRA)** and the **Equal Credit Opportunity Act (ECOA)**:
1. Borrowers subjected to adverse credit decisions (loan denial, lower credit limit, higher interest rate) are entitled to an **Adverse Action Notice** detailing the principal reasons for the decision.
2. Internal risk auditors and regulators require verifiable mathematical proof that credit scoring models do not rely on prohibited proxies or exhibit uncalibrated bias.

This module specifies an enterprise-grade explainability framework powered by **SHAP (SHapley Additive exPlanations)**, providing mathematically grounded feature attributions at both **portfolio-wide (global)** and **applicant-specific (local)** granularities, coupled with an automated **Natural Language Risk Driver Generator**.

---

## 2. Mathematical Foundation: Shapley Values

SHAP unifies cooperative game theory with machine learning by calculating the marginal contribution of each feature across all possible feature subsets:

$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$

Where:
* $F$ is the complete set of features.
* $S$ is a subset of features excluding feature $i$.
* $f(S)$ is the model prediction conditioned on feature subset $S$.
* $\phi_i(x)$ is the Shapley value (additive log-odds or probability contribution) for feature $i$ on record $x$.

### Additive Efficiency Property
The sum of all SHAP values plus the baseline expected value $\mathbb{E}[f(x)]$ equals the individual applicant prediction:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
Where $\phi_0 = \mathbb{E}[f(X)]$ is the average log-odds risk across the training population.

---

## 3. Global Explainability Specifications

Global explainability reveals the overall logic, risk sensitivity, and macroeconomic structure learned by the model across the portfolio.

```mermaid
flowchart LR
    A[Test Dataset Partition] --> B[Fitted Champion Model Pipeline]
    B --> C[SHAP TreeExplainer / KernelExplainer]
    C --> D[1. Mean |SHAP| Importance Bar Chart]
    C --> E[2. Beeswarm Summary Plot (Directional Impact)]
    C --> F[3. Partial Dependence / Interaction Curves]
```

### 3.1 Mean Absolute SHAP Importance Bar Chart
* **Metric:** $I_j = \frac{1}{N} \sum_{k=1}^N |\phi_j(x_k)|$.
* **Visualization:** Horizontal bar chart ranking features by descending average impact on risk prediction.
* **Interpretation:** Unambiguously identifies the top drivers of financial risk across the institution's portfolio.

### 3.2 Beeswarm Summary Plot
* Displays every individual test record as a dot for each feature.
* **X-axis:** SHAP value (positive values shift prediction toward High Risk; negative values shift toward Low Risk).
* **Color:** Feature value magnitude (Red = High value, Blue = Low value).
* **Financial Insight:** Instantly illustrates directional impact (e.g., High `debt_to_income` is red and pushes strongly to the right $\rightarrow$ increases default probability).

---

## 4. Local Explainability Specifications (Per-Applicant Decision)

Every prediction evaluated in the dashboard or via API must provide an interpretable local breakdown.

### 4.1 SHAP Waterfall Plot
The Waterfall plot visually decomposes how a specific applicant's baseline portfolio probability ($\phi_0 \approx 21.8\%$) was adjusted upward or downward to arrive at their final risk score:

```text
E[f(x)] = 0.218 (Portfolio Baseline Risk)
  + 0.280  [debt_to_income = 0.52]           -----> Pushes Risk UP
  + 0.185  [delinq_severity_index = 8.5]      -----> Pushes Risk UP
  + 0.110  [revolving_utilization = 0.89]     -----> Pushes Risk UP
  - 0.095  [person_income = $85,000]          <----- Pushes Risk DOWN
  - 0.040  [loan_grade = B]                   <----- Pushes Risk DOWN
---------------------------------------------------------------------
f(x)    = 0.658 (Final Predicted Risk Probability = 65.8% -> HIGH RISK)
```

### 4.2 Waterfall Rendering
Implemented via `plotly` or `matplotlib` with color coding:
* **Crimson Red (`#dc3545`):** Positive SHAP contribution (increases financial risk).
* **Forest Green (`#28a745`):** Negative SHAP contribution (mitigates financial risk, promotes creditworthiness).

---

## 5. Natural Language Risk Driver Generator Engine

Analysts and loan officers cannot be expected to interpret raw log-odds or KaTeX math during underwriting conversations. The **Natural Language Risk Driver Generator** (`src/explainability/driver_generator.py`) automatically translates top positive and negative SHAP values into clean, compliant plain-English narratives.

### 5.1 Linguistic Mapping Dictionary

| Feature Name | High Value Risk Description ($x > \mu$) | Low Value Risk Description ($x < \mu$) | Mitigating Factor Description |
| :--- | :--- | :--- | :--- |
| `debt_to_income` | "High debt-to-income ratio ({val:.1%})" | — | "Conservative debt-to-income ratio ({val:.1%})" |
| `delinq_2yrs` / `delinq_severity_index` | "{val:.0f} recent delinquent credit payment(s)" | — | "Clean delinquency history with zero past-due accounts" |
| `revolving_utilization` | "Elevated credit line utilization ({val:.1%})" | — | "Low revolving credit utilization ({val:.1%})" |
| `person_income` | — | "Limited annual income (${val:,.0f})" | "Strong annual income (${val:,.0f})" |
| `cb_person_cred_hist_length` | — | "Short credit bureau history ({val:.0f} yrs)" | "Established multi-year credit bureau tenure ({val:.0f} yrs)" |
| `loan_percent_income` | "Requested loan represents high proportion of income ({val:.1%})" | — | "Modest loan amount relative to income ({val:.1%})" |
| `cb_person_default_on_file` | "Historical credit default recorded on bureau file" | — | "Zero prior credit defaults on file" |
| `total_outstanding_debt` | "Substantial outstanding liabilities (${val:,.0f})" | — | "Minimal outstanding debt liabilities" |

### 5.2 Algorithmic Implementation

```python
from typing import Dict, List, Any
import numpy as np

class NaturalLanguageRiskExplainer:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def explain_instance(
        self,
        raw_record: Dict[str, Any],
        shap_values: np.ndarray,
        predicted_prob: float,
        risk_category: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Converts SHAP contributions into structured natural language reasons."""
        
        # Pair feature names with their directional SHAP values
        contributions = []
        for name, shap_val in zip(self.feature_names, shap_values):
            contributions.append({
                "feature": name,
                "shap_value": float(shap_val),
                "raw_value": raw_record.get(name)
            })
            
        # Sort by positive risk drivers (increasing risk) and mitigating factors
        positive_drivers = sorted(
            [c for c in contributions if c["shap_value"] > 0],
            key=lambda x: x["shap_value"],
            reverse=True
        )[:top_k]
        
        mitigating_factors = sorted(
            [c for c in contributions if c["shap_value"] < 0],
            key=lambda x: abs(x["shap_value"]),
            reverse=True
        )[:top_k]
        
        # Translate to human text
        key_risk_factors = [self._format_risk_phrase(c) for c in positive_drivers]
        positive_credit_factors = [self._format_mitigating_phrase(c) for c in mitigating_factors]
        
        summary_sentence = (
            f"Applicant classified as {risk_category} ({predicted_prob:.1%} probability). "
            f"Key risk drivers: {', '.join(key_risk_factors) if key_risk_factors else 'None significant'}."
        )
        
        return {
            "risk_category": risk_category,
            "risk_probability": round(predicted_prob, 4),
            "summary_statement": summary_sentence,
            "key_risk_factors": key_risk_factors,
            "positive_credit_factors": positive_credit_factors,
            "raw_contributions": contributions
        }

    def _format_risk_phrase(self, item: Dict[str, Any]) -> str:
        feat = item["feature"]
        val = item["raw_value"]
        if "debt_to_income" in feat and val is not None:
            return f"High debt-to-income ratio ({val:.1%})"
        elif "delinq" in feat and val is not None:
            return f"{int(val)} recent delinquent payment(s)"
        elif "utilization" in feat and val is not None:
            return f"High credit utilization ({val:.1%})"
        elif "loan_percent_income" in feat and val is not None:
            return f"Loan amount consumes {val:.1%} of annual income"
        elif "default_on_file" in feat:
            return "Prior credit default on file"
        elif "total_outstanding_debt" in feat and val is not None:
            return f"High outstanding liabilities (${val:,.0f})"
        elif "income" in feat and val is not None:
            return f"Low annual income (${val:,.0f})"
        elif "cred_hist" in feat and val is not None:
            return f"Short credit history ({val} yrs)"
        return f"Adverse {feat.replace('_', ' ')}"

    def _format_mitigating_phrase(self, item: Dict[str, Any]) -> str:
        feat = item["feature"]
        val = item["raw_value"]
        if "income" in feat and val is not None:
            return f"Strong annual income (${val:,.0f})"
        elif "debt_to_income" in feat and val is not None:
            return f"Low debt-to-income ratio ({val:.1%})"
        elif "cred_hist" in feat and val is not None:
            return f"Established credit tenure ({val} yrs)"
        elif "utilization" in feat and val is not None:
            return f"Conservative credit utilization ({val:.1%})"
        return f"Favorable {feat.replace('_', ' ')}"
```

---

## 6. Computational Optimization for Real-Time Inference

### 6.1 TreeExplainer vs KernelExplainer
* **Tree Models (XGBoost, Random Forest):** Must use `shap.TreeExplainer(model, feature_perturbation='interventional')`. Latency is $< 15\text{ms}$ per applicant.
* **Kernel Models (SVM, Linear):** KernelExplainer evaluates $2^{|F|}$ combinations and can take $> 5$ seconds. For non-tree models, downsample background dataset to $N = 100$ representative medoids using `shap.kmeans(X_train, 100)`.

### 6.2 Pre-Calculation & Background Caching
The background reference distribution `X_background_sample.parquet` ($N=200$) is computed once during model serialization and persisted in `models/`. The API loads this reference set into memory at startup to achieve $< 50\text{ms}$ latency on `/api/v1/explain/{id}` requests.
