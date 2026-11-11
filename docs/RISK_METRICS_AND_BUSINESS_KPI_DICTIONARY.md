# Financial Risk Metrics & Business KPI Dictionary

## Project: Financial Risk Prediction Platform (`Project 3`)
**Version:** 1.0  
**Target Reference:** Technical, Data Science, and Financial Underwriting Teams  

---

## 1. Overview & Purpose

This dictionary defines every financial risk ratio, predictive score, statistical metric, and operational key performance indicator (KPI) used across the platform. It standardizes naming conventions, mathematical formulas, acceptable thresholds, and business interpretations to ensure consistent implementation across feature engineering, model evaluation, API payloads, and dashboard reporting.

---

## 2. Core Financial Risk Indicators (Feature Domain)

| Metric / Feature Name | Formula / Definition | Unit / Scale | Benchmark Thresholds | Risk Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Debt-to-Income (DTI)** | $\frac{\text{Total Annual Debt Obligations}}{\text{Annual Gross Income}}$ | Ratio / % ($0.0 - 5.0$) | $\le 0.36$: Prime<br>$0.36 - 0.43$: Acceptable<br>$> 0.43$: High Risk | Higher DTI indicates excessive leverage; applicant is vulnerable to income loss or interest spikes. |
| **Loan-to-Income (LTI)** | $\frac{\text{Requested Principal Amount}}{\text{Annual Gross Income}}$ | Ratio / % ($0.0 - 2.0$) | $\le 0.15$: Low Risk<br>$0.15 - 0.30$: Moderate<br>$> 0.30$: High Risk | Measures whether the requested borrowing scale is disproportionate to annual cash generation. |
| **Revolving Utilization** | $\frac{\text{Total Revolving Credit Balances}}{\text{Total Credit Card Limits}}$ | Ratio / % ($0.0 - 1.5$) | $\le 0.30$: Ideal<br>$0.30 - 0.70$: Moderate<br>$> 0.70$: High Stress | Heavy credit card borrowing reflects acute cash flow deficits or liquidity strain. |
| **Payment-to-Income (PTI)** | $\frac{\text{Monthly New Loan Amortization}}{\text{Monthly Gross Income}}$ | Ratio / % ($0.0 - 1.0$) | $\le 0.10$: Low Burden<br>$0.10 - 0.20$: Moderate<br>$> 0.20$: Stressed | Assesses immediate monthly cash flow absorption caused by approving this specific loan. |
| **Disposable Income** | $\frac{\text{Income} - 0.35 \times \text{Debt}}{12}$ | USD (\$) / Month | $> \$3,000$: Comfortable<br>$\$1,000 - \$3,000$: Tight<br>$< \$1,000$: Deficit Risk | Estimated uncommitted monthly liquidity available to absorb living expenses and emergencies. |
| **Delinquency Severity Index (DSI)** | $\frac{\text{Delinquencies in 2 Yrs} \times 10}{\text{Credit History Length} + 1}$ | Numeric Index ($0 - 50$) | $0.0$: Spotless<br>$1.0 - 5.0$: Mild<br>$> 5.0$: Severe Velocity | Normalizes past-due payment counts against tenure. A new borrower with recent late payments is exceptionally high risk. |
| **Financial Stress Score (FSCS)** | $0.30 \times \text{Util} + 0.30 \times \text{DTI} + 0.25 \times \frac{\text{Rate}}{30} + 0.15 \times \text{DefaultFlag}$ | Score ($0.0 - 1.0$) | $< 0.30$: Sound<br>$0.30 - 0.55$: Moderate<br>$> 0.55$: Critical Distress | Holistic multi-factor index combining leverage, liquidity usage, cost of debt, and historical default flag. |
| **Employment Stability Index** | $\frac{\text{Years Employed} + 1}{\text{Credit History Length} + 1}$ | Ratio ($0.0 - 10.0$) | $> 0.50$: Stable Career<br>$< 0.20$: Frequent Transitions | Higher tenure indicates career and income continuity. |

---

## 3. Machine Learning & Statistical Performance Metrics

| Evaluation Metric | Mathematical Formula | Optimal Value | Business Target | Real-World Financial Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | $1.0$ (100%) | $\ge 82.0\%$ | Percentage of all credit decisions (both approvals and denials) that were correct. |
| **Recall (Sensitivity / TPR)** | $\frac{TP}{TP + FN}$ | $1.0$ (100%) | $\ge 75.0\%$ | **Primary Risk Metric:** Percentage of all actual defaulters caught and flagged before approval. |
| **Precision (PPV)** | $\frac{TP}{TP + FP}$ | $1.0$ (100%) | $\ge 68.0\%$ | Percentage of flagged high-risk applicants who would actually default. Prevents over-rejection of good customers. |
| **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | $1.0$ | $\ge 0.72$ | Harmonic balance between catching defaults and avoiding unnecessary customer friction. |
| **ROC-AUC** | $\int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$ | $1.0$ | $\ge 0.85$ | Model's ability to rank a randomly chosen defaulter higher than a randomly chosen non-defaulter. |
| **Precision-Recall AUC (PR-AUC)** | Area under $P(R)$ curve | $1.0$ | $\ge 0.70$ | Evaluates positive (minority default) class performance without distortion from the large negative class. |
| **Brier Score** | $\frac{1}{N} \sum (p_i - y_i)^2$ | $0.0$ | $\le 0.12$ | Measures accuracy of probabilistic risk forecasts (calibration). Essential for calculating risk-adjusted interest rates. |

---

## 4. Underwriting & Business Portfolio KPIs

| Executive Portfolio KPI | Calculation Formula | Benchmark Range | Strategic Action Trigger |
| :--- | :--- | :---: | :--- |
| **Portfolio Default Rate (DR)** | $\frac{\text{Total Charge-Offs (\#)}}{\text{Total Loans Originated (\#)}}$ | $2.5\% - 4.5\%$ (Annualized) | If $\text{DR} > 5.0\%$, tighten classification threshold $\tau$ from $0.40 \rightarrow 0.32$. |
| **Adverse Action / Rejection Rate** | $\frac{\text{Predicted High-Risk Loans}}{\text{Total Incoming Applications}}$ | $18.0\% - 25.0\%$ | If rejection rate $> 35\%$, re-calibrate score cutoffs to avoid losing market share. |
| **Expected Portfolio Loss (EL)** | $\sum \left( \text{Probability of Default (PD)} \times \text{Exposure at Default (EAD)} \times \text{Loss Given Default (LGD)} \right)$ | Varies by portfolio size | Used by treasury to set aside regulatory capital reserves under Basel III/IV. |
| **Average Loan Spread** | $\text{Weighted Avg Loan Int Rate} - \text{Cost of Funds}$ | $5.0\% - 9.0\%$ | Ensures interest margin adequately covers expected portfolio credit risk. |
| **Cost of False Negative** | $\$10,000$ per missed default | Fixed cost model | Represents unrecovered principal, recovery legal fees, and administrative write-offs. |
| **Cost of False Positive** | $\$500$ per unnecessary denial | Fixed cost model | Represents cost of secondary manual underwriting review and lost lifetime customer value. |
