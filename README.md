# Credit Risk Model

## 📊 Credit Scoring Business Understanding

### 1. Impact of Basel II on Model Interpretability

The Basel II Capital Accord emphasizes that financial institutions must actively measure and manage their exposure to credit risk using sound, quantitative methods. This includes maintaining transparency, explainability, and auditability in their credit scoring systems. 

In practice, this means any predictive model used to assess borrower creditworthiness must be interpretable. Financial regulators and internal compliance teams must be able to understand and justify why a model made a specific decision—especially in cases of loan rejection. As a result, our model must not only be accurate, but also well-documented and explainable. Clear variable definitions, interpretable feature effects (e.g., via Weight of Evidence or SHAP values), and consistent behavior across time are critical for regulatory acceptance and trust in the model.

---

### 2. Why a Proxy Target Is Necessary and the Risks Involved

The provided dataset lacks a direct "default" label, meaning we do not have ground truth data that tells us whether a customer failed to repay a loan. To solve this, we must define a **proxy target variable**—a stand-in indicator of high-risk behavior.

In this project, we use RFM (Recency, Frequency, Monetary) analysis to segment customers based on their engagement patterns. Customers who transact infrequently, spend less, and haven’t transacted recently may be grouped into a "high-risk" cluster, which we label as 1. Others are labeled 0 (low risk).

However, relying on a proxy introduces business risks:
- **False Positives**: We may misclassify good customers as high-risk, leading to lost business opportunities.
- **False Negatives**: We may approve credit for customers who will default, increasing financial losses.
- **Bias**: The proxy may reflect engagement behavior more than actual financial behavior, introducing bias against certain customer groups.

Hence, our proxy must be defined with business logic and carefully evaluated to avoid long-term damage to customer trust and bank profitability.

---

### 3. Trade-offs Between Interpretable and Complex Models

In credit risk modeling—especially within regulated financial environments—there is often a trade-off between interpretability and predictive performance.

**Simple models**, such as Logistic Regression combined with Weight of Evidence (WoE) encoding, offer:
- High transparency: it's easy to explain decisions.
- Regulatory friendliness: easier to audit and justify to compliance teams.
- Robustness: lower risk of overfitting and easier monitoring in production.

**Complex models**, such as Gradient Boosting Machines (GBM), provide:
- Higher predictive power: can capture non-linear patterns in the data.
- Better handling of variable interactions and missing data.

However, complex models are often seen as "black boxes" and may require additional tools like SHAP or LIME to provide post-hoc explanations. These explanations, while helpful, may not always meet regulatory requirements for auditability.

In regulated settings like banking, it's often best to start with an interpretable baseline model. Once trust and performance are established, more complex models can be introduced with sufficient documentation and explainability tools.

---


## Structure
- `data/`: Raw and processed data (not tracked by git)
- `notebooks/`: EDA and experiment tracking
- `src/`: Source code (data processing, training, prediction, API)
- `tests/`: Unit tests

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run tests: `pytest tests/`
3. Start API: `uvicorn src.api.main:app --reload`

See each file for more details.
